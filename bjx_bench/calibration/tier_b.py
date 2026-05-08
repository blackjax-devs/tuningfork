"""Tier-B per-algorithm tuning via Optuna BO. Foundation layer (T2.6a).

This module owns the *types* and *pure helpers* for the BO loop. The actual
loop body (``tune_algorithm`` body) is implemented in T2.6b.

The tuning-difficulty metric (``PLAN_bjx_bench_API_phase2.md``
§"Tuning Difficulty Metric") is the companion to the headline ESS/grad: it
captures HOW HARD it was to find good HPs, not just what the optimum is.
Trial 0 is special — it uses a deterministic default config (geometric mean
for loguniform, midpoint for uniform, etc.) so "out-of-the-box" performance
is reproducible across seeds.

Design note: ``TuningResult.history`` is typed ``tuple[dict, ...]`` rather
than ``list[dict]`` so that the frozen dataclass remains hash-stable and safe
to store as a value in another frozen container. Callers construct it by
converting their mutable list of per-trial dicts to a tuple at the point of
``TuningResult`` construction; this is cheap (one allocation) and the
immutability is load-bearing for the evaluation pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import optuna
import optuna.distributions

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

__all__ = [
    "TuningDifficulty",
    "TuningResult",
    "default_value_for_space",
    "default_params_for",
    "optuna_distribution_for_space",
    "optuna_distributions_for",
    "tune_algorithm",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningDifficulty:
    """Profile of how hard it was to reach a useful HP setting.

    Per ``PLAN_bjx_bench_API_phase2.md`` §"Tuning Difficulty Metric".

    Parameters
    ----------
    default_score
        Headline metric at trial 0, where trial 0 uses the
        prior-mean / mid-of-range of ``default_hp_space`` (a single
        canonical "default config", deterministic and seed-independent).
    best_score
        Headline metric at the best trial in the BO study.
    threshold_score
        ``max(default_score, 0.5 * best_score)`` — the minimum score that
        counts as "good enough to use".  Edge case: if ``best_score < 0``
        (e.g. all trials diverged), ``threshold_score`` is clamped so that
        ``0.5 * best_score`` is always below ``default_score``, which
        degrades gracefully to ``threshold_score = default_score``.
    default_works
        ``True`` iff ``default_score >= 0.5 * best_score``.  When
        ``True``, out-of-box tuning suffices and no BO is needed.
    n_trials_to_threshold
        First BO trial number (1-indexed, counting from the first
        *non-default* TPE trial) whose score reaches ``threshold_score``.
        **Exactly 0 iff ``default_works`` is True** (no tuning required).
    n_trials_to_best
        Trial number at which the best score is first achieved.
    wall_seconds_to_threshold
        Cumulative wallclock seconds from study start to the
        threshold-passing trial.  0.0 iff ``default_works``.
    wall_seconds_to_best
        Cumulative wallclock seconds from study start to the best trial.
    """

    default_score: float
    best_score: float
    threshold_score: float
    default_works: bool
    n_trials_to_threshold: int
    n_trials_to_best: int
    wall_seconds_to_threshold: float
    wall_seconds_to_best: float


@dataclass(frozen=True)
class TuningResult:
    """Outcome of one Tier-B ``tune_algorithm`` run.

    Parameters
    ----------
    algorithm_name
        Registry name of the algorithm, e.g. ``"hmc"``.
    posterior_name
        Registry name of the posterior target, e.g. ``"mvn_10"``.
    best_params
        Best hyperparameter dict found by the BO study; keys match
        ``HyperparamSpace.name`` fields of the algorithm's
        ``default_hp_space``.
    best_score
        Headline ``min_bulk_ess_per_grad`` at ``best_params``.
    n_trials_completed
        Total Optuna trials run (includes the injected default trial 0).
    n_seeds
        Number of random seeds averaged per trial.
    history
        Immutable tuple of per-trial records.  Each record is a dict with
        keys ``{"trial", "params", "score", "certified", "wall_seconds"}``.
        Stored as a tuple (not list) to preserve frozen-dataclass
        hash-stability; callers convert from a mutable accumulator list at
        construction time.
    difficulty
        Companion tuning-difficulty profile computed from ``history``.
    """

    algorithm_name: str
    posterior_name: str
    best_params: dict[str, Any]
    best_score: float
    n_trials_completed: int
    n_seeds: int
    history: tuple[dict, ...]
    difficulty: TuningDifficulty


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def default_value_for_space(space: HyperparamSpace) -> Any:
    """Deterministic default value for a single ``HyperparamSpace``.

    Used to seed Optuna trial 0 (the "out-of-the-box" reference point).
    The default is chosen so that it sits at the centre of the search
    space on the scale that BO will explore:

    - ``"loguniform"``: geometric mean ``sqrt(low * high)`` — the midpoint
      on log-scale, matching the scale TPE uses internally.
    - ``"uniform"``: arithmetic midpoint ``(low + high) / 2``.
    - ``"int"``: integer midpoint ``(low + high) // 2``.  Integer division
      avoids returning a float for an integer parameter.  For even sums
      this equals ``int((low + high) / 2)``; the two formulas agree for
      all integer inputs because ``(a + b) // 2 == int((a + b) / 2)`` when
      ``a, b`` are integers (Python truncates toward zero and the sum is
      always non-negative here given ``low <= high``).
    - ``"categorical"``: first element of ``choices``.

    Parameters
    ----------
    space
        A single hyperparameter search space descriptor.

    Returns
    -------
    Any
        The default value.  Type matches the kind:
        ``float`` for loguniform/uniform, ``int`` for int,
        the type of ``choices[0]`` for categorical.

    Raises
    ------
    ValueError
        If ``space.kind`` is not one of the four recognised kinds (should
        not happen if ``HyperparamSpace.__post_init__`` validation was run).
    """
    if space.kind == "loguniform":
        # Geometric mean: centre on log-scale
        return math.sqrt(space.low * space.high)  # type: ignore[operator]
    elif space.kind == "uniform":
        return (space.low + space.high) / 2  # type: ignore[operator]
    elif space.kind == "int":
        # Integer midpoint: (low + high) // 2, truncated toward low.
        # Chosen over int((low+high)/2) for explicitness with integer types.
        return (space.low + space.high) // 2  # type: ignore[operator]
    elif space.kind == "categorical":
        return space.choices[0]  # type: ignore[index]
    else:
        raise ValueError(
            f"default_value_for_space: unknown kind '{space.kind}'. "
            "Expected one of 'loguniform', 'uniform', 'int', 'categorical'."
        )


def default_params_for(entry: AlgorithmEntry) -> dict[str, Any]:
    """Map ``default_value_for_space`` across all spaces in ``entry``.

    Returns a ``dict`` keyed by ``HyperparamSpace.name``, ready to pass
    directly to ``optuna.Study.enqueue_trial``.

    Parameters
    ----------
    entry
        An ``AlgorithmEntry`` whose ``default_hp_space`` defines the search
        space to map over.

    Returns
    -------
    dict[str, Any]
        ``{space.name: default_value_for_space(space) for space in entry.default_hp_space}``
    """
    return {
        space.name: default_value_for_space(space) for space in entry.default_hp_space
    }


def optuna_distribution_for_space(
    space: HyperparamSpace,
) -> optuna.distributions.BaseDistribution:
    """Convert a ``HyperparamSpace`` to an Optuna distribution.

    Mapping:

    - ``"loguniform"`` → ``FloatDistribution(low, high, log=True)``
    - ``"uniform"``    → ``FloatDistribution(low, high, log=False)``
    - ``"int"``        → ``IntDistribution(low, high)``
    - ``"categorical"`` → ``CategoricalDistribution(choices)``

    Parameters
    ----------
    space
        A single hyperparameter search space descriptor.

    Returns
    -------
    optuna.distributions.BaseDistribution
        The corresponding Optuna distribution object.

    Raises
    ------
    ValueError
        If ``space.kind`` is not one of the four recognised kinds.
    """
    if space.kind == "loguniform":
        return optuna.distributions.FloatDistribution(
            low=float(space.low),  # type: ignore[arg-type]
            high=float(space.high),  # type: ignore[arg-type]
            log=True,
        )
    elif space.kind == "uniform":
        return optuna.distributions.FloatDistribution(
            low=float(space.low),  # type: ignore[arg-type]
            high=float(space.high),  # type: ignore[arg-type]
            log=False,
        )
    elif space.kind == "int":
        return optuna.distributions.IntDistribution(
            low=int(space.low),  # type: ignore[arg-type]
            high=int(space.high),  # type: ignore[arg-type]
        )
    elif space.kind == "categorical":
        return optuna.distributions.CategoricalDistribution(
            choices=list(space.choices),  # type: ignore[arg-type]
        )
    else:
        raise ValueError(
            f"optuna_distribution_for_space: unknown kind '{space.kind}'. "
            "Expected one of 'loguniform', 'uniform', 'int', 'categorical'."
        )


def optuna_distributions_for(
    entry: AlgorithmEntry,
) -> dict[str, optuna.distributions.BaseDistribution]:
    """Build the dict-of-distributions Optuna's trial API expects.

    Iterates ``entry.default_hp_space`` and calls
    ``optuna_distribution_for_space`` for each space.

    Parameters
    ----------
    entry
        An ``AlgorithmEntry`` whose ``default_hp_space`` defines the search
        space.

    Returns
    -------
    dict[str, optuna.distributions.BaseDistribution]
        ``{space.name: optuna_distribution_for_space(space) for space in entry.default_hp_space}``
        Ready to pass to ``study.add_trial(params=…, distributions=…)``.
    """
    return {
        space.name: optuna_distribution_for_space(space)
        for space in entry.default_hp_space
    }


# ---------------------------------------------------------------------------
# Loop stub — body implemented in T2.6b
# ---------------------------------------------------------------------------


def tune_algorithm(
    posterior_entry: Any,  # PosteriorEntry; quoted to avoid registry import (defer to T2.6b)
    algorithm_entry: AlgorithmEntry,
    *,
    n_trials: int = 50,
    n_seeds: int = 5,
    n_chains: int = 4,
    n_samples: int = 500,
    n_warmup: int = 1000,
    rng_key: Any = None,  # jax.Array — concrete signature deferred to T2.6b
    storage: str | None = None,
) -> TuningResult:
    """Optuna TPE BO over ``algorithm_entry.default_hp_space``.

    NOT IMPLEMENTED YET — T2.6b wires the actual loop. T2.6a establishes
    the public signature and result types so callers (CLI, runner) can be
    written in parallel.

    Parameters
    ----------
    posterior_entry
        A ``PosteriorEntry`` describing the target distribution.  Import
        deferred to T2.6b to avoid a circular dependency.
    algorithm_entry
        The ``AlgorithmEntry`` whose ``default_hp_space`` defines the BO
        search space and whose ``factory`` creates the kernel.
    n_trials
        Total Optuna trials to run (including the injected default trial 0).
    n_seeds
        Number of random seeds to average per trial.
    n_chains
        Number of MCMC chains per seed.
    n_samples
        Post-warmup samples per chain.
    n_warmup
        Warmup steps per chain.
    rng_key
        Base JAX random key; subkeys are folded-in per trial and per seed.
    storage
        Optional Optuna RDB URL (e.g. ``"sqlite:///tuning.db"``) for
        persistent study storage.  ``None`` for in-memory (default).

    Returns
    -------
    TuningResult
        Full tuning outcome including best params, score, history, and
        difficulty profile.

    Raises
    ------
    NotImplementedError
        Always — this stub is replaced in T2.6b.
    """
    raise NotImplementedError(
        "tune_algorithm body lands in Subtask T2.6b. T2.6a only establishes "
        "TuningResult/TuningDifficulty types and the helper functions."
    )
