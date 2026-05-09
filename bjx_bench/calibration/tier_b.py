# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tier-B per-algorithm tuning via Optuna BO. Foundation layer (T2.6a).

Extended in T2.6b (mass-matrix kernels) and T2.6c (sampler-swap, MALA/RWM,
MCLMC dispatch, best_trial robustness guard).

This module owns the *types* and *pure helpers* for the BO loop. The actual
loop body (``tune_algorithm``) is implemented in T2.6b/T2.6c.

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

Fallback behaviour: if all trials diverge, ``study.best_trial`` raises
``ValueError``. The loop catches this and returns trial-0 (the enqueued
default) as the "best" params with score ``-inf``.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optuna
import optuna.distributions
from blackjax.util import run_inference_algorithm

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace
from bjx_bench.metrics.grad_counter import total_grad_evals
from bjx_bench.metrics.headline import min_bulk_ess_per_grad

# Suppress Optuna's verbose INFO logging globally.  Individual tests may
# re-set verbosity to DEBUG if needed, but WARNING is the right default for
# automated runs so that pytest does not treat Optuna INFO lines as captured
# output that pollutes the test report.
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    base_method_name
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

    base_method_name: str
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

    - ``"loguniform"``: 70th-percentile on log-scale
      ``low * (high / low) ** 0.7`` — biased toward the high end of the
      log-range.  For ``step_size [1e-3, 1.0]`` this gives ``0.1`` rather
      than the geometric mean ``0.032``.  Empirically closer to typical
      BO-best step_size values observed in Phase 3 HIGH recipes (P4.0 tweak).
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
        # 70th-percentile on log-scale: low * (high/low)**0.7
        # For step_size [1e-3, 1.0]: 1e-3 * (1e3)**0.7 ≈ 0.1
        # (P4.0 tweak; previously sqrt(low*high) = 50th-percentile ≈ 0.032)
        return space.low * (space.high / space.low) ** 0.7  # type: ignore[operator]
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


def default_params_for(entry: BaseMethod) -> dict[str, Any]:
    """Map ``default_value_for_space`` across all spaces in ``entry``.

    Returns a ``dict`` keyed by ``HyperparamSpace.name``, ready to pass
    directly to ``optuna.Study.enqueue_trial``.

    Parameters
    ----------
    entry
        An ``BaseMethod`` whose ``default_hp_space`` defines the search
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
    entry: BaseMethod,
) -> dict[str, optuna.distributions.BaseDistribution]:
    """Build the dict-of-distributions Optuna's trial API expects.

    Iterates ``entry.default_hp_space`` and calls
    ``optuna_distribution_for_space`` for each space.

    Parameters
    ----------
    entry
        An ``BaseMethod`` whose ``default_hp_space`` defines the search
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
# Private helpers for the BO loop
# ---------------------------------------------------------------------------


def _run_warmup(
    logdensity_fn: Any,
    init_position: Any,
    algorithm_entry: BaseMethod,
    n_warmup: int,
    rng_key: jax.Array,
    warmup_name: str = "stan_window",
) -> tuple[Any, dict[str, Any]]:
    """Run warmup and return (adapted_state, adapted_params).

    Dispatch is via the ``WARMUPS`` registry (Phase 3, P3.1).  The
    ``warmup_name`` selects the warmup procedure; ``tune_algorithm``
    resolves ``None`` to the right default before calling this function.

    Behaviour by warmup:

    - ``"stan_window"`` (mass-matrix kernels — NUTS, HMC, Barker, MALA):
      ``blackjax.window_adaptation`` runs once.  ``adapted_params`` contains
      at least ``step_size`` and ``inverse_mass_matrix``.  These are reused
      across all BO trials; only the BO-tunable HPs (e.g. ``step_size``,
      ``num_integration_steps``) are overridden per trial.
    - ``"no_warmup"`` (MALA, RWM): kernel state is initialised from the
      default HPs; no adaptation is run.  ``adapted_params`` is empty (``{}``);
      all kernel params come from the BO trial.
    - ``"mclmc_tuning"`` (MCLMC): ``blackjax.mclmc_find_L_and_step_size``
      runs for ``n_warmup`` steps.  The returned ``MCLMCAdaptationState``
      (fields: ``L``, ``step_size``, ``inverse_mass_matrix``) is converted to
      a plain ``dict`` and returned.  BO trials override ``step_size`` and
      ``L``; the warmup ``inverse_mass_matrix`` is used as a fixed diagonal
      preconditioner.

    Parameters
    ----------
    logdensity_fn
        BlackJAX-compatible log-density.
    init_position
        Initial parameter dict (unconstrained).
    algorithm_entry
        Registry entry describing the algorithm.
    n_warmup
        Number of warmup steps.
    rng_key
        JAX random key for the warmup run.
    warmup_name
        Key into the ``WARMUPS`` registry.  ``tune_algorithm`` resolves
        ``None`` → the auto-dispatched default before calling here.

    Returns
    -------
    adapted_state
        The BlackJAX state object after warmup (or a freshly initialised
        state for no-warmup algorithms).
    adapted_params
        Dict of warmup-adapted parameters to merge with per-trial BO params.
        Empty dict for ``no_warmup`` (all params come from BO).

    Raises
    ------
    ValueError
        If ``warmup_name`` is not in ``WARMUPS`` or if the selected warmup
        is not compatible with ``algorithm_entry.name``.
    """
    from bjx_bench.inference.warmup import WARMUPS

    if warmup_name not in WARMUPS:
        raise ValueError(
            f"_run_warmup: unknown warmup {warmup_name!r}; "
            f"available: {sorted(WARMUPS)}"
        )
    warmup = WARMUPS[warmup_name]
    if not warmup.is_compatible(algorithm_entry.name):
        raise ValueError(
            f"warmup {warmup_name!r} is not compatible with "
            f"base_method {algorithm_entry.name!r}; "
            f"compatible_methods = {warmup.compatible_methods}"
        )
    return warmup.runner(
        rng_key,
        init_position,
        n_warmup,
        algorithm_entry,
        logdensity_fn=logdensity_fn,
    )


def _run_trial(
    logdensity_fn: Any,
    adapted_state: Any,
    algorithm_entry: BaseMethod,
    kernel_params: dict[str, Any],
    n_chains: int,
    n_samples: int,
    rng_key: jax.Array,
) -> float:
    """Run one BO trial: sample n_chains chains, compute headline metric.

    Parameters
    ----------
    logdensity_fn
        BlackJAX-compatible log-density.
    adapted_state
        Warmup-adapted initial state (reused across all trials).
    algorithm_entry
        Registry entry carrying ``factory`` and ``grad_count_per_step``.
    kernel_params
        Full kernel parameter dict: warmup IMM merged with BO trial HPs.
    n_chains
        Number of independent chains.
    n_samples
        Post-warmup samples per chain.
    rng_key
        JAX random key.

    Returns
    -------
    float
        ``min_bulk_ess_per_grad`` across all chains.  Returns
        ``-jnp.inf`` if the run produced non-finite values or diverged.

    Notes
    -----
    Multi-chain is implemented as a Python loop over ``n_chains`` rather
    than ``jax.vmap`` to keep complexity minimal in T2.6b.  T2.6c may add
    vmap-based parallelism if profiling shows it to be worthwhile.

    The warmup-adapted state is **reused across trials** (one warmup per
    study, not per trial).  Trial HPs override the BO-tunable parameters
    (e.g. ``step_size``, ``num_integration_steps`` for HMC) but the
    ``inverse_mass_matrix`` from warmup persists.
    """
    try:
        kernel = algorithm_entry.factory(logdensity_fn, **kernel_params)

        # Collect samples across n_chains (Python loop; vmap deferred to T2.6c)
        chain_positions: list[dict[str, Any]] = []
        chain_grad_evals: int = 0

        for i in range(n_chains):
            chain_key = jax.random.fold_in(rng_key, i)
            _, (states, infos) = run_inference_algorithm(
                rng_key=chain_key,
                inference_algorithm=kernel,
                num_steps=n_samples,
                initial_state=adapted_state,
            )
            chain_positions.append(states.position)
            chain_grad_evals += total_grad_evals(
                infos, algorithm_entry.grad_count_per_step
            )

        if chain_grad_evals == 0:
            # Algorithm has no gradient cost (e.g. RWM) — return 0-grad headline
            # but this path should not be reached for needs_mass_matrix=True algorithms.
            return float("inf")

        # Build (n_chains, n_samples, *site_shape) arrays for headline metric
        # Each chain_positions[i] is a dict {site: (n_samples, *shape)}
        all_sites: dict[str, Any] = {}
        for site in chain_positions[0]:
            per_chain = [cp[site] for cp in chain_positions]
            # Stack: (n_chains, n_samples, *shape)
            stacked = jnp.stack(per_chain, axis=0)
            all_sites[site] = stacked

        # Check for non-finite values — diverged chains poison the score
        for site_arr in all_sites.values():
            if not bool(jnp.all(jnp.isfinite(site_arr))):
                return float("-inf")

        score = min_bulk_ess_per_grad(all_sites, chain_grad_evals)

        if not math.isfinite(score):
            return float("-inf")

        return score

    except Exception:  # noqa: BLE001 — any JAX/runtime error → failed trial
        return float("-inf")


def _build_tuning_difficulty(
    history: list[dict[str, Any]],
    cumulative_wall_times: list[float],
) -> TuningDifficulty:
    """Compute TuningDifficulty from the per-trial history.

    Parameters
    ----------
    history
        List of per-trial records (already populated).  Each record has
        ``{"trial", "params", "score", "certified", "wall_seconds"}``.
    cumulative_wall_times
        Cumulative wall-clock seconds from study start, one entry per trial.
        ``cumulative_wall_times[i]`` is the time elapsed at the END of
        trial ``i``.

    Returns
    -------
    TuningDifficulty
        Fully populated difficulty profile.
    """
    n = len(history)
    if n == 0:
        return TuningDifficulty(
            default_score=float("-inf"),
            best_score=float("-inf"),
            threshold_score=float("-inf"),
            default_works=False,
            n_trials_to_threshold=0,
            n_trials_to_best=0,
            wall_seconds_to_threshold=0.0,
            wall_seconds_to_best=0.0,
        )

    scores = [rec["score"] for rec in history]
    default_score = float(scores[0])
    best_score = float(max(scores))
    threshold_score = float(max(default_score, 0.5 * best_score))
    default_works = bool(default_score >= 0.5 * best_score)

    # Trial number at which the best score first appears
    n_trials_to_best = int(scores.index(best_score))
    wall_seconds_to_best = float(cumulative_wall_times[n_trials_to_best])

    if default_works:
        n_trials_to_threshold = 0
        wall_seconds_to_threshold = 0.0
    else:
        # First trial (after trial 0) that reaches threshold_score
        n_trials_to_threshold = n  # sentinel: "never reached"
        wall_seconds_to_threshold = float(cumulative_wall_times[-1])
        for idx, score in enumerate(scores):
            if float(score) >= threshold_score:
                n_trials_to_threshold = idx
                wall_seconds_to_threshold = float(cumulative_wall_times[idx])
                break

    return TuningDifficulty(
        default_score=default_score,
        best_score=best_score,
        threshold_score=threshold_score,
        default_works=default_works,
        n_trials_to_threshold=n_trials_to_threshold,
        n_trials_to_best=n_trials_to_best,
        wall_seconds_to_threshold=wall_seconds_to_threshold,
        wall_seconds_to_best=wall_seconds_to_best,
    )


# ---------------------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------------------


def tune_algorithm(
    posterior_entry: Any,  # Posterior; import deferred to avoid circular dep
    algorithm_entry: BaseMethod,
    *,
    n_trials: int = 50,
    n_seeds: int = 5,
    n_chains: int = 4,
    n_samples: int = 500,
    n_warmup: int = 1000,
    rng_key: Any = None,  # jax.Array
    sampler: Literal["tpe", "random"] = "tpe",
    storage: str | None = None,
    warmup_name: str | None = None,
) -> TuningResult:
    """Optuna BO over ``algorithm_entry.default_hp_space``.

    Architecture (T2.6b + T2.6c):

    1. **Warmup dispatch** (via ``_run_warmup``):

       - ``needs_mass_matrix=True`` (NUTS, HMC, Barker): ``window_adaptation``
         runs ONCE per study; adapted ``inverse_mass_matrix`` is reused across
         all trials.
       - MALA / RWM (no warmup): kernel state is initialised from default HPs;
         all kernel params come from BO trial suggestions.
       - MCLMC: ``blackjax.mclmc_find_L_and_step_size`` runs for ``n_warmup``
         steps; the returned ``L``, ``step_size``, and ``inverse_mass_matrix``
         are warm-start values that BO trials can override.

    2. **Trial 0 = default**: before calling ``study.optimize``, the loop
       enqueues a deterministic default configuration via
       ``study.enqueue_trial(default_params_for(algorithm_entry))``.  This
       pins trial 0 as the "out-of-the-box" reference; the BO difficulty
       metric is defined relative to this baseline.

    3. **Sampler**: controlled by the ``sampler`` argument.  ``"tpe"``
       (default) uses ``optuna.samplers.TPESampler``; ``"random"`` uses
       ``optuna.samplers.RandomSampler``.  Both are seeded deterministically
       from ``rng_key`` so results are reproducible.  This enables the
       dogfood comparison described in
       ``PLAN_bjx_bench_API_phase2.md`` §"BO library choice — rationale".

    4. **Optuna direction**: "maximize" (higher ``min_bulk_ess_per_grad`` is
       better).

    5. **Multi-chain / multi-seed**: Python loop over ``n_chains`` (no vmap)
       and ``n_seeds`` (averaged score).

    6. **best_trial fallback**: if all trials diverge,
       ``study.best_trial`` raises ``ValueError``.  The loop catches this
       and returns trial-0 (the enqueued default) as the fallback "best"
       with score ``-inf``.

    7. **TuningDifficulty**: computed from the per-trial history after all
       trials complete.

    Parameters
    ----------
    posterior_entry
        A ``Posterior`` describing the target distribution.
    algorithm_entry
        The ``BaseMethod`` whose ``default_hp_space`` defines the BO
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
        Warmup steps per seed (window adaptation, or MCLMC tuning steps).
    rng_key
        Base JAX random key; subkeys are folded-in per trial and per seed.
    sampler
        Optuna suggestion strategy.  ``"tpe"`` (default) uses
        ``TPESampler``; ``"random"`` uses ``RandomSampler``.  Both options
        preserve the full Optuna machinery (enqueue_trial, difficulty
        profile, result schema).
    storage
        Optional Optuna RDB URL (e.g. ``"sqlite:///tuning.db"``) for
        persistent study storage.  ``None`` for in-memory (default).
    warmup_name
        Name of the warmup procedure from the ``WARMUPS`` registry
        (``"stan_window"``, ``"mclmc_tuning"``, ``"no_warmup"``).
        When ``None`` (default), the warmup is auto-dispatched:

        - ``"mclmc_tuning"`` for ``algorithm_entry.name == "mclmc"``
        - ``"stan_window"`` for ``algorithm_entry.needs_mass_matrix == True``
          (NUTS, HMC, Barker, MALA)
        - ``"no_warmup"`` for all remaining algorithms (RWM, etc.)

        This auto-dispatch reproduces the inline behavior of the Phase 2
        ``_run_warmup`` exactly.

    Returns
    -------
    TuningResult
        Full tuning outcome including best params, score, history, and
        difficulty profile.  If all trials diverge, ``best_params`` holds
        the trial-0 (default) params and ``best_score`` is ``-inf``.

    Raises
    ------
    ValueError
        If ``sampler`` is not ``"tpe"`` or ``"random"``, or if
        ``warmup_name`` is unknown or incompatible with the algorithm.
    """
    from bjx_bench.model._numpyro import build_logdensity_fn

    # ------------------------------------------------------------------
    # 1. Build logdensity_fn from posterior entry
    # ------------------------------------------------------------------
    rng_key_init, rng_key_warmup, rng_key_study = jax.random.split(rng_key, 3)

    init_position, logdensity_fn, _ = build_logdensity_fn(rng_key_init, posterior_entry)

    # ------------------------------------------------------------------
    # 2. Resolve warmup_name (auto-dispatch when None)
    # ------------------------------------------------------------------
    if warmup_name is None:
        if algorithm_entry.name == "mclmc":
            warmup_name = "mclmc_tuning"
        elif algorithm_entry.needs_mass_matrix:
            warmup_name = "stan_window"
        else:
            warmup_name = "no_warmup"

    # ------------------------------------------------------------------
    # 3. Single warmup run (reused across ALL trials)
    # ------------------------------------------------------------------
    adapted_state, warmup_params = _run_warmup(
        logdensity_fn=logdensity_fn,
        init_position=init_position,
        algorithm_entry=algorithm_entry,
        n_warmup=n_warmup,
        rng_key=rng_key_warmup,
        warmup_name=warmup_name,
    )
    # warmup_params for stan_window (MM-kernels):
    #   {"step_size": ..., "inverse_mass_matrix": ...}
    # warmup_params for no_warmup (MALA/RWM): {} (all params come from BO)
    # warmup_params for mclmc_tuning:
    #   {"L": ..., "step_size": ..., "inverse_mass_matrix": ...,
    #    "_total_tuning_steps": ...}

    # ------------------------------------------------------------------
    # 3. Optuna study with selected sampler
    # ------------------------------------------------------------------
    sampler_seed = int(jax.random.bits(jax.random.fold_in(rng_key_study, 0)))
    if sampler == "tpe":
        optuna_sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    elif sampler == "random":
        optuna_sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    else:
        raise ValueError(f"sampler must be 'tpe' or 'random', got {sampler!r}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna_sampler,
        storage=storage,
    )

    # Enqueue deterministic default as trial 0 (pins the "out-of-the-box"
    # reference point for the TuningDifficulty calculation).
    study.enqueue_trial(default_params_for(algorithm_entry))

    # ------------------------------------------------------------------
    # 5. Per-trial objective (closure over warmup state + logdensity_fn)
    # ------------------------------------------------------------------
    history: list[dict[str, Any]] = []
    cumulative_wall_times: list[float] = []
    study_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        # Suggest HPs from the search space.
        trial_params: dict[str, Any] = {}
        for space in algorithm_entry.default_hp_space:
            dist = optuna_distribution_for_space(space)
            if isinstance(dist, optuna.distributions.FloatDistribution):
                trial_params[space.name] = trial.suggest_float(
                    space.name,
                    dist.low,
                    dist.high,
                    log=dist.log,
                )
            elif isinstance(dist, optuna.distributions.IntDistribution):
                trial_params[space.name] = trial.suggest_int(
                    space.name,
                    dist.low,
                    dist.high,
                )
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                trial_params[space.name] = trial.suggest_categorical(
                    space.name,
                    list(dist.choices),
                )

        # Merge: warmup IMM is fixed; trial_params override BO-tunable HPs.
        # The BO search space intentionally does NOT include
        # inverse_mass_matrix (verified by tests in T2.2), so the merge
        # is safe: warmup_params provides IMM; trial_params provide the rest.
        # Strip internal metadata keys (underscore-prefixed, e.g.
        # "_total_tuning_steps" from mclmc_tuning) before passing to the
        # kernel factory — those keys are for Recipe.calibration_budget, not
        # for blackjax kernel construction.
        public_warmup_params = {
            k: v for k, v in warmup_params.items() if not k.startswith("_")
        }
        kernel_params = {**public_warmup_params, **trial_params}

        # Average score over n_seeds
        seed_scores: list[float] = []
        trial_num = trial.number
        t_start = time.perf_counter()
        for seed_idx in range(n_seeds):
            seed_key = jax.random.fold_in(
                jax.random.fold_in(rng_key_study, trial_num + 1),
                seed_idx,
            )
            score = _run_trial(
                logdensity_fn=logdensity_fn,
                adapted_state=adapted_state,
                algorithm_entry=algorithm_entry,
                kernel_params=kernel_params,
                n_chains=n_chains,
                n_samples=n_samples,
                rng_key=seed_key,
            )
            seed_scores.append(score)

        # Mean over seeds; if all diverged, propagate -inf
        finite_scores = [s for s in seed_scores if math.isfinite(s)]
        if not finite_scores:
            avg_score = float("-inf")
        else:
            avg_score = float(sum(finite_scores) / len(finite_scores))

        t_end = time.perf_counter()
        wall_s = t_end - t_start

        # Record per-trial history and cumulative wall time
        history.append(
            {
                "trial": trial_num,
                "params": dict(trial_params),
                "score": avg_score,
                "certified": math.isfinite(avg_score),
                "wall_seconds": wall_s,
            }
        )
        cumulative_wall_times.append(t_end - study_start)

        return avg_score

    study.optimize(objective, n_trials=n_trials)

    # ------------------------------------------------------------------
    # 6. Build TuningDifficulty from history
    # ------------------------------------------------------------------
    difficulty = _build_tuning_difficulty(history, cumulative_wall_times)

    # ------------------------------------------------------------------
    # 7. Build and return TuningResult
    # ------------------------------------------------------------------
    # Robustness guard: if ALL trials diverged, study.best_trial raises
    # ValueError (Optuna has no finite-valued trial to report as best).
    # We fall back to trial-0 (the enqueued default) so the caller always
    # receives a valid TuningResult rather than an exception.  The
    # best_score of -inf signals "all trials diverged" to the caller.
    try:
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_score = float(best_trial.value)
    except ValueError:
        # All trials returned -inf.  Fall back to trial-0 (default params).
        if not study.trials:
            raise RuntimeError("No trials completed; this should not happen")
        fallback = study.trials[0]
        best_params = dict(fallback.params)
        best_score = (
            float(fallback.value) if fallback.value is not None else float("-inf")
        )

    return TuningResult(
        base_method_name=algorithm_entry.name,
        posterior_name=posterior_entry.name,
        best_params=best_params,
        best_score=best_score,
        n_trials_completed=len(study.trials),
        n_seeds=n_seeds,
        history=tuple(history),
        difficulty=difficulty,
    )
