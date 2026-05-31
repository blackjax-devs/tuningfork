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
"""Step-policy sweep runner for ``dynamic_hmc`` and ``dmhmc``.

Evaluates integration-step-count policy candidates in cheapest-first order,
gates each on R̂ / ESS / divergence, and promotes the first passing candidate
to a MEDIUM recipe.

Candidate ordering (V7 demoted per statistician finding on miscalibration at
large step_size):
    V1 → V4 → V6 → V7 → V2 → V5

Gate cascade:
    PASS:  rhat < 1.01, min_bulk_ess ≥ 400, n_div_rate ≤ 5%
    First PASS wins. All attempted candidates are recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CandidateResult",
    "SweepResult",
    "build_default_candidates",
    "sweep_and_pick",
    "GATE_RHAT_PASS",
    "GATE_ESS_PASS",
    "GATE_DIV_RATE_PASS",
]

# ---------------------------------------------------------------------------
# Gate thresholds# ---------------------------------------------------------------------------

GATE_RHAT_PASS: float = 1.01  # rhat must be strictly below this
GATE_ESS_PASS: float = 400.0  # min bulk-ESS must be at least this
GATE_DIV_RATE_PASS: float = 0.05  # divergence rate (n_div / total_draws) ≤ 5%

# ---------------------------------------------------------------------------
# Default candidate ordering (§9, V7 demoted per statistician finding #2)
# ---------------------------------------------------------------------------

#: Step-policy spec for V1 — uniform_int [5, 50).  Cheapest; adequate for
#: NIS_median ≤ ~40 models (radon, irt_2pl, eight_schools_ncp).
V1_SPEC: dict[str, Any] = {"kind": "uniform_int", "low": 5, "high": 50}
#: Step-policy spec for V2 — uniform_int [50, 200).  Fallback for moderate-L models.
V2_SPEC: dict[str, Any] = {"kind": "uniform_int", "low": 50, "high": 200}
#: Step-policy spec for V5 — log_uniform_int [1, 1024].  Broad fallback / banana probe.
V5_SPEC: dict[str, Any] = {"kind": "log_uniform_int", "low": 1, "high": 1024}
#: Step-policy spec for V6 — pow2_choice {2,4,8,16,32,64}.  NUTS-like doubling grid.
V6_SPEC: dict[str, Any] = {"kind": "pow2_choice", "options": [2, 4, 8, 16, 32, 64]}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Result from a single step_policy candidate evaluation.

    Parameters
    ----------
    step_policy_spec
        The step_policy spec dict that was evaluated.
    verdict
        ``"PASS"``, ``"FAIL"``, or ``"REVIEW"`` from the auto-gate.
    rhat_max
        Maximum rank-normalised split-R̂ across all parameters.
    min_bulk_ess
        Minimum bulk-ESS across all parameters.
    n_divergences
        Total divergent transitions.
    n_draws_total
        Total number of draws (num_chains × n_samples).
    wall_seconds
        Wall clock time for warmup + sampling.
    wall_s_per_1k_ess
        Wall-time efficiency metric: ``wall_seconds / (min_bulk_ess / 1000)``.
        Smaller = more efficient.  ``None`` when min_bulk_ess ≤ 0.
    """

    step_policy_spec: dict[str, Any]
    verdict: str  # "PASS" | "FAIL" | "REVIEW"
    rhat_max: float
    min_bulk_ess: float
    n_divergences: int
    n_draws_total: int
    wall_seconds: float
    wall_s_per_1k_ess: float | None = None

    def __post_init__(self) -> None:
        if self.min_bulk_ess > 0 and self.wall_s_per_1k_ess is None:
            self.wall_s_per_1k_ess = self.wall_seconds / (self.min_bulk_ess / 1000.0)

    @property
    def div_rate(self) -> float:
        """Divergence rate: n_divergences / n_draws_total."""
        if self.n_draws_total == 0:
            return 0.0
        return self.n_divergences / self.n_draws_total

    def passes_gate(self) -> bool:
        """Return True iff this candidate passes all gate criteria (§9)."""
        return (
            self.rhat_max < GATE_RHAT_PASS
            and self.min_bulk_ess >= GATE_ESS_PASS
            and self.div_rate <= GATE_DIV_RATE_PASS
        )


@dataclass
class SweepResult:
    """Result from a complete sweep_and_pick run.

    Parameters
    ----------
    model
        Model name.
    warmup
        Warmup name.
    sampler
        Sampler name (``"dynamic_hmc"`` or ``"dmhmc"``).
    winner
        Winning ``CandidateResult``, or ``None`` if no candidate passed.
    all_candidates
        Ordered list of all evaluated candidates (pass and fail).
    """

    model: str
    warmup: str
    sampler: str
    winner: CandidateResult | None
    all_candidates: list[CandidateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff at least one candidate passed the gate."""
        return self.winner is not None

    def to_attempted_configurations(self) -> list[dict[str, Any]]:
        """Format for ``Recipe.attempted_configurations`` (recording spec).

        Returns
        -------
        list[dict]
            Each entry has ``"step_policy"``, ``"outcome"``,
            ``"gate_evidence"``, and (for passes) ``"wall_s_per_1k_ess"``.
        """
        configs = []
        for c in self.all_candidates:
            entry: dict[str, Any] = {
                "step_policy": c.step_policy_spec,
                "outcome": c.verdict,
                "gate_evidence": {
                    "rhat_max": c.rhat_max,
                    "min_bulk_ess": c.min_bulk_ess,
                    "n_divergences": c.n_divergences,
                },
            }
            if c.wall_s_per_1k_ess is not None:
                entry["wall_s_per_1k_ess"] = c.wall_s_per_1k_ess
            configs.append(entry)
        return configs


# ---------------------------------------------------------------------------
# Candidate set builder
# ---------------------------------------------------------------------------


def build_default_candidates(
    *,
    nis_median: int | None = None,
    chain_stats_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Build the default candidate list in sweep order.

    Candidate ordering (V7 demoted per statistician finding #2):
    V1 → V4 → V6 → V7 → V2 → V5

    Parameters
    ----------
    nis_median
        The model's NIS median (from ``reference/adaptation.json``
        ``["num_leapfrog_median"]``).  Used to parameterise V4 (Poisson lam)
        and to suppress V6 when NIS_med > 64.  Pass ``None`` to skip V4/V6.
    chain_stats_path
        Path to a ``nuts × wadapt`` ``chain_stats.npz`` cache for the model.
        Required for V7 (empirical oracle); omit to skip V7.

    Returns
    -------
    list[dict]
        Ordered list of step_policy spec dicts, cheapest-first.
    """
    candidates: list[dict[str, Any]] = []

    # V1: cheap parametric baseline
    candidates.append(V1_SPEC)

    # V4: Poisson(lam=NIS_median) — if median available
    if nis_median is not None:
        candidates.append(
            {"kind": "poisson", "lam": int(nis_median), "low": 1, "high": None}
        )

    # V6: pow2 grid — only when NIS_median ≤ 64 (grid covers the typical range)
    if nis_median is None or nis_median <= 64:
        candidates.append(V6_SPEC)

    # V7: empirical oracle — if chain_stats available
    if chain_stats_path is not None:
        from tuningfork.base_method._step_policy_registry import (
            harvest_step_policy_from_chain_stats,
        )

        try:
            v7_spec = harvest_step_policy_from_chain_stats(chain_stats_path)
            candidates.append(v7_spec)
        except (KeyError, ValueError):
            pass  # chain_stats missing NIS field — skip V7 silently

    # V2: medium-range fallback
    candidates.append(V2_SPEC)

    # V5: broad log-uniform last resort
    candidates.append(V5_SPEC)

    return candidates


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------


def sweep_and_pick(
    model: str,
    warmup: str,
    sampler: str,
    candidates: list[dict[str, Any]],
    *,
    n_warmup: int = 2000,
    n_samples: int = 1000,
    num_chains: int = 4,
    ta: float = 0.8,
    catalog_root: Path | None = None,
    seed: int = 20260517,
    stop_at_first_pass: bool = True,
    verbose: bool = True,
) -> SweepResult:
    """Run step_policy candidates in order and return the first (or best) PASS.

    Implements the gate cascade:

    1. Evaluate each candidate in order via ``emit_low_recipe_for_cell``.
    2. Apply gate: rhat < 1.01, min_ESS ≥ 400, n_div_rate ≤ 5%.
    3. First candidate to pass wins (``stop_at_first_pass=True``).
    4. All attempted candidates are recorded.

    Parameters
    ----------
    model
        Model name (e.g. ``"radon"``).
    warmup
        Warmup name (e.g. ``"window_adaptation_diag_imm"``).
    sampler
        Sampler name — must be ``"dynamic_hmc"`` or ``"dmhmc"``.
    candidates
        Ordered list of step_policy spec dicts (cheapest-first).
        See ``build_default_candidates`` for the canonical ordering.
    n_warmup
        Warmup steps per chain (default 2000).
    n_samples
        Post-warmup samples per chain (default 1000).
    num_chains
        Number of parallel chains (default 4).
    ta
        Target acceptance rate (default 0.8; use 0.99 for stiff ODE models).
    catalog_root
        Root of the catalog directory. ``None`` uses the committed catalog.
    seed
        Master random seed (default 20260517).
    stop_at_first_pass
        When ``True`` (default), stop after the first passing candidate.
        When ``False`` (full sweep mode), run all candidates regardless.
    verbose
        Print per-candidate results to stdout.

    Returns
    -------
    SweepResult
        Contains the winning candidate (or ``None``) and all attempted results.
    """
    import time
    from pathlib import Path as _Path

    from tuningfork.recipes._base import Effort
    from tuningfork.recipes._recipe_runner import (
        _CATALOG_ROOT,
        emit_low_recipe_for_cell,
    )

    if sampler not in ("dynamic_hmc", "dmhmc"):
        raise ValueError(
            f"sweep_and_pick only supports 'dynamic_hmc' and 'dmhmc'; "
            f"got sampler={sampler!r}. For other samplers, use emit_low_recipe_for_cell."
        )

    _root = _Path(catalog_root) if catalog_root is not None else _CATALOG_ROOT
    all_results: list[CandidateResult] = []
    winner: CandidateResult | None = None

    for idx, spec in enumerate(candidates):
        kind = spec.get("kind", "unknown")
        if verbose:
            print(
                f"  Candidate {idx + 1}/{len(candidates)}: kind={kind!r} spec={spec!r}"
            )

        # Derive a deterministic policy_tag from the spec for filename tagging.
        _slug = _spec_to_slug(spec)
        _tag = f"policy_{_slug}"

        t0 = time.perf_counter()
        result = emit_low_recipe_for_cell(
            model_name=model,
            warmup_name=warmup,
            sampler_name=sampler,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            seed=seed,
            catalog_root=_root,
            verbose=False,
            step_policy=spec,
            policy_tag=_tag,
            effort=Effort.MEDIUM,
            target_acceptance=ta,
        )
        wall = time.perf_counter() - t0

        rhat = result.gate_rhat_max if result.gate_rhat_max is not None else 9.9
        ess = result.gate_min_ess if result.gate_min_ess is not None else 0.0
        n_div = result.gate_n_div if result.gate_n_div is not None else 0
        n_total = n_samples * num_chains

        candidate_result = CandidateResult(
            step_policy_spec=spec,
            verdict=result.verdict,
            rhat_max=rhat,
            min_bulk_ess=ess,
            n_divergences=n_div,
            n_draws_total=n_total,
            wall_seconds=wall,
        )
        all_results.append(candidate_result)

        if verbose:
            print(
                f"    → {result.verdict}  rhat={rhat:.4f}  ess={ess:.0f}  "
                f"div={n_div}  wall={wall:.1f}s"
            )

        if candidate_result.passes_gate():
            winner = candidate_result
            if stop_at_first_pass:
                if verbose:
                    print(
                        f"  First PASS found at candidate {idx + 1} — stopping sweep."
                    )
                break

    if verbose:
        if winner is not None:
            print(f"  WINNER: {winner.step_policy_spec!r}")
        else:
            print("  No candidate passed the gate cascade.")

    return SweepResult(
        model=model,
        warmup=warmup,
        sampler=sampler,
        winner=winner,
        all_candidates=all_results,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_to_slug(spec: dict[str, Any]) -> str:
    """Convert a step_policy spec to a short filename slug.

    Used for the ``policy_tag`` argument to ``emit_low_recipe_for_cell``.

    Examples
    --------
    >>> _spec_to_slug({"kind": "uniform_int", "low": 5, "high": 50})
    'v1-uniform5-50'
    >>> _spec_to_slug({"kind": "poisson", "lam": 15, "low": 1})
    'v4-poisson15'
    >>> _spec_to_slug({"kind": "log_uniform_int", "low": 1, "high": 1024})
    'v5-logunif'
    >>> _spec_to_slug({"kind": "pow2_choice", "options": [2, 4, 8, 16, 32, 64]})
    'v6-pow2'
    >>> _spec_to_slug({"kind": "empirical", "values": [10, 20], "weights": [0.5, 0.5]})
    'v7-empirical'
    """
    kind = spec.get("kind", "unknown")
    if kind == "uniform_int":
        low = spec.get("low", "?")
        high = spec.get("high", "?")
        return (
            f"v1-uniform{low}-{high}"
            if (low, high) != (50, 200)
            else "v2-uniform50-200"
        )
    if kind == "poisson":
        lam = spec.get("lam", "?")
        return f"v4-poisson{lam}"
    if kind == "log_uniform_int":
        return "v5-logunif"
    if kind == "pow2_choice":
        return "v6-pow2"
    if kind == "empirical":
        return "v7-empirical"
    return f"policy-{kind}"
