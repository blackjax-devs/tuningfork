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
"""Starter recipe emission — candidate generator for the Statistician gate.

This script generates candidate recipes for the conventional cells in the
``(model, warmup, sampler)`` space — pairs where the warmup is the natural
adaptation procedure for the sampler (see ``NATURAL_WARMUP_FOR_SAMPLER`` below).
Each candidate is a ``Recipe`` with ``effort=Effort.LOW``; the Statistician
auto-gate (``bjx_bench.calibration.statistician_gate``) decides whether to
commit (PASS) or escalate to MEDIUM / HIGH.  See ``_base.py``'s ``Effort``
docstring for the per-tier semantics.

Usage
-----

  cd bjx-bench
  uv run python -m bjx_bench.inference.recipes._generate_starter
  uv run python -m bjx_bench.inference.recipes._generate_starter --only radon
  uv run python -m bjx_bench.inference.recipes._generate_starter --sampler nuts

Flag semantics:
  - ``--only <m>``    restrict to one model (must be in STARTER_MODEL_NAMES)
  - ``--warmup <w>``  restrict to one warmup (``"no_warmup"`` or ``"stan_window"``)
  - ``--sampler <s>`` restrict to one base method

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps.

MEDIUM / HIGH escalation
------------------------

The ``emit_medium_recipes`` and ``emit_high_recipes`` functions are helpers
the Statistician can call during escalation; they are NOT exposed via the CLI.
"""

import time
from pathlib import Path

import jax

from bjx_bench._version import __version__ as _bjx_bench_version
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes._base import Recipe
from bjx_bench.inference.warmup import WARMUPS
from bjx_bench.model import MODELS

# Starter models: Phase 2.5 seed set + Phase 4 models as they land.
# P4.1 adds ill_cond_50 (Block A: 50-D ill-conditioned Gaussian, κ≈1000).
# P4.2 adds banana (Block A: 2-D banana/Rosenbrock-style, curved manifold).
# P4.3 adds gmm_25 (Block A: 2-D 25-mode Gaussian mixture on a 5x5 grid).
# P4.4 adds logistic_synthetic (Block B: 3-D logistic regression on 2-D bicluster).
# P4.5 adds german_credit (Block B: 26-D logistic regression on real UCI German Credit data).
# P4.6 adds horseshoe (Block B: 204-D Finnish horseshoe sparse linear regression, NCP).
# P4.7 adds radon (Block C: 391-D NCP hierarchical, posteriordb radon_all).
# P4.8 adds irt_2pl (Block C: 144-D NCP IRT 2PL, J=100 I=20, no posteriordb xcheck).
# P4.9 adds stoch_vol (Block D: 503-D NCP AR(1) state-space, KSC 1998).
# P4.10 adds lotka_volterra (Block D: 7-D ODE inverse via ProbDiffEq).
# P4.11 adds gp_regression (Block D: 203-D Cholesky-NCP joint GP, FINAL P4 model).
STARTER_MODEL_NAMES = [
    "mvn_10",
    "ill_cond_50",
    "neals_funnel",
    "eight_schools_ncp",
    "banana",
    "gmm_25",
    "logistic_synthetic",
    "german_credit",
    "horseshoe",
    "radon",
    "irt_2pl",
    "stoch_vol",
    "lotka_volterra",
    "gp_regression",
]

# All 6 algorithms (Phase 3: MALA/Barker/RWM/MCLMC added; LOW only)
ALL_METHOD_NAMES = ["hmc", "nuts", "mala", "barker", "rwm", "mclmc"]

# Only nuts and hmc support stan_window warmup in starter recipes (Phase 3 scope)
MEDIUM_METHOD_NAMES = ["nuts", "hmc"]

# ---------------------------------------------------------------------------
# Conventional pairing map (Phase 6 regen consumes this)
# ---------------------------------------------------------------------------
# The "natural" warmup for each base method — the warmup that the BlackJAX
# community / literature pairs with that sampler by default.  Under canonical
# Effort taxonomy (see _base.py), LOW recipes operate on these conventional
# pairings; MEDIUM recipes explore unconventional but technically-possible
# combinations (e.g., stan_window + mala, stan_window + rmhmc); HIGH recipes
# add oracle-tuned warmup HPs and model-specific injection.
#
# This map is not yet consumed by the legacy emit_* helpers below; the Phase 6
# regen pipeline will use it to drive cell-space iteration.
NATURAL_WARMUP_FOR_SAMPLER: dict[str, str] = {
    # Window-adaptation family (stan_window's compatible_methods)
    "hmc": "stan_window",
    "nuts": "stan_window",
    "mhmc": "stan_window",
    "barker": "stan_window",
    "mala": "stan_window",
    "rmhmc": "stan_window",
    # MCLMC family
    "mclmc": "mclmc_tuning",
    "adjusted_mclmc": "adjusted_mclmc_tuning",
    "adjusted_mclmc_dynamic": "adjusted_mclmc_tuning",
    # Multi-chain adaptation
    "ghmc": "meads",
    "dynamic_hmc": "chees",
    "dmhmc": "chees",
    # Gradient-free / specialised — no canonical warmup
    "rwm": "no_warmup",
    "irmh": "no_warmup",
    "additive_step_random_walk": "no_warmup",
    "elliptical_slice": "no_warmup",
    "mgrad_gaussian": "no_warmup",
    "orbital_hmc": "no_warmup",
    "laplace_hmc": "no_warmup",
    "laplace_dhmc": "no_warmup",
    "laplace_mhmc": "no_warmup",
    "laplace_dmhmc": "no_warmup",
    # VI as base method (the warmup *is* the optimization; no separate warmup)
    "meanfield_vi": "no_warmup",
    "fullrank_vi": "no_warmup",
}

# Root of the starter/ directory (relative to this file)
_STARTER_ROOT = Path(__file__).parent / "starter"


def emit_low_recipes(
    seed: int = 0,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit (B-taxonomy) LOW recipes — no_warmup + sampler defaults, no MCMC run.

    .. deprecated:: Phase 6
       Encodes the pre-reframe B-taxonomy (LOW = no_warmup + sampler stub with
       no MCMC).  Under canonical-C taxonomy (see ``_base.py`` ``Effort``
       docstring), LOW means "conventional pairing + library defaults +
       auto-gate passed", which requires running both warmup and sampler at
       recipe-build time.  The Phase 6 regen pipeline replaces this helper
       with a ``NATURAL_WARMUP_FOR_SAMPLER``-aware emitter that runs warmup
       + sampler + auto_gate per cell.

    Idempotent — overwrites existing ``low__*.json`` files in place.

    Parameters
    ----------
    seed
        Random seed for potential future use (currently unused; LOW is deterministic).
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``ALL_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    _ = seed  # currently unused; kept for symmetry with emit_medium_recipes
    generated: list[Path] = []
    repo_root = Path(__file__).parent.parent.parent.parent

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in ALL_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]
            recipe = Recipe.from_default_config(
                posterior,
                base_method,
                bjx_bench_version=_bjx_bench_version,
            )
            path = recipe.save(_STARTER_ROOT)
            generated.append(path)
            try:
                pretty = path.relative_to(repo_root)
            except ValueError:
                # _STARTER_ROOT was redirected outside the repo (e.g. tests using tmp_path);
                # fall back to absolute path so logging never crashes.
                pretty = path
            print(f"  LOW   {pretty}")

    return generated


def emit_medium_recipes(
    seed: int = 0,
    n_warmup: int = 1000,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit (B-taxonomy) MEDIUM recipes — stan_window adapt only, no sampler chain.

    .. deprecated:: Phase 6
       Encodes the pre-reframe B-taxonomy (MEDIUM = warmup-only adaptation,
       no sampler chain run).  Under canonical-C, MEDIUM is the result of
       Statistician investigation: either (a) a LOW-emit gate failure was
       recovered with a workaround (seed change, init change, obvious-bug
       fix), or (b) the cell explores a technically-possible-but-
       unconventional ``(warmup, sampler)`` pairing (e.g., stan_window +
       mala, stan_window + rmhmc).  Both branches run warmup + sampler +
       auto_gate; the Phase 6 regen pipeline drives them under Statistician
       direction.

    Idempotent: re-running overwrites with deterministic content (same seed → same key).

    Compatibility is limited by WARMUPS["stan_window"].compatible_methods,
    which is ("hmc", "nuts", "barker", "mala").  For Phase 3 scope, only
    nuts and hmc get MEDIUM recipes; the others (barker, mala) are deferred.

    Parameters
    ----------
    seed
        Base random seed; fold_in derives per-recipe keys deterministically.
    n_warmup
        Number of warmup adaptation steps (default 1000).
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``MEDIUM_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    stan_window = WARMUPS["stan_window"]
    generated: list[Path] = []
    repo_root = Path(__file__).parent.parent.parent.parent

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]

            # Check compatibility
            if not stan_window.is_compatible(method_name):
                print(
                    f"  SKIP  {model_name}/{method_name}: " f"stan_window incompatible"
                )
                continue

            # Derive a deterministic per-recipe key via fold_in.
            # hash() can produce values outside uint32; mask to 32 bits.
            hash_val = hash((model_name, method_name)) & 0xFFFFFFFF
            key = jax.random.fold_in(jax.random.key(seed), hash_val)

            recipe = Recipe.from_warmup_only(
                posterior,
                base_method,
                stan_window,
                n_warmup=n_warmup,
                rng_key=key,
                bjx_bench_version=_bjx_bench_version,
            )
            path = recipe.save(_STARTER_ROOT)
            generated.append(path)
            try:
                pretty = path.relative_to(repo_root)
            except ValueError:
                pretty = path
            print(f"  MEDIUM {pretty}")

    return generated


def emit_high_recipes(
    seed: int = 0,
    n_trials: int = 20,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit (B-taxonomy) HIGH recipes via Tier-B BO over sampler hyperparameters.

    .. deprecated:: Phase 6
       Encodes the pre-reframe B-taxonomy (HIGH = Tier-B BO over **sampler**
       hyperparameters with stan_window warmup).  Under canonical-C, HIGH is
       the Statistician's recovery path after LOW and MEDIUM both fail:
       compare against a NUTS + window_adaptation oracle, run BO over
       **warmup** hyperparameters (BO repurposed per Phase 5 reframe),
       inject model-specific parameters, and record the full Bayesian-
       workflow journey in ``Recipe.workflow``.  CI consumes the result by
       reading the pinned scalars + IMM sidecar (sampler-only at runtime).
       The Phase 6 regen pipeline replaces this helper.

    Runs ``tune_algorithm`` at ``n_trials=20`` (starter default per
    PLAN_bjx_bench_phase3.md; production recipes use 50).  Converts each
    ``TuningResult`` to a HIGH ``Recipe`` via ``Recipe.from_tuning_result``.

    Compatibility is limited to ``nuts`` and ``hmc`` for Phase 3 scope
    (the other 4 algorithms get LOW recipes only; their HIGH recipes land
    in Phase 4 alongside the new models).

    Parameters
    ----------
    seed
        Base random seed; ``jax.random.fold_in`` derives per-recipe keys
        deterministically from ``(model_name, method_name, "high")``.
    n_trials
        Total Optuna trials per study (including the injected default trial
        0).  Default is 20 for starter recipes; production uses 50.
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``MEDIUM_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    from bjx_bench.calibration.tier_b import tune_algorithm

    stan_window = WARMUPS["stan_window"]
    generated: list[Path] = []

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]

            if not stan_window.is_compatible(method_name):
                print(f"  SKIP  {model_name}/{method_name}: stan_window incompatible")
                continue

            # Derive a deterministic per-recipe key via fold_in.
            # hash() can produce values outside uint32; mask to 32 bits.
            hash_val = hash((model_name, method_name, "high")) & 0xFFFFFFFF
            key = jax.random.fold_in(jax.random.key(seed), hash_val)

            print(f"  tuning {model_name} + {method_name} " f"(n_trials={n_trials})...")
            t0 = time.perf_counter()
            result = tune_algorithm(
                posterior,
                base_method,
                n_trials=n_trials,
                n_seeds=2,
                n_chains=2,
                n_samples=400,
                n_warmup=500,
                rng_key=key,
                sampler="tpe",
                warmup_name="stan_window",
            )
            elapsed = time.perf_counter() - t0

            recipe = Recipe.from_tuning_result(
                result,
                posterior=posterior,
                base_method=base_method,
                warmup=stan_window,
                bjx_bench_version=_bjx_bench_version,
            )
            path = recipe.save(_STARTER_ROOT)
            generated.append(path)
            print(
                f"    done in {elapsed:.1f}s; "
                f"best_score={result.best_score:.4f}; "
                f"default_works={result.difficulty.default_works}"
            )

    return generated


def main() -> None:
    """LOW-emit step: produce candidate recipes for the Statistician gate.

    See the module docstring for the full Usage section.  Three filter flags
    (``--only``, ``--warmup``, ``--sampler``) compose intersectionally; each
    defaults to "all" and narrows the matrix when set.

    P5.Q5A removed the ``--effort`` flag because effort is gate-driven, not
    something the script chooses (only LOW candidates are emitted here;
    MEDIUM and HIGH come from a Statistician escalation workflow).

    Phase 3 (P3.3 + P3.4): 18 LOW + 6 MEDIUM + 6 HIGH = 30 starter recipes
    for the 3 original models.
    Phase 4 (P4.1+): adds ill_cond_50 and subsequent models as they land.
    Phase 5 (P5.Q5A): scoped to LOW emission; ``--effort`` removed.
    Phase 6 (in flight): emit-logic rewrite to canonical-C (NATURAL_WARMUP_FOR_SAMPLER-
    aware iteration; warmup + sampler + auto_gate per cell).  This CLI still drives
    the legacy B-taxonomy emit helpers for backward compatibility until the regen
    pipeline lands.
    """
    import argparse

    valid_warmups = {"no_warmup", "stan_window"}
    valid_samplers = set(ALL_METHOD_NAMES)

    parser = argparse.ArgumentParser(
        description=(
            "Emit candidate recipes (legacy B-taxonomy: LOW=no_warmup stub, "
            "MEDIUM=stan_window adapt, HIGH=Tier-B BO).  All three filters "
            "compose; each defaults to 'all'.  Under canonical-C taxonomy, "
            "MEDIUM/HIGH recipes come from a Statistician escalation "
            "workflow, not from this CLI; the Phase 6 regen pipeline "
            "replaces this script."
        )
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Restrict to one model.  Must be in STARTER_MODEL_NAMES.  "
            "Default: all starter models."
        ),
    )
    parser.add_argument(
        "--warmup",
        default=None,
        choices=sorted(valid_warmups),
        help=(
            "Restrict to one warmup.  'no_warmup' applies to the LOW pass "
            "for samplers without trajectory adaptation; 'stan_window' "
            "applies to NUTS/HMC and other window-compatible samplers.  "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--sampler",
        default=None,
        choices=sorted(valid_samplers),
        help=(
            "Restrict to one base method.  Default: all of "
            "{hmc, nuts, mala, barker, rwm, mclmc}."
        ),
    )
    args = parser.parse_args()

    # ── Validation ──────────────────────────────────────────────────────────
    if args.only is not None and args.only not in STARTER_MODEL_NAMES:
        raise SystemExit(
            f"--only {args.only!r} not in STARTER_MODEL_NAMES "
            f"= {STARTER_MODEL_NAMES}"
        )

    names: list[str] | None = [args.only] if args.only is not None else None

    # The warmup filter selects which LOW pass to run.
    # `no_warmup`   → emit_low_recipes   (no adaptation; identity warmup)
    # `stan_window` → emit_medium_recipes (window adaptation; *as a LOW candidate*
    #                                      under the new framing — this function
    #                                      runs the default warmup and produces
    #                                      candidate output for the gate)
    do_no_warmup = args.warmup in (None, "no_warmup")
    do_stan_window = args.warmup in (None, "stan_window")

    # ── Echo selection ──────────────────────────────────────────────────────
    selection = []
    if args.only is not None:
        selection.append(f"model={args.only}")
    if args.warmup is not None:
        selection.append(f"warmup={args.warmup}")
    if args.sampler is not None:
        selection.append(f"sampler={args.sampler}")
    if selection:
        print(f"Emitting LOW candidates filtered by: {', '.join(selection)}")
    else:
        print("Emitting ALL LOW candidates (no filters set).")

    no_warmup_paths: list[Path] = []
    stan_window_paths: list[Path] = []

    if do_no_warmup:
        print(
            "\nEmitting candidates for warmup=no_warmup "
            f"({'all algorithms' if args.sampler is None else args.sampler})..."
        )
        no_warmup_paths = emit_low_recipes(model_names=names, sampler=args.sampler)

    if do_stan_window:
        print(
            "\nEmitting candidates for warmup=stan_window "
            f"({'nuts/hmc' if args.sampler is None else args.sampler})..."
        )
        stan_window_paths = emit_medium_recipes(model_names=names, sampler=args.sampler)

    total = len(no_warmup_paths) + len(stan_window_paths)
    print(
        f"\n✓ Emitted {len(no_warmup_paths)} no_warmup + "
        f"{len(stan_window_paths)} stan_window = {total} LOW candidates.  "
        f"Next: Statistician gate (P5.0.5)."
    )


if __name__ == "__main__":
    main()
