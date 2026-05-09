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
"""One-shot script to emit canonical starter recipes for all efforts and algorithms.

Subtask P3.3 (Phase 3): generate 12 LOW + 6 MEDIUM starter recipes.
Subtask P3.4 (Phase 3): generate 6 HIGH starter recipes via Tier-B BO.
Subtask P4.1 (Phase 4): added ill_cond_50 to STARTER_MODEL_NAMES.
Subtask P5.0 (Phase 5): per-cell flag-filtering — ``--warmup``, ``--sampler``,
``--effort`` flags added alongside the existing ``--only <model>`` (Q5.A
decision: extend flags rather than introduce a YAML cell-spec file).

**LOW recipes** (zero-cost; uses no_warmup):
  - N starter models × 6 algorithms (hmc, nuts, mala, barker, rwm, mclmc)
  - Calls ``Recipe.from_default_config`` for each pair.
  - Deterministic — re-running produces bit-identical output.

**MEDIUM recipes** (warmup-only; uses stan_window):
  - N starter models × 2 algorithms (nuts, hmc)
  - Runs stan_window warmup at n_warmup=1000; records adapted params + elapsed.
  - Calls ``Recipe.from_warmup_only`` for each pair.

**HIGH recipes** (Tier-B BO; uses stan_window):
  - N starter models × 2 algorithms (nuts, hmc)
  - Runs tune_algorithm at n_trials=20, n_seeds=2, n_chains=2, n_samples=400,
    n_warmup=500; records BO-tuned config + TuningDifficulty profile.
  - Calls ``Recipe.from_tuning_result`` for each pair.

Usage
-----

The recipe space is 3-D: ``model × (warmup, sampler) × effort``. The four
filter flags compose intersectionally — each defaults to "all" and narrows
the matrix when set.

Common patterns:

  # 1. Generate everything (full sweep — slow; ~3-5 min LOW+MED + ~18-30 min HIGH per model)
  cd bjx-bench
  uv run python -m bjx_bench.inference.recipes._generate_starter

  # 2. Regenerate one model's full set after editing it (Phase 4 pattern)
  uv run python -m bjx_bench.inference.recipes._generate_starter --only radon

  # 3. Regenerate one cell only (Phase 5+ pattern; cheapest after a tweak)
  uv run python -m bjx_bench.inference.recipes._generate_starter \\
      --only radon --warmup stan_window --sampler nuts --effort medium

  # 4. Refresh all LOW recipes after a default_params_for change
  uv run python -m bjx_bench.inference.recipes._generate_starter --effort low

  # 5. Refresh NUTS HIGH across every model (after a Tier-B BO bug fix)
  uv run python -m bjx_bench.inference.recipes._generate_starter \\
      --sampler nuts --effort high

  # 6. Regenerate everything that uses a particular warmup
  uv run python -m bjx_bench.inference.recipes._generate_starter --warmup stan_window

Flag semantics:
  - ``--only <m>``       restrict to one model (must be in STARTER_MODEL_NAMES)
  - ``--warmup <w>``     restrict to one warmup; valid: "no_warmup" (LOW only),
                         "stan_window" (MED + HIGH currently)
  - ``--sampler <s>``    restrict to one base method; valid:
                         "hmc", "nuts", "mala", "barker", "rwm", "mclmc"
  - ``--effort <e>``     restrict to one effort tier; valid: "low", "medium", "high"

Validity rules (enforced at top of ``main``):
  - ``--effort low`` is incompatible with ``--warmup stan_window`` (LOW always
    uses ``no_warmup``).
  - ``--effort {medium, high}`` is incompatible with ``--warmup no_warmup``
    (those tiers always run a real warmup).
  - Sampler / warmup compatibility (e.g., ``mclmc`` is not compatible with
    ``stan_window``) is checked per-cell at emit time and reported as ``SKIP``.

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps. Compute: ~3-5 min total for LOW+MEDIUM per model;
~18-30 min for HIGH per model (n_trials=20).
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

# Root of the starter/ directory (relative to this file)
_STARTER_ROOT = Path(__file__).parent / "starter"


def emit_low_recipes(
    seed: int = 0,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit LOW recipes for every (model, base_method) combination.

    Idempotent — overwrites existing low__*.json files in place.
    Each recipe uses ``no_warmup`` (identity warmup) and default hyperparameters.

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
    """Emit MEDIUM recipes for stan_window-compatible algorithms.

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
    """Emit HIGH recipes via Tier-B BO for stan_window-compatible algorithms.

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
    """Generate and save LOW + MEDIUM + HIGH starter recipes (filtered by flags).

    See the module docstring for the full Usage section. Briefly: each filter
    flag (``--only``, ``--warmup``, ``--sampler``, ``--effort``) defaults to
    "all" and narrows the recipe matrix when set.

    Phase 3 (P3.3 + P3.4): 18 LOW + 6 MEDIUM + 6 HIGH = 30 starter recipes
    for the 3 original models.
    Phase 4 (P4.1+): adds ill_cond_50 and subsequent models as they land.
    Phase 5 (P5.0): the four flags now compose intersectionally — see module
    docstring Usage section for the 6 canonical patterns.
    """
    import argparse

    valid_warmups = {"no_warmup", "stan_window"}
    valid_samplers = set(ALL_METHOD_NAMES)
    valid_efforts = {"low", "medium", "high"}

    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical starter recipes (LOW/MEDIUM/HIGH).  "
            "All four filters compose; each defaults to 'all'."
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
            "Restrict to one warmup.  'no_warmup' is LOW-only; 'stan_window' "
            "is MED+HIGH only.  Default: all."
        ),
    )
    parser.add_argument(
        "--sampler",
        default=None,
        choices=sorted(valid_samplers),
        help=(
            "Restrict to one base method.  Note that 'mala', 'barker', "
            "'rwm', 'mclmc' are LOW-only currently.  Default: all."
        ),
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=sorted(valid_efforts),
        help="Restrict to one effort tier.  Default: all.",
    )
    args = parser.parse_args()

    # ── Validation ──────────────────────────────────────────────────────────
    if args.only is not None and args.only not in STARTER_MODEL_NAMES:
        raise SystemExit(
            f"--only {args.only!r} not in STARTER_MODEL_NAMES "
            f"= {STARTER_MODEL_NAMES}"
        )

    # Cross-flag coherence: warmup vs effort
    if args.effort == "low" and args.warmup == "stan_window":
        raise SystemExit(
            "--effort low is incompatible with --warmup stan_window "
            "(LOW recipes always use no_warmup)."
        )
    if args.effort in {"medium", "high"} and args.warmup == "no_warmup":
        raise SystemExit(
            f"--effort {args.effort} is incompatible with --warmup no_warmup "
            f"({args.effort.upper()} recipes always run a real warmup)."
        )

    names: list[str] | None = [args.only] if args.only is not None else None

    # ── Effort gating ───────────────────────────────────────────────────────
    # Each effort tier runs iff (a) --effort wasn't set, OR (b) --effort matches.
    # The warmup filter further gates each tier to its native warmup.
    do_low = args.effort in (None, "low") and args.warmup in (None, "no_warmup")
    do_medium = args.effort in (None, "medium") and args.warmup in (
        None,
        "stan_window",
    )
    do_high = args.effort in (None, "high") and args.warmup in (None, "stan_window")

    # ── Echo selection ──────────────────────────────────────────────────────
    selection = []
    if args.only is not None:
        selection.append(f"model={args.only}")
    if args.warmup is not None:
        selection.append(f"warmup={args.warmup}")
    if args.sampler is not None:
        selection.append(f"sampler={args.sampler}")
    if args.effort is not None:
        selection.append(f"effort={args.effort}")
    if selection:
        print(f"Generating recipes filtered by: {', '.join(selection)}")
    else:
        print("Generating ALL starter recipes (no filters set).")

    low: list[Path] = []
    medium: list[Path] = []
    high: list[Path] = []

    if do_low:
        print(
            "\nGenerating LOW-effort recipes "
            f"({'all' if args.sampler is None else args.sampler} algorithms)..."
        )
        low = emit_low_recipes(model_names=names, sampler=args.sampler)

    if do_medium:
        print(
            "\nGenerating MEDIUM-effort recipes "
            f"(stan_window warmup; "
            f"{'nuts/hmc' if args.sampler is None else args.sampler})..."
        )
        medium = emit_medium_recipes(model_names=names, sampler=args.sampler)

    if do_high:
        print(
            "\nGenerating HIGH-effort recipes "
            f"(Tier-B BO, n_trials=20; "
            f"{'nuts/hmc' if args.sampler is None else args.sampler})..."
        )
        high = emit_high_recipes(model_names=names, sampler=args.sampler)

    total = len(low) + len(medium) + len(high)
    print(
        f"\n✓ Emitted {len(low)} LOW + {len(medium)} MEDIUM + {len(high)} HIGH"
        f" = {total} starter recipes."
    )


if __name__ == "__main__":
    main()
