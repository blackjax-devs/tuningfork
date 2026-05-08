"""One-shot script to emit canonical starter recipes for all efforts and algorithms.

Subtask P3.3 (Phase 3): generate 12 LOW + 6 MEDIUM starter recipes.
Subtask P3.4 (Phase 3): generate 6 HIGH starter recipes via Tier-B BO.
Subtask P4.1 (Phase 4): added ill_cond_50 to STARTER_MODEL_NAMES.

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
    cd bjx-bench
    uv run python -m bjx_bench.inference.recipes._generate_starter

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps.

Compute: ~3–5 min total for LOW+MEDIUM per model; ~18–30 min for HIGH per model
(n_trials=20).
"""

from __future__ import annotations

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
STARTER_MODEL_NAMES = [
    "mvn_10",
    "ill_cond_50",
    "neals_funnel",
    "eight_schools_ncp",
    "banana",
    "gmm_25",
    "logistic_synthetic",
    "german_credit",
]

# All 6 algorithms (Phase 3: MALA/Barker/RWM/MCLMC added; LOW only)
ALL_METHOD_NAMES = ["hmc", "nuts", "mala", "barker", "rwm", "mclmc"]

# Only nuts and hmc support stan_window warmup in starter recipes (Phase 3 scope)
MEDIUM_METHOD_NAMES = ["nuts", "hmc"]

# Root of the starter/ directory (relative to this file)
_STARTER_ROOT = Path(__file__).parent / "starter"


def emit_low_recipes(seed: int = 0) -> list[Path]:
    """Emit LOW recipes for every (model, base_method) combination.

    Idempotent — overwrites existing low__*.json files in place.
    Each recipe uses ``no_warmup`` (identity warmup) and default hyperparameters.

    Parameters
    ----------
    seed
        Random seed for potential future use (currently unused; LOW is deterministic).

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    _ = seed  # currently unused; kept for symmetry with emit_medium_recipes
    generated: list[Path] = []

    for model_name in STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in ALL_METHOD_NAMES:
            base_method = BASE_METHODS[method_name]
            recipe = Recipe.from_default_config(
                posterior,
                base_method,
                bjx_bench_version=_bjx_bench_version,
            )
            path = recipe.save(_STARTER_ROOT)
            generated.append(path)
            print(
                f"  LOW   {path.relative_to(Path(__file__).parent.parent.parent.parent)}"
            )

    return generated


def emit_medium_recipes(seed: int = 0, n_warmup: int = 1000) -> list[Path]:
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

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    stan_window = WARMUPS["stan_window"]
    generated: list[Path] = []

    for model_name in STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
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
            print(
                f"  MEDIUM {path.relative_to(Path(__file__).parent.parent.parent.parent)}"
            )

    return generated


def emit_high_recipes(seed: int = 0, n_trials: int = 20) -> list[Path]:
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

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    from bjx_bench.calibration.tier_b import tune_algorithm

    stan_window = WARMUPS["stan_window"]
    generated: list[Path] = []

    for model_name in STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
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
    """Generate and save all LOW + MEDIUM + HIGH starter recipes.

    Phase 3 (P3.3 + P3.4): 18 LOW + 6 MEDIUM + 6 HIGH = 30 starter recipes
    for the 3 original models.
    Phase 4 (P4.1+): adds ill_cond_50 and subsequent models as they land.
    Expected compute: ~3–5 min for LOW+MEDIUM per model; ~18–30 min for HIGH
    per model (n_trials=20).
    """
    print("Generating LOW-effort recipes (6 algorithms, 3 models)...")
    low = emit_low_recipes()

    print("\nGenerating MEDIUM-effort recipes (stan_window warmup; nuts, hmc only)...")
    medium = emit_medium_recipes()

    print(
        "\nGenerating HIGH-effort recipes "
        "(Tier-B BO, n_trials=20; nuts, hmc only)..."
    )
    high = emit_high_recipes()

    total = len(low) + len(medium) + len(high)
    print(
        f"\n✓ Emitted {len(low)} LOW + {len(medium)} MEDIUM + {len(high)} HIGH"
        f" = {total} starter recipes."
    )


if __name__ == "__main__":
    main()
