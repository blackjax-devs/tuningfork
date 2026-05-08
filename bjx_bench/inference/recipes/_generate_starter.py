"""One-shot script to emit canonical starter recipes for all efforts and algorithms.

Subtask P3.3 (Phase 3): generate 12 LOW + 6 MEDIUM starter recipes.

**LOW recipes** (zero-cost; uses no_warmup):
  - 3 starter models × 6 algorithms (hmc, nuts, mala, barker, rwm, mclmc)
  - Calls ``Recipe.from_default_config`` for each pair.
  - The 6 existing LOW recipes for {hmc, nuts} are regenerated bit-identical
    (from_default_config is deterministic).

**MEDIUM recipes** (warmup-only; uses stan_window):
  - 3 starter models × 2 algorithms (nuts, hmc)
  - Runs stan_window warmup at n_warmup=1000; records adapted params + elapsed.
  - Calls ``Recipe.from_warmup_only`` for each pair.
  - Total: 18 LOW + 6 MEDIUM = 24 starter recipes on disk.

Usage
-----
    cd bjx-bench
    uv run python -m bjx_bench.inference.recipes._generate_starter

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps.

Compute: ~3–5 min total (6 MEDIUM warmups dominate; each ~30–60s on CPU).
"""

from __future__ import annotations

from pathlib import Path

import jax

from bjx_bench._version import __version__ as _bjx_bench_version
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes._base import Recipe
from bjx_bench.inference.warmup import WARMUPS
from bjx_bench.model import MODELS

# The 3 starter models (Phase 2.5 scope; Phase 4 adds the remaining 11)
STARTER_MODEL_NAMES = ["mvn_10", "neals_funnel", "eight_schools_ncp"]

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


def main() -> None:
    """Generate and save all LOW + MEDIUM starter recipes.

    Subtask P3.3: emit 18 LOW + 6 MEDIUM = 24 total starter recipes.
    Expected compute: ~3–5 min total.
    """
    print("Generating LOW-effort recipes (6 algorithms, 3 models)...")
    low = emit_low_recipes()

    print("\nGenerating MEDIUM-effort recipes (stan_window warmup; nuts, hmc only)...")
    medium = emit_medium_recipes()

    total = len(low) + len(medium)
    print(
        f"\n✓ Emitted {len(low)} LOW + {len(medium)} MEDIUM = {total} starter recipes."
    )


if __name__ == "__main__":
    main()
