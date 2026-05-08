"""One-shot script to emit LOW-effort starter recipes.

Iterates over the 3 starter models × 2 algorithms (hmc, nuts) and calls
``Recipe.from_default_config`` for each pair. Saves the resulting JSON files
under ``bjx_bench/inference/recipes/starter/<model>/``.

MEDIUM and HIGH recipes are deferred to a follow-up spawn (they require
running warmup or Tier-B BO, which take minutes each).

Usage
-----
    cd bjx-bench
    uv run python bjx_bench/inference/recipes/_generate_starter.py

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps.
"""

from __future__ import annotations

from pathlib import Path

from bjx_bench._version import __version__ as _bjx_bench_version
from bjx_bench.inference.base_method import BASE_METHODS
from bjx_bench.inference.recipes._base import Recipe
from bjx_bench.model import MODELS

# The 3 starter models (Phase 2.5 scope; Phase 4 adds the remaining 11)
STARTER_MODEL_NAMES = ["mvn_10", "neals_funnel", "eight_schools_ncp"]

# LOW-effort recipes target hmc and nuts only (both are mass-matrix kernels
# whose default HPs are well-specified; MALA/RWM/MCLMC recipes land in Phase 3
# when the warmup wrappers for them are committed)
STARTER_METHOD_NAMES = ["hmc", "nuts"]

# Root of the starter/ directory (relative to this file)
_STARTER_ROOT = Path(__file__).parent / "starter"


def main() -> None:
    """Generate and save all LOW-effort starter recipes."""
    generated: list[Path] = []

    for model_name in STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in STARTER_METHOD_NAMES:
            base_method = BASE_METHODS[method_name]
            recipe = Recipe.from_default_config(
                posterior,
                base_method,
                bjx_bench_version=_bjx_bench_version,
            )
            path = recipe.save(_STARTER_ROOT)
            generated.append(path)
            print(
                f"  wrote {path.relative_to(Path(__file__).parent.parent.parent.parent)}"
            )

    print(f"\nGenerated {len(generated)} LOW-effort starter recipes.")
    print("MEDIUM and HIGH recipes are deferred to a follow-up spawn.")


if __name__ == "__main__":
    main()
