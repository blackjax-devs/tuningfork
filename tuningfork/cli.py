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
"""tuningfork CLI — main entry point for reference and leaderboard subcommands."""

import argparse
import sys
import time


def _cmd_reference(args: argparse.Namespace) -> int:
    """Handle the `reference` subcommand."""
    import jax

    from tuningfork._cache_io import get_reference_draws, get_reference_summaries
    from tuningfork.model import MODELS

    if args.model not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 1

    entry = MODELS[args.model]

    # Per-model x64 requirement: auto-enable BEFORE any JAX computation.
    # Must happen before jax.random.key() below, which triggers JAX initialisation.
    # Strictly analogous to how gp_regression and lotka_volterra need float64.
    if entry.requires_x64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    key = jax.random.key(args.seed)

    # NUTS-specific parameters: pass through CLI flags; fall back to
    # get_reference_draws production defaults (n_warmup=5000, n_chunks=4).
    nuts_kwargs: dict = {}
    if entry.reference_method.value == "nuts":
        nuts_kwargs = {
            "n_warmup": args.n_warmup,
            "n_chunks": 4,
            "target_acceptance": entry.reference_target_acceptance,
        }

    t0 = time.monotonic()
    draws = get_reference_draws(
        entry,
        n=args.n,
        rng_key=key,
        force_regenerate=args.force,
        **nuts_kwargs,
    )
    elapsed = time.monotonic() - t0

    # Ensure summaries are cached (side-effect: writes summaries/<name>.json)
    get_reference_summaries(entry)

    # Total sample count (first dim of any site)
    n_actual = next(iter(draws.values())).shape[0]

    # Certification info from the last metadata stamp
    from tuningfork._cache_io import _load_metadata, _resolve_cache_dir

    cache_dir = _resolve_cache_dir(None)
    meta = _load_metadata(entry.name, cache_dir)
    cert = meta.get("certification", {}) if meta else {}
    cert_passed = cert.get("passed", None)

    # Summary table
    col_w = 22
    print()
    print("tuningfork reference summary")
    print("=" * 55)
    print(f"{'model':<{col_w}} {entry.name}")
    print(f"{'class':<{col_w}} {entry.class_}")
    print(f"{'generator':<{col_w}} {entry.reference_method.value}")
    print(f"{'dim':<{col_w}} {entry.dim}")
    print(f"{'num_samples':<{col_w}} {n_actual:,}")
    print(f"{'seed':<{col_w}} {args.seed}")
    if cert_passed is not None:
        passed_str = "PASSED" if cert_passed else "FAILED"
        print(f"{'certification':<{col_w}} {passed_str}")
        if "split_rhat_max" in cert and cert["split_rhat_max"] is not None:
            print(f"{'  split_rhat_max':<{col_w}} {cert['split_rhat_max']:.4f}")
        if "min_chunk_bulk_ess" in cert and cert["min_chunk_bulk_ess"] is not None:
            print(f"{'  min_chunk_ess':<{col_w}} {cert['min_chunk_bulk_ess']:.1f}")
        if "num_divergences" in cert and cert["num_divergences"] is not None:
            print(f"{'  num_divergences':<{col_w}} {cert['num_divergences']}")
        if "e_bfmi" in cert and cert["e_bfmi"] is not None:
            print(f"{'  e_bfmi':<{col_w}} {cert['e_bfmi']:.4f}")
    print(f"{'wall_time_s':<{col_w}} {elapsed:.2f}")
    print("=" * 55)
    print()

    # Exit code: 0 on cert passed or analytic (always passes), 1 on NUTS fail
    if cert_passed is False:
        return 1
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Handle the ``leaderboard`` subcommand.

    Scans tuningfork/catalog/<model>/{groundtruth.json, recipes/*.json}, loads recipes,
    filters by effort (if specified), and renders a markdown table (default)
    or JSON list.

    SMC recipes (``smc__*.json``) are excluded from the MCMC ranking — they
    use a fundamentally different execution model (particles instead of chains,
    no warmup phase, different gate metrics).  A note is printed when SMC
    recipes are present so the user knows they exist.
    """
    import json
    from pathlib import Path

    from tuningfork.model import MODELS
    from tuningfork.recipes._base import Recipe
    from tuningfork.recipes._base_smc import SMCRecipe

    # ------------------------------------------------------------------ #
    # 1. Validate model                                                  #
    # ------------------------------------------------------------------ #
    if args.model not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------ #
    # 2. Glob recipes from disk                                          #
    # ------------------------------------------------------------------ #
    recipe_dir = Path(__file__).parent / "catalog" / args.model / "recipes"
    recipes: list[Recipe] = []
    n_smc_skipped = 0
    if not recipe_dir.exists():
        # Model directory doesn't exist; no recipes
        pass
    else:
        recipe_files = sorted(recipe_dir.glob("*.json"))
        for recipe_file in recipe_files:
            try:
                loaded = _load_recipe_for_leaderboard(recipe_file)
                if isinstance(loaded, SMCRecipe):
                    # SMC recipes are excluded from the MCMC leaderboard ranking.
                    # They have no effort/base_method_name attrs and use a separate
                    # execution model (particles, no warmup, different gate metrics).
                    n_smc_skipped += 1
                else:
                    recipes.append(loaded)
            except Exception as e:
                print(
                    f"warning: failed to load {recipe_file}: {e}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------ #
    # 3. Filter by effort if specified                                   #
    # ------------------------------------------------------------------ #
    if args.effort:
        recipes = [r for r in recipes if r.effort.value == args.effort]

    # ------------------------------------------------------------------ #
    # 4. Sort: effort (LOW → MEDIUM → HIGH), then headline_metric desc  #
    #    within HIGH (null sorts to the end), then alphabetically        #
    # ------------------------------------------------------------------ #
    def sort_key(recipe):
        effort_order = {"low": 0, "medium": 1, "high": 2}
        effort_val = effort_order.get(recipe.effort.value, 999)

        # For sorting within HIGH: by headline_metric descending (nulls last),
        # then by algorithm name alphabetically
        if recipe.effort.value == "high" and recipe.headline_metric is not None:
            headline_sort = (-recipe.headline_metric, recipe.base_method_name)
        else:
            headline_sort = (float("inf"), recipe.base_method_name)

        return (effort_val, headline_sort)

    recipes.sort(key=sort_key)

    # ------------------------------------------------------------------ #
    # 5. Render output                                                   #
    # ------------------------------------------------------------------ #
    if args.format == "json":
        output = [
            {
                "model_name": recipe.model_name,
                "base_method_name": recipe.base_method_name,
                "warmup_name": recipe.warmup_name,
                "effort": recipe.effort.value,
                "headline_metric": recipe.headline_metric,
                "default_works": (
                    recipe.difficulty["default_works"] if recipe.difficulty else None
                ),
                "n_trials_to_threshold": (
                    recipe.difficulty["n_trials_to_threshold"]
                    if recipe.difficulty
                    else None
                ),
            }
            for recipe in recipes
        ]
        print(json.dumps(output, indent=2))
    else:
        # markdown format (default)
        print()
        print(f"Leaderboard for {args.model}:")
        if n_smc_skipped > 0:
            print(
                f"({n_smc_skipped} SMC recipe{'s' if n_smc_skipped != 1 else ''} present"
                " — not ranked; SMC uses a separate execution model)"
            )
        print()

        if not recipes:
            print("No MCMC recipes found.")
            return 0

        # Render markdown table
        print(
            "| effort | algorithm | warmup       | headline   | default_works | n_to_thresh |"
        )
        print(
            "|--------|-----------|--------------|------------|---------------|-------------|"
        )

        for recipe in recipes:
            effort = recipe.effort.value
            algorithm = recipe.base_method_name
            warmup = recipe.warmup_name

            # headline_metric: null for LOW/MEDIUM, show as "n/a"
            if recipe.headline_metric is None:
                headline_str = "n/a"
            else:
                headline_str = f"{recipe.headline_metric:.4g}"

            # default_works and n_to_thresh: only for HIGH
            if recipe.difficulty is not None:
                default_works_str = str(recipe.difficulty.get("default_works", "n/a"))
                n_to_thresh_str = str(
                    recipe.difficulty.get("n_trials_to_threshold", "n/a")
                )
            else:
                default_works_str = "n/a"
                n_to_thresh_str = "n/a"

            print(
                f"| {effort:<6} | {algorithm:<9} | {warmup:<12} | {headline_str:<10} | {default_works_str:<13} | {n_to_thresh_str:<11} |"
            )

        print()

    return 0


def _load_recipe_for_leaderboard(recipe_file):
    """Load a recipe file, dispatching to SMCRecipe or Recipe based on content."""
    import json

    from tuningfork.recipes._base import Recipe
    from tuningfork.recipes._base_smc import SMCRecipe

    d = json.loads(recipe_file.read_text())
    if "smc_method_name" in d:
        return SMCRecipe.load(recipe_file)
    return Recipe.load(recipe_file)


def main() -> int:
    """CLI entry point for tuningfork."""
    parser = argparse.ArgumentParser(
        prog="tuningfork",
        description="BlackJAX benchmark harness",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- reference subcommand ----
    p_reference = sub.add_parser(
        "reference",
        help="Generate or load certified reference draws",
    )
    p_reference.add_argument(
        "model",
        help="Model name, e.g. mvn_10, eight_schools_ncp",
    )
    p_reference.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="Number of reference draws (default: 100000)",
    )
    p_reference.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    p_reference.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache is valid",
    )
    p_reference.add_argument(
        "--n-warmup",
        type=int,
        default=5_000,
        dest="n_warmup",
        help="NUTS warmup steps (default: 5000; ignored for analytic models)",
    )

    # ---- leaderboard subcommand ----
    p_leaderboard = sub.add_parser(
        "leaderboard",
        help="Render a leaderboard table of recipes for a model",
    )
    p_leaderboard.add_argument(
        "model",
        help="Model name, e.g. mvn_10, neals_funnel",
    )
    p_leaderboard.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Filter by effort level (default: show all)",
    )
    p_leaderboard.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (default) or json",
    )

    args = parser.parse_args()
    if args.cmd == "reference":
        return _cmd_reference(args)
    if args.cmd == "leaderboard":
        return _cmd_leaderboard(args)

    # Should not be reachable (subparsers required=True)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
