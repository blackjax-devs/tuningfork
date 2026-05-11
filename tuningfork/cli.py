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
"""tuningfork CLI — main entry point for reference, tune, warmup, and leaderboard subcommands."""

import argparse
import sys
import time


def _cmd_reference(args: argparse.Namespace) -> int:
    """Handle the `reference` subcommand."""
    import jax

    from tuningfork.model import MODELS
    from tuningfork.reference._io import get_reference_draws, get_reference_summaries

    if args.model not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 1

    entry = MODELS[args.model]
    key = jax.random.key(args.seed)

    # NUTS-specific parameters: use small fixed values for v1 CLI
    # (full 100k-sample production run requires explicit --n-warmup wiring in v2)
    nuts_kwargs: dict = {}
    if entry.reference_method.value == "nuts":
        # Use conservative small defaults so the CLI smoke-test is fast;
        # production runs should invoke get_reference_draws() directly with
        # n_warmup=5000, n_samples=100_000.
        nuts_kwargs = {"n_warmup": 500, "n_chunks": 4}

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
    from tuningfork.reference._io import _load_metadata, _resolve_cache_dir

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


def _serialize_tuning_result(result: object) -> dict:
    """Serialize a TuningResult dataclass to a JSON-friendly dict.

    Conversion rules applied:
    - ``dataclasses.asdict()`` recursively converts nested dataclasses to dicts
      and tuples to lists.
    - ``jax.Array`` values in ``best_params`` (and nested dicts) are coerced to
      Python floats/ints via ``float()``/``int()`` depending on their dtype.
    - numpy scalars (np.float32, np.int32, etc.) are coerced the same way.
    - ``tuple`` → ``list`` is already handled by ``dataclasses.asdict()`` for
      top-level named fields; inner tuples in ``best_params`` or ``history``
      are handled by ``_coerce_for_json``.

    The ``history`` field is a tuple of per-trial dicts, each containing:
        {"trial": int, "params": dict, "score": float, "certified": bool,
         "wall_seconds": float}
    Per-trial ``params`` dicts may also contain jax.Array/numpy scalars and
    are coerced recursively.

    Parameters
    ----------
    result
        A ``TuningResult`` instance.

    Returns
    -------
    dict
        JSON-serializable dict.
    """
    import dataclasses
    import math

    import numpy as np

    try:
        import jax as _jax_mod

        _jax_available = True
    except ImportError:
        _jax_available = False

    def _coerce_value(v: object) -> object:
        """Coerce a single value to a JSON-serializable Python primitive."""
        # JAX arrays
        if _jax_available:
            if isinstance(v, _jax_mod.Array):
                arr = np.asarray(v)
                if arr.ndim == 0:
                    scalar = arr.item()
                    # Represent as float unless it's exactly integral
                    if isinstance(scalar, float) and scalar == int(scalar):
                        return scalar  # keep float; json.dumps handles it fine
                    return scalar
                # 1-D+ arrays → list of Python primitives
                return arr.tolist()
        # numpy scalars / 0-d arrays
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        # already primitive
        return v

    def _coerce_for_json(obj: object) -> object:
        """Recursively coerce a nested structure to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: _coerce_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce_for_json(item) for item in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            # JSON does not support nan/inf; represent as null-like strings
            return str(obj)
        return _coerce_value(obj)

    raw = dataclasses.asdict(result)  # type: ignore[call-overload]
    return _coerce_for_json(raw)  # type: ignore[return-value]


def _cmd_warmup(args: argparse.Namespace) -> int:
    """Handle the ``warmup`` subcommand.

    Validates model/algo/warmup names, runs warmup-only via
    Recipe.from_warmup_only, prints a summary table, and optionally saves
    the recipe JSON to ``--save PATH``.
    """
    import json

    from tuningfork.inference.base_method import BASE_METHODS
    from tuningfork.inference.warmup import WARMUPS
    from tuningfork.model import MODELS

    # ------------------------------------------------------------------ #
    # 1. Validate model, algo, and warmup                                #
    # ------------------------------------------------------------------ #
    if args.model not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 2

    if args.algo not in BASE_METHODS:
        known = ", ".join(sorted(BASE_METHODS.keys()))
        print(
            f"error: unknown algorithm {args.algo!r}. Known algorithms: {known}",
            file=sys.stderr,
        )
        return 2

    if args.warmup not in WARMUPS:
        known = ", ".join(sorted(WARMUPS.keys()))
        print(
            f"error: unknown warmup {args.warmup!r}. Known warmups: {known}",
            file=sys.stderr,
        )
        return 2

    posterior_entry = MODELS[args.model]
    algorithm_entry = BASE_METHODS[args.algo]
    warmup_entry = WARMUPS[args.warmup]

    # Validate warmup-algo compatibility.
    if not warmup_entry.is_compatible(algorithm_entry.name):
        print(
            f"error: warmup {args.warmup!r} is not compatible with "
            f"algorithm {args.algo!r}. "
            f"Warmup {args.warmup!r} supports: {warmup_entry.compatible_methods}",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------ #
    # 2. Banner                                                           #
    # ------------------------------------------------------------------ #
    print()
    print(
        f"tuningfork warmup  model={args.model}  algo={args.algo}  "
        f"warmup={args.warmup}  n_warmup={args.n_warmup}  seed={args.seed}"
    )
    print()

    # ------------------------------------------------------------------ #
    # 3. Run warmup-only recipe generation                               #
    # ------------------------------------------------------------------ #
    import jax

    from tuningfork.inference.recipes._base import Recipe

    rng_key = jax.random.key(args.seed)
    t0 = time.monotonic()
    recipe = Recipe.from_warmup_only(
        posterior_entry,
        algorithm_entry,
        warmup_entry,
        n_warmup=args.n_warmup,
        rng_key=rng_key,
    )
    wall_total = time.monotonic() - t0

    # ------------------------------------------------------------------ #
    # 4. Print summary table                                              #
    # ------------------------------------------------------------------ #
    col_w = 26
    print(f"warmup {args.model} {args.algo}")
    print(f"  {'effort:':<{col_w}} {recipe.effort.value}")
    print(f"  {'warmup_name:':<{col_w}} {recipe.warmup_name}")
    print(f"  {'n_warmup:':<{col_w}} {args.n_warmup}")
    print(f"  {'wall_seconds_estimate:':<{col_w}} {wall_total:.2f}")
    # Display key adapted parameters
    if "step_size" in recipe.base_method_params:
        print(
            f"  {'adapted_step_size:':<{col_w}} "
            f"{recipe.base_method_params['step_size']:.6g}"
        )
    if "inverse_mass_matrix" in recipe.base_method_params:
        imm = recipe.base_method_params["inverse_mass_matrix"]
        if isinstance(imm, (int, float)):
            imm_shape = "scalar"
        elif isinstance(imm, (list, tuple)):
            imm_shape = f"vector[{len(imm)}]"
        else:
            imm_shape = str(type(imm).__name__)
        print(f"  {'inverse_mass_matrix:':<{col_w}} {imm_shape}")
    print(
        f"  {'base_method_params keys:':<{col_w}} {', '.join(sorted(recipe.base_method_params.keys()))}"
    )
    print()

    # ------------------------------------------------------------------ #
    # 5. Optionally save JSON                                             #
    # ------------------------------------------------------------------ #
    if args.save:
        from pathlib import Path

        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        d = recipe.__dict__.copy()
        # Convert effort Enum to string value
        d["effort"] = recipe.effort.value
        with save_path.open("w") as fh:
            json.dump(d, fh, indent=2, default=str)
        print(f"Recipe saved to {save_path}")

    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Handle the ``leaderboard`` subcommand.

    Scans tuningfork/inference/recipes/starter/<model>/*.json, loads recipes,
    filters by effort (if specified), and renders a markdown table (default)
    or JSON list.
    """
    import json
    from pathlib import Path

    from tuningfork.inference.recipes._base import Recipe
    from tuningfork.model import MODELS

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
    recipe_dir = (
        Path(__file__).parent / "inference" / "recipes" / "starter" / args.model
    )
    recipes: list[Recipe] = []
    if not recipe_dir.exists():
        # Model directory doesn't exist; no recipes
        pass
    else:
        recipe_files = sorted(recipe_dir.glob("*.json"))
        for recipe_file in recipe_files:
            try:
                recipe = Recipe.load(recipe_file)
                recipes.append(recipe)
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
        print()

        if not recipes:
            print("No recipes found.")
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


def _cmd_tune(args: argparse.Namespace) -> int:
    """Handle the ``tune`` subcommand.

    Validates model/algo names, calls ``tune_algorithm``, prints a summary
    table, and optionally saves the result JSON to ``--save PATH``.
    """
    import json

    from tuningfork.inference.base_method import BASE_METHODS
    from tuningfork.model import MODELS

    # ------------------------------------------------------------------ #
    # 1. Validate model and algo                                          #
    # ------------------------------------------------------------------ #
    if args.model not in MODELS:
        known = ", ".join(sorted(MODELS.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 2

    if args.algo not in BASE_METHODS:
        known = ", ".join(sorted(BASE_METHODS.keys()))
        print(
            f"error: unknown algorithm {args.algo!r}. Known algorithms: {known}",
            file=sys.stderr,
        )
        return 2

    posterior_entry = MODELS[args.model]
    algorithm_entry = BASE_METHODS[args.algo]

    # ------------------------------------------------------------------ #
    # 2. Banner                                                           #
    # ------------------------------------------------------------------ #
    print()
    print(
        f"tuningfork tune  model={args.model}  algo={args.algo}"
        f"  sampler={args.sampler}  n_trials={args.n_trials}  seed={args.seed}"
    )
    print()

    # ------------------------------------------------------------------ #
    # 3. Run tuning                                                       #
    # ------------------------------------------------------------------ #
    import jax

    from tuningfork.calibration.tune import tune_algorithm

    rng_key = jax.random.key(args.seed)
    t0 = time.monotonic()
    result = tune_algorithm(
        posterior_entry,
        algorithm_entry,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        n_chains=args.n_chains,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        rng_key=rng_key,
        sampler=args.sampler,
    )
    wall_total = time.monotonic() - t0

    # ------------------------------------------------------------------ #
    # 4. Print summary table                                              #
    # ------------------------------------------------------------------ #
    diff = result.difficulty
    col_w = 26
    print(f"tune {args.model} {args.algo}")
    print(f"  {'sampler:':<{col_w}} {args.sampler}")
    print(f"  {'n_trials:':<{col_w}} {result.n_trials_completed}")
    print(f"  {'best_score:':<{col_w}} {result.best_score:.6g} ESS/grad")
    print(f"  {'default_score:':<{col_w}} {diff.default_score:.6g} ESS/grad")
    print(f"  {'default_works:':<{col_w}} {diff.default_works}")
    print(f"  {'n_trials_to_threshold:':<{col_w}} {diff.n_trials_to_threshold}")
    print(f"  {'n_trials_to_best:':<{col_w}} {diff.n_trials_to_best}")
    print(f"  {'wall_seconds_total:':<{col_w}} {wall_total:.1f}")
    # Serialize best_params for display (coerce jax.Array → Python)
    import json as _json

    best_params_display = {
        k: (float(v) if hasattr(v, "__float__") else v)
        for k, v in result.best_params.items()
    }
    print(f"  {'best_params:':<{col_w}} {_json.dumps(best_params_display)}")
    print()

    # ------------------------------------------------------------------ #
    # 5. Optionally save JSON                                             #
    # ------------------------------------------------------------------ #
    if args.save:
        from pathlib import Path

        serialized = _serialize_tuning_result(result)
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as fh:
            json.dump(serialized, fh, indent=2)
        print(f"Result saved to {save_path}")

    return 0


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

    # ---- warmup subcommand ----
    p_warmup = sub.add_parser(
        "warmup",
        help="Run warmup-only and output a MEDIUM-effort recipe.",
    )
    p_warmup.add_argument(
        "model",
        help="Model name, e.g. mvn_10, neals_funnel",
    )
    p_warmup.add_argument(
        "algo",
        help="Algorithm name, e.g. nuts, hmc, mala, barker, rwm, mclmc",
    )
    p_warmup.add_argument(
        "--warmup",
        default="stan_window",
        help="Warmup strategy: stan_window, mclmc_tuning, no_warmup (default: stan_window)",
    )
    p_warmup.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        dest="n_warmup",
        help="Number of warmup steps (default: 1000)",
    )
    p_warmup.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    p_warmup.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="If set, write JSON recipe to this path",
    )

    # ---- tune subcommand ----
    p_tune = sub.add_parser(
        "tune",
        help="Find best hyperparameters via Optuna BO tuning loop",
    )
    p_tune.add_argument(
        "model",
        help="Model name, e.g. mvn_10, neals_funnel",
    )
    p_tune.add_argument(
        "algo",
        help="Algorithm name, e.g. nuts, hmc, mala, barker, rwm, mclmc",
    )
    p_tune.add_argument(
        "--n-trials",
        type=int,
        default=5,
        dest="n_trials",
        help="Total Optuna trials (including default trial 0) (default: 5)",
    )
    p_tune.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        dest="n_seeds",
        help="Random seeds averaged per trial (default: 1)",
    )
    p_tune.add_argument(
        "--n-chains",
        type=int,
        default=1,
        dest="n_chains",
        help="MCMC chains per seed (default: 1)",
    )
    p_tune.add_argument(
        "--n-samples",
        type=int,
        default=200,
        dest="n_samples",
        help="Post-warmup samples per chain (default: 200)",
    )
    p_tune.add_argument(
        "--n-warmup",
        type=int,
        default=200,
        dest="n_warmup",
        help="Warmup steps (default: 200)",
    )
    p_tune.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (default: 0)",
    )
    p_tune.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna suggestion strategy: tpe or random (default: tpe)",
    )
    p_tune.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="If set, write JSON-serialized TuningResult to this path",
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
    if args.cmd == "warmup":
        return _cmd_warmup(args)
    if args.cmd == "tune":
        return _cmd_tune(args)
    if args.cmd == "leaderboard":
        return _cmd_leaderboard(args)

    # Should not be reachable (subparsers required=True)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
