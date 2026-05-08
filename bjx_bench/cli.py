"""bjx-bench CLI — Phase 1 + Phase 2 subcommands.

Phase 1 surface:
    bjx-bench tier-a <model> [--n N] [--seed S] [--force]

Phase 2 surface:
    bjx-bench tune <model> <algo> [--n-trials N] [--n-seeds N]
                                  [--n-chains N] [--n-samples N]
                                  [--n-warmup N] [--seed S]
                                  [--sampler tpe|random] [--save PATH]

Future phases will add:
    bjx-bench warmup <model>        # Tier-C warmup-isolation cross-product
    bjx-bench report                # leaderboard + figures
"""

from __future__ import annotations

import argparse
import sys
import time


def _cmd_tier_a(args: argparse.Namespace) -> int:
    """Handle the `tier-a` subcommand."""
    import jax

    from bjx_bench.model import MODELS
    from bjx_bench.reference._io import get_reference_draws, get_reference_summaries

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
    from bjx_bench.reference._io import _load_metadata, _resolve_cache_dir

    cache_dir = _resolve_cache_dir(None)
    meta = _load_metadata(entry.name, cache_dir)
    cert = meta.get("certification", {}) if meta else {}
    cert_passed = cert.get("passed", None)

    # Summary table
    col_w = 22
    print()
    print("bjx-bench tier-a summary")
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


def _cmd_tune(args: argparse.Namespace) -> int:
    """Handle the ``tune`` subcommand.

    Validates model/algo names, calls ``tune_algorithm``, prints a summary
    table, and optionally saves the result JSON to ``--save PATH``.
    """
    import json

    from bjx_bench.inference.base_method import BASE_METHODS
    from bjx_bench.model import MODELS

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
        f"bjx-bench tune  model={args.model}  algo={args.algo}"
        f"  sampler={args.sampler}  n_trials={args.n_trials}  seed={args.seed}"
    )
    print()

    # ------------------------------------------------------------------ #
    # 3. Run tuning                                                       #
    # ------------------------------------------------------------------ #
    import jax

    from bjx_bench.calibration.tier_b import tune_algorithm

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
    """CLI entry point for bjx-bench."""
    parser = argparse.ArgumentParser(
        prog="bjx-bench",
        description="BlackJAX benchmark harness — Phase 1",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- tier-a subcommand ----
    p_tier_a = sub.add_parser(
        "tier-a",
        help="Generate or load Tier-A reference draws",
    )
    p_tier_a.add_argument(
        "model",
        help="Model name, e.g. mvn_10, eight_schools_ncp",
    )
    p_tier_a.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="Number of reference draws (default: 100000)",
    )
    p_tier_a.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0)",
    )
    p_tier_a.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache is valid",
    )

    # ---- tune subcommand ----
    p_tune = sub.add_parser(
        "tune",
        help="Find best hyperparameters via Optuna Tier-B BO loop",
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

    args = parser.parse_args()
    if args.cmd == "tier-a":
        return _cmd_tier_a(args)
    if args.cmd == "tune":
        return _cmd_tune(args)

    # Should not be reachable (subparsers required=True)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
