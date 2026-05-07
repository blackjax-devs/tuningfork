"""bjx-bench CLI — Phase 1 subcommands.

Phase 1 surface:
    bjx-bench tier-a <model> [--n N] [--seed S] [--force]

Generates or loads Tier-A reference draws and prints a summary table.

Future phases will add:
    bjx-bench tune <model> <algo>   # Tier-B per-algorithm tuning
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

    from bjx_bench.reference._io import get_reference_draws, get_reference_summaries
    from bjx_bench.registry import REGISTRY

    if args.model not in REGISTRY:
        known = ", ".join(sorted(REGISTRY.keys()))
        print(
            f"error: unknown model {args.model!r}. Known models: {known}",
            file=sys.stderr,
        )
        return 1

    entry = REGISTRY[args.model]
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

    args = parser.parse_args()
    if args.cmd == "tier-a":
        return _cmd_tier_a(args)

    # Should not be reachable (subparsers required=True)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
