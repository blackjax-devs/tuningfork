"""bjx-bench CLI entry point. Phase 6 will populate this with subcommands:

    bjx-bench tier-a <model>     # certify gold reference draws
    bjx-bench tune <model> <algo>  # Tier-B per-algorithm tuning
    bjx-bench warmup <model>     # Tier-C warmup-isolation cross-product
    bjx-bench report             # leaderboard + figures

For Phase 0 (scaffold), this is a stub that prints the design status.
"""


def main() -> int:
    print(
        "bjx-bench (scaffold) — see PLAN_bjx_bench.md for the design.\n"
        "Subcommands not implemented yet (Phase 6)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
