"""bjx-bench warmup registry. Real warmup modules land in Phase 3."""

from bjx_bench.inference.warmup._base import Warmup

WARMUPS: dict[str, Warmup] = {}

__all__ = ["WARMUPS", "Warmup"]
