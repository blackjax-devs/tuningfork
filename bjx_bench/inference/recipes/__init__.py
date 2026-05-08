"""bjx-bench recipes: pinned configurations per (model, base_method, effort).

See PLAN_bjx_bench_restructure.md § "Recipe schema" and PLAN_bjx_bench_API_phase2.md
§ "Tuning Difficulty Metric" for the design.
"""

from bjx_bench.inference.recipes._base import Effort, Recipe

__all__ = ["Recipe", "Effort"]
