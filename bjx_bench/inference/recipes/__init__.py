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
"""bjx-bench recipes: pinned configurations per (model, base_method, effort).

See PLAN_bjx_bench_restructure.md § "Recipe schema" and PLAN_bjx_bench_API_phase2.md
§ "Tuning Difficulty Metric" for the design.
"""

from bjx_bench.inference.recipes._base import Effort, Recipe

__all__ = ["Recipe", "Effort"]
