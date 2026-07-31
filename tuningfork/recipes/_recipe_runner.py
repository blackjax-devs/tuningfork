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
"""Compatibility facade for codegen-only recipe certification.

Historically this module contained a second warmup and sampling implementation.
The public entry point now delegates to the generated-program lifecycle. A few
private aliases remain while older callers move to their focused modules.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from tuningfork.recipes import _certification_runner

CellResult = _certification_runner.CellResult
emit_low_recipe_for_cell = _certification_runner.emit_low_recipe_for_cell
RECIPE_N_WARMUP = _certification_runner.RECIPE_N_WARMUP
RECIPE_N_SAMPLES = _certification_runner.RECIPE_N_SAMPLES
RECIPE_NUM_CHAINS = _certification_runner.RECIPE_NUM_CHAINS
RECIPE_SEED = _certification_runner.RECIPE_SEED
RECIPE_N_CHUNKS = _certification_runner.RECIPE_N_CHUNKS
RECIPE_TARGET_ACCEPTANCE = _certification_runner.RECIPE_TARGET_ACCEPTANCE

_CATALOG_ROOT = _certification_runner.DEFAULT_CATALOG_ROOT
_OUTCOMES_FILE = _certification_runner.DEFAULT_OUTCOMES_FILE

__all__ = [
    "CellResult",
    "emit_low_recipe_for_cell",
]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate, execute, evaluate, and record one recipe certification")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--n-warmup", type=int, default=RECIPE_N_WARMUP)
    parser.add_argument("--n-samples", type=int, default=RECIPE_N_SAMPLES)
    parser.add_argument("--num-chains", type=int, default=RECIPE_NUM_CHAINS)
    parser.add_argument("--seed", type=int, default=RECIPE_SEED)
    parser.add_argument("--target-acceptance", type=float, default=None)
    parser.add_argument("--num-integration-steps", type=int, default=None)
    args = parser.parse_args()

    sampler_overrides: dict[str, Any] | None = None
    if args.num_integration_steps is not None:
        sampler_overrides = {"num_integration_steps": args.num_integration_steps}
    result = emit_low_recipe_for_cell(
        model_name=args.model,
        warmup_name=args.warmup,
        sampler_name=args.sampler,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        target_acceptance=args.target_acceptance,
        sampler_kwargs_override=sampler_overrides,
    )
    sys.exit(0 if result.verdict == "PASS" else 1)


if __name__ == "__main__":
    _main()
