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
"""Shared boilerplate for Laplace-family base methods (hmc, dhmc, mhmc, dmhmc)."""

import jax.numpy as jnp

try:
    # Available after blackjax PR #928 merges to main.
    from blackjax.mcmc.laplace_marginal import (
        laplace_lbfgs_grad_evals as _laplace_grad_count,
    )
except ImportError:
    # Fallback: ×5 heuristic until blackjax is updated.
    def _laplace_grad_count(info):  # type: ignore[misc]
        return jnp.asarray(info.num_integration_steps * 5)


__all__ = ["_laplace_grad_count"]
