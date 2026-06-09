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
"""Fast regression tests for diagnostics leaf-aggregation helpers.

These tests are pure-JAX (no sampling) and intentionally kept out of
test_mclmc_lrd.py (which carries pytestmark=slow) so that CI runs them
without the --slow flag.
"""

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.fast
def test_diagnostics_aggregation_mixed_shape_pytree():
    """Regression: rhat/ESS leaf-aggregation must not crash on mixed-shape pytrees.

    stoch_vol has h:(500,) + phi/sigma/mu:() — mixing ndim=1 and ndim=0 leaves.
    The old pattern ``jnp.array(jax.tree.leaves(tree))`` raises
    ``TypeError: Cannot concatenate arrays with different numbers of dimensions``
    because JAX prepends a dim per element and then calls jnp.concatenate, which
    requires uniform ndim.

    Fix in emit_mclmc_lrd._run_cert_seed (commit 76e1dfd):
    ``jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(tree)])`` ravels
    every leaf to 1-D before concatenation regardless of original shape.

    This test is pure-JAX (no sampling) — exercises the aggregation logic with
    synthetic trees shaped like stoch_vol's parameter pytree.
    """
    # Synthetic rhat_tree: h leaf is vector (500,), scalars are shape ()
    rhat_tree = {
        "h": jnp.full((500,), 1.02),  # vector — highest rhat
        "mu": jnp.array(1.00),
        "phi": jnp.array(1.01),
        "sigma": jnp.array(1.005),
    }
    ess_tree = {
        "h": jnp.full((500,), 150.0),  # vector
        "mu": jnp.array(200.0),
        "phi": jnp.array(180.0),
        "sigma": jnp.array(120.0),  # scalar — lowest ESS
    }

    # Must not raise "Cannot concatenate arrays with different numbers of dimensions".
    rhat_max = float(
        jnp.max(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(rhat_tree)]))
    )
    min_bulk_ess = float(
        jnp.min(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(ess_tree)]))
    )

    # h leaf provides the worst rhat (1.02); all scalar leaves are <= 1.02.
    assert abs(rhat_max - 1.02) < 1e-5, f"rhat_max expected ≈1.02, got {rhat_max}"
    # sigma scalar provides the lowest ESS (120.0).
    assert (
        abs(min_bulk_ess - 120.0) < 1e-3
    ), f"min_bulk_ess expected ≈120.0, got {min_bulk_ess}"
