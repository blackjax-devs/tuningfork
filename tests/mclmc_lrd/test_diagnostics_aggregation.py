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
from blackjax.mcmc.metrics import LowRankInverseMassMatrix


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


@pytest.mark.fast
def test_cert_sweep_bakes_from_adapted_params_exactly(tmp_path, monkeypatch):
    """R1 regression: baked recipe step_size/L must equal best cert seed's adapted_params.

    The bake step must use adapted_params carried from _run_cert_seed, NOT re-run
    warmup with a fresh key (which would use a different key derivation and produce
    a fourth, uncertified warmup realisation whose params differ from gate_evidence).

    Strategy: monkeypatch _run_cert_seed to return a PASS result with sentinel
    step_size/L values, then assert the saved recipe carries those exact values.

    Requires numpyro: emit_mclmc_lrd imports MODELS → _numpyro at module level.
    Skipped automatically when numpyro is not installed.
    """
    pytest.importorskip("numpyro")
    # Sentinel adapted_params with known step_size/L — chosen to be recognisable
    # floats that a re-run warmup would never accidentally reproduce.
    _sentinel_step_size = jnp.array(0.12345678)
    _sentinel_L = jnp.array(9.87654321)
    _sentinel_imm = LowRankInverseMassMatrix(
        sigma=jnp.ones(5),
        U=jnp.eye(5, 3),
        lam=jnp.array([1.0, 0.5, 0.25]),
    )

    fake_result = {
        "seed": 42,
        "verdict": "PASS",
        "rhat_max": 1.001,
        "min_bulk_ess": 500.0,
        "n_divergences": 0,
        "div_rate": 0.0,
        "ess_per_grad": 0.1,
        "total_grad_evals": 5000,
        "wall_seconds": 0.5,
        "adapted_params": {
            "step_size": _sentinel_step_size,
            "L": _sentinel_L,
            "inverse_mass_matrix": _sentinel_imm,
        },
    }

    import tuningfork.recipes.emit_mclmc_lrd as _mod

    monkeypatch.setattr(_mod, "_run_cert_seed", lambda **_kw: fake_result)

    from tuningfork.recipes.emit_mclmc_lrd import _emit_lrd_cert_sweep

    written = _emit_lrd_cert_sweep(
        ["ill_cond_50"],
        cert_seeds=(42,),
        n_warmup=100,
        n_samples=10,
        num_chains=2,
        k_rank=3,
        catalog_root=tmp_path,
        variant_label="mclmc_lrd",
    )

    assert len(written) == 1, f"Expected 1 written recipe, got {len(written)}"

    from tuningfork.recipes import Recipe

    recipe = Recipe.load(written[0])

    # The baked recipe must carry exactly the sentinel step_size and L.
    saved_step = recipe.base_method_params["step_size"]
    saved_L = recipe.base_method_params["L"]
    assert abs(saved_step - float(_sentinel_step_size)) < 1e-7, (
        f"R1 regression: step_size mismatch — recipe has {saved_step}, "
        f"adapted_params had {float(_sentinel_step_size)}. "
        "Bake step re-ran warmup instead of using adapted_params."
    )
    assert abs(saved_L - float(_sentinel_L)) < 1e-7, (
        f"R1 regression: L mismatch — recipe has {saved_L}, "
        f"adapted_params had {float(_sentinel_L)}. "
        "Bake step re-ran warmup instead of using adapted_params."
    )
    # IMM should be saved as sidecar (LRD namedtuple → auto-sidecar path).
    assert (
        recipe.inverse_mass_matrix_path is not None
    ), "Expected LRD sidecar to be written for LowRankInverseMassMatrix"
