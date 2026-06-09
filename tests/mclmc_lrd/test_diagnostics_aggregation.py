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
    # Use a BATCHED IMM (num_chains=2, d=5, k=3) matching the certified runner's
    # output format.  The bake step must de-broadcast to (d,)/(d,k)/(k,) before
    # writing the sidecar — this test verifies that slice [0] is taken correctly.
    _sigma_row = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # sentinel row, shape (5,)
    _U_row = jnp.eye(5, 3)  # sentinel U slice, shape (5, 3)
    _lam_row = jnp.array([1.0, 0.5, 0.25])  # sentinel lam slice, shape (3,)
    _sentinel_imm = LowRankInverseMassMatrix(
        sigma=jnp.stack([_sigma_row, _sigma_row * 2]),  # (2, 5) — two identical chains
        U=jnp.stack([_U_row, _U_row]),  # (2, 5, 3)
        lam=jnp.stack([_lam_row, _lam_row]),  # (2, 3)
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

    # De-broadcast assertion: sidecar must contain UNBATCHED (d,)/(d,k)/(k,) arrays,
    # NOT the (num_chains,d)/(num_chains,d,k)/(num_chains,k) form from adapted_params.
    saved_imm = recipe.load_imm_sidecar(tmp_path)
    assert saved_imm is not None, "load_imm_sidecar returned None after save"
    assert saved_imm.sigma.shape == (5,), (
        f"De-broadcast regression: expected sigma.shape=(5,), got {saved_imm.sigma.shape}. "
        "Bake must take leaf[0] of the batched IMM before writing the sidecar."
    )
    assert saved_imm.U.shape == (
        5,
        3,
    ), f"De-broadcast regression: expected U.shape=(5,3), got {saved_imm.U.shape}."
    assert saved_imm.lam.shape == (
        3,
    ), f"De-broadcast regression: expected lam.shape=(3,), got {saved_imm.lam.shape}."
    # Verify the de-broadcast took chain-0's values (not chain-1's which differ).
    assert jnp.allclose(
        saved_imm.sigma, _sigma_row
    ), "De-broadcast: sigma values don't match chain-0 slice of batched IMM."

    # k_rank provenance must be in calibration_budget.baked_from.
    baked_from = recipe.calibration_budget.get("baked_from", {})
    assert baked_from.get("k_rank") == 3, (
        f"k_rank not recorded in calibration_budget.baked_from — got {baked_from!r}. "
        "Provenance ruling: k_rank used during cert run must be stored so the golden "
        "can be reproduced at the same rank."
    )


@pytest.mark.fast
def test_save_updates_self_inverse_mass_matrix_path(tmp_path):
    """M1 regression: save(imm_sidecar='auto') must update the in-memory Recipe.

    Recipe is frozen=True.  The old save() wrote inverse_mass_matrix_path into the
    JSON but left self.inverse_mass_matrix_path=None, so recipe.load_imm_sidecar()
    would silently return None on the same object that just wrote the sidecar.

    Fix (commit fixing M1): object.__setattr__ patches the frozen instance so the
    in-memory recipe is consistent with the on-disk artifact after save().
    """
    from tuningfork.recipes._base import Effort, Recipe

    lrd_imm = LowRankInverseMassMatrix(
        sigma=jnp.ones(4),
        U=jnp.eye(4, 2),
        lam=jnp.array([2.0, 1.0]),
    )
    recipe = Recipe(
        model_name="test_model",
        base_method_name="mclmc",
        warmup_name="",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "L": 1.0, "inverse_mass_matrix": lrd_imm},
        warmup_params={},
        warmups=[],
        headline_metric=0.05,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        notes="",
        variant_label="mclmc_lrd",
        gate_evidence={},
        tuning_seed=42,
        tuningfork_version="0.0.0",
        blackjax_version="0.0.0",
        jax_version="0.0.0",
        timestamp_utc="2026-01-01T00:00:00Z",
    )

    assert (
        recipe.inverse_mass_matrix_path is None
    ), "Pre-condition: path unset before save"

    recipe.save(tmp_path, imm_sidecar="auto")

    # M1: in-memory recipe must now carry the sidecar relative path.
    assert recipe.inverse_mass_matrix_path is not None, (
        "M1 regression: save(imm_sidecar='auto') did not update self.inverse_mass_matrix_path. "
        "The in-memory recipe is inconsistent with the on-disk artifact."
    )
    # The in-memory path must resolve to an existing sidecar file.
    sidecar = tmp_path / recipe.inverse_mass_matrix_path
    assert sidecar.exists(), f"Sidecar file missing at {sidecar}"

    # The in-memory recipe should also no longer carry the raw IMM in base_method_params
    # (it was extracted to the sidecar).
    assert "inverse_mass_matrix" not in recipe.base_method_params, (
        "M1: base_method_params still contains 'inverse_mass_matrix' after save — "
        "should have been removed when the sidecar was written."
    )

    # Round-trip: load_imm_sidecar on the SAME in-memory object must work.
    loaded_imm = recipe.load_imm_sidecar(tmp_path)
    assert loaded_imm is not None, (
        "M1: recipe.load_imm_sidecar() returned None on the in-memory recipe after save. "
        "inverse_mass_matrix_path was not updated."
    )


@pytest.mark.fast
def test_cert_sweep_bake_no_double_squeeze_when_imm_already_unbatched(
    tmp_path, monkeypatch
):
    """De-broadcast guard regression: already-unbatched LRD IMM must not be re-indexed.

    _run_cert_seed runs the warmup with num_chains=1 and then calls
    squeeze_single_chain — so adapted_params["inverse_mass_matrix"] arrives at the
    bake step with sigma shape (d,), NOT (1, d).  The old de-broadcast code did
    ``sigma[0]`` unconditionally, which gives a scalar instead of (d,) and corrupts
    the sidecar.

    Fix (ndim > 1 guard): de-broadcast is skipped when sigma.ndim == 1 (already
    unbatched).

    This test provides an UNBATCHED sentinel (sigma shape (5,)) and asserts that the
    saved sidecar carries the full (5,) vector — not a scalar or (k,) slice.
    """
    pytest.importorskip("numpyro")

    _sigma = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # shape (5,) — already squeezed
    _U = jnp.eye(5, 3)  # shape (5, 3)
    _lam = jnp.array([1.0, 0.5, 0.25])  # shape (3,)
    _unbatched_imm = LowRankInverseMassMatrix(sigma=_sigma, U=_U, lam=_lam)

    fake_result = {
        "seed": 77,
        "verdict": "PASS",
        "rhat_max": 1.001,
        "min_bulk_ess": 500.0,
        "n_divergences": 0,
        "div_rate": 0.0,
        "ess_per_grad": 0.1,
        "total_grad_evals": 5000,
        "wall_seconds": 0.5,
        "adapted_params": {
            "step_size": jnp.array(0.25),
            "L": jnp.array(3.0),
            "inverse_mass_matrix": _unbatched_imm,
        },
    }

    import tuningfork.recipes.emit_mclmc_lrd as _mod

    monkeypatch.setattr(_mod, "_run_cert_seed", lambda **_kw: fake_result)

    from tuningfork.recipes.emit_mclmc_lrd import _emit_lrd_cert_sweep

    written = _emit_lrd_cert_sweep(
        ["ill_cond_50"],
        cert_seeds=(77,),
        n_warmup=100,
        n_samples=10,
        num_chains=1,
        k_rank=3,
        catalog_root=tmp_path,
        variant_label="mclmc_lrd",
    )

    assert len(written) == 1

    from tuningfork.recipes import Recipe

    recipe = Recipe.load(written[0])
    saved_imm = recipe.load_imm_sidecar(tmp_path)
    assert saved_imm is not None

    # Guard regression: sigma must still be (5,) — NOT scalar or first element.
    assert saved_imm.sigma.shape == (5,), (
        f"Double-squeeze bug: expected sigma.shape=(5,), got {saved_imm.sigma.shape}. "
        "The ndim>1 guard on de-broadcast is missing or broken."
    )
    assert saved_imm.U.shape == (
        5,
        3,
    ), f"Double-squeeze bug: expected U.shape=(5,3), got {saved_imm.U.shape}."
    assert saved_imm.lam.shape == (
        3,
    ), f"Double-squeeze bug: expected lam.shape=(3,), got {saved_imm.lam.shape}."
    # Values must be unchanged (no indexing applied).
    assert jnp.allclose(saved_imm.sigma, _sigma), (
        "Double-squeeze bug: sigma values modified — de-broadcast applied to "
        "already-unbatched IMM."
    )
