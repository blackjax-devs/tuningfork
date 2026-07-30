from pathlib import Path
from types import SimpleNamespace

import pytest

from tuningfork.catalog import emit_script, load_recipe
from tuningfork.recipes._base import Effort
from tuningfork.recipes._emit_script import _build_inference_loop

pytestmark = pytest.mark.fast
_CATALOG_ROOT = Path(__file__).parents[2] / "tuningfork" / "catalog"


def _recipe() -> SimpleNamespace:
    return SimpleNamespace(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 10},
            }
        ],
        calibration_budget={"n_samples": 20, "num_chains": 2},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
        warmup_num_chains=None,
        gate_evidence={},
    )


def test_emit_script_plan_overrides_win():
    script = emit_script(
        _recipe(),
        sampler_seed=8,
        num_samples=9,
        num_chains=3,
        num_warmup=[7],
        warmup_num_chains=[1],
        progress_bar=True,
    )

    assert "jax.random.key(8)" in script
    assert "_NUM_SAMPLES = 9" in script
    assert "keys = jax.random.split(rng_key, 3)" in script
    assert 'with blackjax.progress_bar(label="sampling")' in script
    assert "n_warmup=7" in script


def test_emit_script_rejects_invalid_plan_override():
    with pytest.raises(ValueError, match="sampler_seed"):
        emit_script(_recipe(), sampler_seed=-1)


def test_inference_loop_uses_resolved_reinit_seed():
    loop = _build_inference_loop(
        num_samples=1,
        sampler_seed=5,
        reinit_seed=1003,
        num_chains=1,
        use_progress_bar=False,
        warmup_is_perchain=False,
        warmup_init_is_single_chain=False,
        needs_state_reinit=True,
    )
    assert "jax.random.split(jax.random.key(1003)," in loop


def test_multichain_warmup_reinitializes_each_chain():
    loop = _build_inference_loop(
        num_samples=1,
        sampler_seed=5,
        reinit_seed=1003,
        num_chains=3,
        use_progress_bar=False,
        warmup_is_perchain=True,
        warmup_init_is_single_chain=False,
        needs_state_reinit=True,
    )
    assert "jax.random.split(jax.random.key(1003), num_chains)" in loop
    assert '_batched_step_size = _adapted_params["step_size"]' in loop
    assert "_state_reinit(ss, imm, s.position, k)" in loop
    assert "jax.vmap(_step_one_chain)" in loop


def test_single_chain_warmup_uses_shared_adaptation():
    loop = _build_inference_loop(
        num_samples=1,
        sampler_seed=5,
        reinit_seed=1003,
        num_chains=3,
        use_progress_bar=False,
        warmup_is_perchain=False,
        warmup_init_is_single_chain=False,
        needs_state_reinit=True,
    )
    assert '_shared_step_size = _adapted_params["step_size"]' in loop
    assert '_shared_imm = _adapted_params["inverse_mass_matrix"]' in loop
    assert "_state_reinit(_shared_step_size, _shared_imm, s.position, k)" in loop
    assert '_batched_step_size = _adapted_params["step_size"]' not in loop


def test_emit_no_warmup_does_not_invent_a_warmup_step():
    recipe = _recipe()
    recipe.warmup_name = "no_warmup"
    recipe.warmup_params = {"n_warmup": 99}
    recipe.warmups = [{"name": "no_warmup", "params": {"n_warmup": 99}}]
    script = emit_script(recipe)
    assert "_adapted_params = {}" in script
    assert "n_warmup=1" not in script


def test_laplace_omitted_topology_emits_per_chain_warmup_and_reinit():
    recipe = load_recipe(
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__laplace_dmhmc__window_adaptation_diag_imm.json"
    )
    script = emit_script(
        recipe,
        num_samples=2,
        num_chains=4,
        reinit_seed=12,
    )

    assert "_warmup_keys = jax.random.split(" in script
    assert "_init_positions = jax.tree.map(" in script
    assert "_warmup_is_perchain = True" in script
    assert "jax.random.split(jax.random.key(12), num_chains)" in script
    assert "_state_reinit(ss, imm, s.position, k)" in script


def test_laplace_explicit_single_chain_topology_emits_shared_warmup_and_reinit():
    recipe = load_recipe(
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__laplace_dmhmc__window_adaptation_diag_imm.json"
    )
    script = emit_script(
        recipe,
        num_samples=2,
        num_chains=4,
        warmup_num_chains=[1],
    )

    assert "_warmup_key = jax.random.fold_in(" in script
    assert "_warmup_is_perchain = False" in script
    assert "_state_reinit(_shared_step_size, _shared_imm, s.position, k)" in script
    assert "_state_reinit(ss, imm, s.position, k)" not in script


def test_catalog_two_phase_laplace_recipe_uses_ordered_warmup_stages():
    recipe = load_recipe(
        _CATALOG_ROOT
        / "gp_regression"
        / "recipes"
        / "high__laplace_mhmc__window_adaptation_dense_imm__inner_laplace_hmc.json"
    )
    script = emit_script(recipe, num_samples=2)

    assert "Phase 1: window_adaptation_diag_imm" in script
    assert "Phase 2: window_adaptation_dense_imm" in script
    assert "# Single-chain warmup: shared step_size + IMM across all chains." in script
