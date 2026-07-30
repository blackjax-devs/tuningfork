import dataclasses
import math
from types import SimpleNamespace

import pytest

from tuningfork.recipes._execution_plan import ExecutionOverrides, canonical_json
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan

pytestmark = pytest.mark.fast


def recipe(**changes):
    data = dict(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort="low",
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        calibration_budget={"n_samples": 20, "num_chains": 2},
        tuning_seed=4,
        warmup_inner_kernel=None,
        init_strategy=None,
        step_policy=None,
        variant_label=None,
    )
    data.update(changes)
    return SimpleNamespace(**data)


def test_precedence_and_warmup_shape():
    plan = resolve_execution_plan(
        recipe(),
        ExecutionOverrides(
            sampler_seed=8,
            num_samples=9,
            num_chains=3,
            num_warmup=[7],
            warmup_num_chains=[1],
        ),
    )
    assert (
        plan.config.sampler_seed,
        plan.config.num_samples,
        plan.config.num_chains,
    ) == (8, 9, 3)
    assert plan.config.warmup_stages[0].num_warmup == 7
    assert plan.config.warmup_stages[0].num_chains == 1
    assert plan.config.reinit_seed == 1003
    assert plan.artifact_filename == "mvn_10__hmc__window_adaptation_diag_imm.draws.npz"
    assert plan.recipe_ref == "mvn_10/low__hmc__window_adaptation_diag_imm"


def test_non_laplace_multiphase_execution_fails_closed():
    r = recipe(
        warmups=[
            {"name": "a", "params": {"n_warmup": 2}},
            {"name": "b", "params": {"n_warmup": 3}},
        ],
        warmup_num_chains=[2, 2],
    )
    with pytest.raises(ValueError):
        resolve_execution_plan(r, ExecutionOverrides(num_warmup=4))
    with pytest.raises(NotImplementedError, match="multi-phase code generation"):
        resolve_execution_plan(r)


def test_single_phase_flat_fields_win_over_compatibility_list():
    r = recipe(
        warmup_name="flat",
        warmup_params={"n_warmup": 17, "target_acceptance": 0.9},
        warmups=[{"name": "stale", "params": {"n_warmup": 2}}],
    )
    plan = resolve_execution_plan(r)
    assert plan.config.warmup_stages[0].name == "flat"
    assert plan.config.warmup_stages[0].num_warmup == 17
    assert plan.config.warmup_params["target_acceptance"] == 0.9


def test_seed_zero_and_default_budget_fallbacks():
    p = resolve_execution_plan(
        recipe(tuning_seed=0, calibration_budget={"n_samples": 0})
    )
    assert (p.config.sampler_seed, p.config.reinit_seed, p.config.num_samples) == (
        1,
        999,
        1000,
    )
    p = resolve_execution_plan(
        recipe(), ExecutionOverrides(sampler_seed=0, reinit_seed=0)
    )
    assert (p.config.sampler_seed, p.config.reinit_seed) == (0, 0)
    for field in ("sampler_seed", "reinit_seed"):
        with pytest.raises(ValueError):
            resolve_execution_plan(recipe(), ExecutionOverrides(**{field: -1}))


def test_window_warmup_accepts_explicit_single_chain_topology():
    assert (
        resolve_execution_plan(recipe(), ExecutionOverrides(warmup_num_chains=[1]))
        .config.warmup_stages[0]
        .num_chains
        == 1
    )


def test_canonical_hashes_and_rejection():
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    for value in (math.nan, math.inf, -math.inf, {"nested": [math.nan]}):
        with pytest.raises(ValueError):
            canonical_json(value)
    with pytest.raises(TypeError):
        canonical_json({1: "not a string key"})
    with pytest.raises(TypeError):
        canonical_json({"nested": {object(): 1}})
    with pytest.raises(TypeError):
        canonical_json(object())


def test_material_mutation_changes_hash_but_presentation_does_not():
    a = resolve_execution_plan(recipe())
    b = resolve_execution_plan(recipe(base_method_params={"step_size": 0.2}))
    assert a.executable_config_hash != b.executable_config_hash
    assert a.plan_hash != b.plan_hash
    assert a.plan_hash == resolve_execution_plan(recipe(notes="presentation")).plan_hash


def test_hashes_ignore_mapping_insertion_order():
    first = resolve_execution_plan(
        recipe(
            base_method_params={"a": 1, "b": {"x": 2, "y": 3}},
            warmup_params={"n_warmup": 10, "target_acceptance": 0.8},
        )
    )
    second = resolve_execution_plan(
        recipe(
            base_method_params={"b": {"y": 3, "x": 2}, "a": 1},
            warmup_params={"target_acceptance": 0.8, "n_warmup": 10},
        )
    )
    assert first.executable_config_hash == second.executable_config_hash
    assert first.plan_hash == second.plan_hash


@pytest.mark.parametrize(
    "overrides",
    [
        ExecutionOverrides(sampler_seed=8),
        ExecutionOverrides(reinit_seed=8),
        ExecutionOverrides(num_samples=21),
        ExecutionOverrides(num_chains=3),
        ExecutionOverrides(num_warmup=[11]),
        ExecutionOverrides(warmup_num_chains=[1]),
        ExecutionOverrides(progress_bar=True),
    ],
)
def test_execution_overrides_change_the_plan_hash(overrides):
    baseline = resolve_execution_plan(recipe())
    changed = resolve_execution_plan(recipe(), overrides)
    assert changed.executable_config_hash != baseline.executable_config_hash
    assert changed.plan_hash != baseline.plan_hash


def test_input_mutation_does_not_change_resolved_plan():
    base = {"nested": [1, 2]}
    warm = {"n_warmup": 10, "nested": {"x": [3]}}
    r = recipe(base_method_params=base, warmup_params=warm)
    p = resolve_execution_plan(r)
    digest = (p.executable_config_hash, p.plan_hash)
    base["nested"].append(9)
    warm["nested"]["x"].append(8)
    r.base_method_params["nested"].append(7)
    assert digest == (p.executable_config_hash, p.plan_hash)


def test_resolved_nested_values_are_immutable():
    p = resolve_execution_plan(recipe(base_method_params={"nested": [1]}))
    with pytest.raises(TypeError):
        p.config.base_method_params["nested"] += (2,)  # type: ignore[index]
    with pytest.raises(TypeError):
        p.config.base_method_params["nested"][0] = 2  # type: ignore[index]


def test_material_fields_affect_config_hash_and_metadata_does_not():
    fields = {
        "base_method_params": {"step_size": 0.2},
        "warmup_params": {"n_warmup": 99},
        "init_strategy": {"type": "zero"},
        "tuning_seed": 8,
        "model_name": "gp_regression",
        "base_method_name": "nuts",
    }
    baseline = resolve_execution_plan(recipe())
    for name, value in fields.items():
        changed = resolve_execution_plan(recipe(**{name: value}))
        assert changed.executable_config_hash != baseline.executable_config_hash, name
    metadata = resolve_execution_plan(
        recipe(effort="high", notes="x", verdict="REVIEW", workflow="y")
    )
    assert metadata.plan_hash == baseline.plan_hash

    dynamic = recipe(base_method_name="dynamic_hmc")
    dynamic_with_policy = recipe(
        base_method_name="dynamic_hmc",
        step_policy={"kind": "uniform_int", "low": 1, "high": 10},
    )
    assert (
        resolve_execution_plan(dynamic).executable_config_hash
        != resolve_execution_plan(dynamic_with_policy).executable_config_hash
    )


def test_requires_x64_comes_from_model_registry():
    assert (
        resolve_execution_plan(recipe(model_name="mvn_10")).config.requires_x64 is False
    )
    assert (
        resolve_execution_plan(recipe(model_name="gp_regression")).config.requires_x64
        is True
    )


def test_plan_is_frozen_and_x64_is_typed():
    p = resolve_execution_plan(recipe())
    assert isinstance(p.config.requires_x64, bool)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.artifact_filename = "x"  # type: ignore[misc]


@pytest.mark.parametrize("override", [None, 0, 8, [0], [8]])
def test_no_warmup_is_canonical_zero_and_ignores_topology(override):
    r = recipe(
        warmup_name="no_warmup",
        warmup_params={"n_warmup": 17},
        warmups=[{"name": "no_warmup", "params": {"n_warmup": 17}}],
    )
    p = resolve_execution_plan(
        r,
        ExecutionOverrides(
            num_warmup=override,
            warmup_num_chains=[1],
        ),
    )
    assert p.config.warmup_stages[0].num_warmup == 0
    assert p.config.warmup_stages[0].num_chains == p.config.num_chains


def test_no_warmup_rejects_negative_override():
    r = recipe(
        warmup_name="no_warmup",
        warmup_params={"n_warmup": 17},
        warmups=[{"name": "no_warmup", "params": {"n_warmup": 17}}],
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_execution_plan(r, ExecutionOverrides(num_warmup=-1))


@pytest.mark.parametrize("warmup_num_chains", [[2], [4]])
def test_unsupported_window_topology_fails_closed(warmup_num_chains):
    with pytest.raises(NotImplementedError, match="warmup chain topology"):
        resolve_execution_plan(
            recipe(),
            ExecutionOverrides(num_chains=3, warmup_num_chains=warmup_num_chains),
        )


def test_single_phase_laplace_defaults_to_sampling_chain_topology():
    r = recipe(
        base_method_name="laplace_hmc",
        warmup_num_chains=None,
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
    )
    plan = resolve_execution_plan(r, ExecutionOverrides(num_chains=2))
    assert plan.config.warmup_stages[0].num_chains == 2

    explicit_single = resolve_execution_plan(
        r, ExecutionOverrides(num_chains=2, warmup_num_chains=[1])
    )
    assert explicit_single.config.warmup_stages[0].num_chains == 1


def test_multiphase_laplace_rejects_sampling_chain_topology():
    r = recipe(
        base_method_name="laplace_hmc",
        warmup_name="window_adaptation_dense_imm",
        warmups=[
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 2}},
            {"name": "window_adaptation_dense_imm", "params": {"n_warmup": 3}},
        ],
    )
    with pytest.raises(NotImplementedError, match="W=2, S=2"):
        resolve_execution_plan(r, ExecutionOverrides(num_chains=2))


def test_two_phase_laplace_accepts_explicit_single_chain_topology():
    r = recipe(
        base_method_name="laplace_hmc",
        warmup_name="window_adaptation_dense_imm",
        warmups=[
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 2}},
            {"name": "window_adaptation_dense_imm", "params": {"n_warmup": 3}},
        ],
        warmup_num_chains=[1, 1],
    )
    plan = resolve_execution_plan(r, ExecutionOverrides(num_chains=2))
    assert [stage.num_chains for stage in plan.config.warmup_stages] == [1, 1]


@pytest.mark.parametrize(
    "warmups",
    [
        [
            {"name": "window_adaptation_dense_imm", "params": {"n_warmup": 2}},
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 3}},
        ],
        [
            {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 2}},
            {"name": "window_adaptation_dense_imm", "params": {"n_warmup": 3}},
            {"name": "window_adaptation_dense_imm", "params": {"n_warmup": 4}},
        ],
    ],
)
def test_multiphase_laplace_rejects_unimplemented_phase_choreography(warmups):
    r = recipe(
        base_method_name="laplace_hmc",
        warmups=warmups,
        warmup_num_chains=[1] * len(warmups),
    )
    with pytest.raises(NotImplementedError, match="diagonal then dense"):
        resolve_execution_plan(r, ExecutionOverrides(num_chains=2))
