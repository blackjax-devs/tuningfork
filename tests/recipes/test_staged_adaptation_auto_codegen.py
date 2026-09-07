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

"""Descriptor, plan, and codegen coverage for ``staged_adaptation_auto``.

The property under test throughout is the *chain topology*.  The window
family at ``W=S`` emits ``jax.vmap`` over S independent ``window_adaptation``
runs and returns per-chain adapted parameters.  ``staged_adaptation_auto``
emits exactly one ``blackjax.staged_adaptation(metric="auto", n_chains=W)``
call whose controller pools the W chains and publishes one shared
``(step_size, inverse_mass_matrix)``.  A refactor that silently turned the
joint call back into a vmap of independent warmups would keep every shape
intact, so the tests assert the call shape directly.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.catalog import emit_script
from tuningfork.model import MODELS
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan
from tuningfork.warmup import WARMUPS

WARMUP = "staged_adaptation_auto"

# The joint controller is only reachable on a blackjax whose
# ``staged_adaptation`` accepts ``n_chains``.  No PyPI release has it as of
# 1.6.2, so the runtime half of this module is capability-gated while the
# descriptor / plan / emission half runs everywhere.
_HAS_N_CHAINS = False
_BLACKJAX_ORIGIN = "<import failed>"
try:  # pragma: no cover - import guard, not a branch under test
    import blackjax as _blackjax

    _HAS_N_CHAINS = (
        "n_chains" in inspect.signature(_blackjax.staged_adaptation).parameters
    )
    _BLACKJAX_ORIGIN = f"{_blackjax.__version__} from {_blackjax.__file__}"
except Exception:  # pragma: no cover - blackjax always present in this suite
    _HAS_N_CHAINS = False

# The dedicated pinned-upstream CI job sets this.  There, a skipped or
# uncollected joint case is a FAILING gate, not an acceptable outcome, so the
# capability shortfall is raised at import (a collection error) rather than
# quietly degrading to a skip.  Capability is probed on the actual signature,
# never on the version string.
_REQUIRE_JOINT = os.environ.get("TUNINGFORK_REQUIRE_JOINT_CONTROLLER", "0") not in {
    "",
    "0",
}
if _REQUIRE_JOINT and not _HAS_N_CHAINS:  # pragma: no cover - CI gate path
    raise RuntimeError(
        "TUNINGFORK_REQUIRE_JOINT_CONTROLLER is set, but the imported "
        "blackjax.staged_adaptation does not accept n_chains. Imported "
        f"blackjax: {_BLACKJAX_ORIGIN}. The pinned-upstream overlay did not "
        "take effect, or a re-sync restored the released package."
    )

requires_joint_controller = pytest.mark.skipif(
    not _HAS_N_CHAINS,
    reason=(
        "installed blackjax.staged_adaptation does not accept n_chains "
        "(multi-chain metric='auto' controller)"
    ),
)

requires_released_without_n_chains = pytest.mark.skipif(
    _HAS_N_CHAINS,
    reason=(
        "installed blackjax.staged_adaptation accepts n_chains; the "
        "capability-error path is only reachable on a released blackjax"
    ),
)


def _recipe(
    *,
    base: str = "nuts",
    num_chains: int = 6,
    n_warmup: int = 200,
    max_grad_budget: int | None = 20_000,
) -> Recipe:
    params: dict[str, object] = {"n_warmup": n_warmup, "num_chains": num_chains}
    if max_grad_budget is not None:
        params["max_grad_budget"] = max_grad_budget
    return replace(
        Recipe.from_default_config(MODELS["mvn_10"], BASE_METHODS[base]),
        warmup_name=WARMUP,
        warmup_params=params,
        calibration_budget={"num_chains": num_chains, "n_samples": 5},
        effort=Effort.LOW,
    )


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_descriptor_is_registered_with_evidenced_compatibility() -> None:
    descriptor = WARMUPS[WARMUP]
    assert descriptor.name == WARMUP
    # Only the kernels that both satisfy the upstream staged_adaptation kernel
    # contract AND report a per-step num_integration_steps, which is the only
    # warmup gradient cost this warmup claims.  barker runs upstream but has no
    # per-step integration count, so its warmup cost would be unknown; mala,
    # ghmc and dynamic_hmc are rejected by the kernel contract; rmhmc is
    # excluded by upstream docs (its kernel takes mass_matrix).
    assert descriptor.compatible_methods == ("nuts", "hmc", "mhmc")
    for excluded in ("barker", "mala", "ghmc", "dynamic_hmc", "rmhmc", "mclmc"):
        assert not descriptor.is_compatible(excluded)
    assert [space.name for space in descriptor.default_hp_space] == ["max_grad_budget"]


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize("warmup_chains", [1, 6])
def test_plan_accepts_both_supported_topologies(warmup_chains: int) -> None:
    plan = resolve_execution_plan(
        _recipe(num_chains=6),
        ExecutionOverrides(warmup_num_chains=[warmup_chains]),
    )
    (stage,) = plan.config.warmup_stages
    assert stage.name == WARMUP
    assert stage.num_chains == warmup_chains
    assert plan.config.num_chains == 6


@pytest.mark.fast
def test_plan_rejects_unsupported_warmup_chain_count() -> None:
    with pytest.raises(NotImplementedError, match="not supported by code generation"):
        resolve_execution_plan(
            _recipe(num_chains=6), ExecutionOverrides(warmup_num_chains=[3])
        )


@pytest.mark.fast
def test_plan_rejects_incompatible_base_method() -> None:
    with pytest.raises(ValueError, match="incompatible with base method"):
        resolve_execution_plan(_recipe(base="mala"))


@pytest.mark.fast
def test_max_grad_budget_is_a_material_plan_field() -> None:
    """Mutating it must move the plan hash (codegen recipe-to-code gate)."""
    base = resolve_execution_plan(_recipe(max_grad_budget=20_000))
    bumped = resolve_execution_plan(_recipe(max_grad_budget=40_000))
    assert base.plan_hash != bumped.plan_hash
    assert base.executable_config_hash != bumped.executable_config_hash
    assert (
        base.config.warmup_stages[0].params["max_grad_budget"] == 20_000
    ), "max_grad_budget must survive into the normalized plan, not just the recipe"


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _emit(
    *,
    num_chains: int = 6,
    n_warmup: int = 200,
    max_grad_budget: int | None = 20_000,
    warmup_num_chains: list[int] | None = None,
) -> str:
    recipe = _recipe(
        num_chains=num_chains, n_warmup=n_warmup, max_grad_budget=max_grad_budget
    )
    return emit_script(recipe, num_samples=5, warmup_num_chains=warmup_num_chains)


@pytest.mark.fast
def test_joint_emission_is_one_shared_controller_not_a_vmap_of_warmups() -> None:
    source = _emit(num_chains=6, n_warmup=200)
    ast.parse(source)

    # Exactly one controller construction, carrying the joint chain count.
    assert source.count("_warmup = blackjax.staged_adaptation(") == 1
    assert source.count("blackjax.staged_adaptation(") == 1
    assert '    metric="auto",' in source
    assert "    n_chains=6," in source
    assert "    max_grad_budget=20000," in source

    # The joint call is NOT vmapped, and no per-chain warmup runner is emitted.
    assert "@jax.vmap\ndef _run_one_warmup" not in source
    assert "_run_one_warmup" not in source
    assert "_warmup.run(_warmup_key, _init_positions, 200)" in source

    # One shared published payload feeds every sampling chain.
    assert "_warmup_is_perchain = False" in source
    assert "_state_post_warmup = _batched_states" in source
    assert '_shared_step_size = _adapted_params["step_size"]' in source
    assert '_batched_step_size = _adapted_params["step_size"]' not in source


@pytest.mark.fast
def test_recipe_n_warmup_is_passed_explicitly_to_run() -> None:
    """Upstream derives num_steps from max_grad_budget when it is omitted.

    Letting that happen would make max_grad_budget silently override the
    recipe's n_warmup -- material behaviour supplied by an undocumented
    default, which the codegen contract forbids.
    """
    source = _emit(num_chains=6, n_warmup=137)
    assert "_warmup.run(_warmup_key, _init_positions, 137)" in source
    assert "if _warmup_nis.shape != (137, 6):" in source


@pytest.mark.fast
def test_joint_emission_guards_the_blackjax_capability() -> None:
    source = _emit(num_chains=6)
    assert '"n_chains" not in _sa_inspect.signature(' in source
    assert "raise RuntimeError(" in source


@pytest.mark.fast
def test_single_chain_emission_stays_portable_and_broadcasts() -> None:
    source = _emit(num_chains=6, warmup_num_chains=[1])
    ast.parse(source)
    # W=1 is the public default, so it neither passes n_chains as a call kwarg
    # nor needs the capability guard -- these recipes run on any blackjax with
    # metric="auto".  (The header comment still records n_chains=1.)
    assert "    n_chains=1," not in source
    assert "_sa_inspect" not in source
    assert "_warmup.run(_warmup_key, init_position, 200)" in source
    assert "if _warmup_nis.shape != (200,):" in source
    assert "_warmup_is_perchain = False" in source
    assert "jnp.broadcast_to(x[None], (num_chains,) + x.shape)" in source


@pytest.mark.fast
def test_gradient_accounting_is_summed_integration_steps() -> None:
    source = _emit(num_chains=6)
    assert "_warmup_grad_evals = int(jnp.sum(_warmup_nis))" in source
    assert "jointly adapted warmup chains" in source


@pytest.mark.fast
def test_telemetry_records_the_shared_scope_without_schema_change() -> None:
    source = _emit(num_chains=6)
    manifest_line = next(
        line
        for line in source.splitlines()
        if line.startswith("EXECUTION_MANIFEST_JSON")
    )
    manifest = json.loads(manifest_line.split("= ", 1)[1].strip().strip("'"))
    assert manifest["executable_config"]["warmup_name"] == WARMUP
    assert manifest["executable_config"]["warmup_params"]["max_grad_budget"] == 20_000
    # geometry_scope is emitted as a literal in the postamble.
    assert "_geometry_scope = 'shared'" in source
    # The existing low-rank marker carries the payload; no new marker type.
    assert "'type': 'low_rank_inverse_mass_matrix'" in source


@pytest.mark.fast
def test_missing_max_grad_budget_fails_at_generation_time() -> None:
    with pytest.raises(ValueError, match="missing 'max_grad_budget'"):
        _emit(num_chains=6, max_grad_budget=None)


@pytest.mark.fast
def test_non_integer_max_grad_budget_fails_at_generation_time() -> None:
    recipe = replace(
        _recipe(num_chains=6),
        warmup_params={"n_warmup": 200, "num_chains": 6, "max_grad_budget": 2.5},
    )
    with pytest.raises(ValueError, match="must be an int"):
        emit_script(recipe, num_samples=5)


# ---------------------------------------------------------------------------
# Generated-vs-public parity
# ---------------------------------------------------------------------------


@requires_joint_controller
@pytest.mark.e2e
def test_generated_program_payload_matches_the_direct_public_call(tmp_path) -> None:
    """Real parity: EXECUTE the generated program, compare to the public call.

    This is the semantic-fidelity gate, so it has to compare two things.  An
    earlier version of this test emitted the source, ran the public API
    separately, and then asserted properties of the PUBLIC result plus a couple
    of substrings of the emitted text -- which compares nothing and would have
    stayed green through any codegen regression that still produced parseable
    source.

    Here the generated program is run through the real launcher and its
    persisted telemetry is compared against a direct
    ``blackjax.staged_adaptation`` call under identical seeds and settings:
    published step size, published shared low-rank metric, and summed warmup
    gradient count.  Semantic fidelity only -- not a convergence claim.
    """
    import blackjax
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tuningfork.catalog import execute_recipe
    from tuningfork.model._numpyro import build_logdensity_fn

    num_chains, n_warmup, budget = 6, 200, 20_000
    recipe = _recipe(num_chains=num_chains, n_warmup=n_warmup, max_grad_budget=budget)

    # --- generated side: emit, launch, read what the program actually wrote ---
    result = execute_recipe(
        recipe, tmp_path / "runs", num_samples=5, progress_bar=False, timeout=900
    )
    assert result.returncode == 0
    _assert_child_blackjax_is_the_parents(result)
    assert result.telemetry_path is not None
    telemetry = json.loads(result.telemetry_path.read_text())
    generated = telemetry["geometry"]

    # --- public side: the same call, by hand, with the emitted choreography ---
    source = result.source_path.read_text()
    assert "jax.random.fold_in(jax.random.key(0), 0)" in source
    init_position, logdensity_fn, _ = build_logdensity_fn(
        jax.random.key(0), MODELS["mvn_10"]
    )
    warmup = blackjax.staged_adaptation(
        blackjax.nuts,
        logdensity_fn,
        metric="auto",
        max_grad_budget=budget,
        n_chains=num_chains,
        target_acceptance_rate=0.8,
    )
    positions = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape), init_position
    )
    (states, params), info = warmup.run(
        jax.random.fold_in(jax.random.key(0), 0), positions, n_warmup
    )

    # --- the actual comparison ---
    assert generated["step_size"] == float(params["step_size"]), (
        f"generated published step_size {generated['step_size']}, direct public "
        f"call published {float(params['step_size'])}"
    )

    imm = params["inverse_mass_matrix"]
    generated_imm = generated["inverse_mass_matrix"]
    assert generated_imm["type"] == "low_rank_inverse_mass_matrix"
    for field in ("sigma", "U", "lam"):
        np.testing.assert_array_equal(
            np.asarray(generated_imm[field]),
            np.asarray(getattr(imm, field)),
            err_msg=f"generated and public disagree on inverse_mass_matrix.{field}",
        )

    expected_grads = int(np.sum(np.asarray(info.info.num_integration_steps)))
    assert telemetry["warmup_grad_evals"] == expected_grads

    # Topology, read off the payload that was actually persisted: one shared
    # step size and one shared metric, not per-chain values.
    assert telemetry["geometry_scope"] == "shared"
    assert not isinstance(generated_imm["sigma"][0], list)
    leaves = jax.tree.leaves(states.position)
    assert leaves and all(np.shape(leaf)[0] == num_chains for leaf in leaves)


@requires_joint_controller
@pytest.mark.slow
def test_short_warmup_degenerates_the_controller_schedule() -> None:
    """One pinned cell of upstream behaviour.  No general minimum is claimed.

    Pins: model mvn_10, base method nuts, n_chains=6, max_grad_budget=20000,
    warmup key ``jax.random.fold_in(jax.random.key(0), 0)``, all chains
    broadcast from one prior_sample position.  Under exactly these settings a
    short n_warmup publishes a runaway step size and upstream raises no
    warning.  The test exists so that behaviour stays visible and attributed to
    upstream rather than to codegen; it deliberately does NOT define an
    n_warmup floor for other models, dimensions, chain counts or budgets, and
    tuningfork adds no controller-policy override for it.
    """
    import blackjax
    import jax
    import jax.numpy as jnp

    from tuningfork.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(
        jax.random.key(0), MODELS["mvn_10"]
    )

    def _step_size(n_warmup: int) -> float:
        warmup = blackjax.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=20_000,
            n_chains=6,
            target_acceptance_rate=0.8,
        )
        positions = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None], (6,) + x.shape), init_position
        )
        (_, params), _ = warmup.run(
            jax.random.fold_in(jax.random.key(0), 0), positions, n_warmup
        )
        return float(params["step_size"])

    assert _step_size(40) > 100.0, "short-warmup runaway no longer reproduces"
    assert 0.1 < _step_size(200) < 10.0


# ---------------------------------------------------------------------------
# Executed generated-program coverage
#
# These run the emitted standalone program in a subprocess via the real
# launcher.  A successful joint run proves the child had the n_chains
# CAPABILITY -- it does not by itself prove the child imported any particular
# build, so identity is asserted separately from the execution receipt's
# child-interpreter provenance.  See _assert_child_blackjax_is_the_parents.
# ---------------------------------------------------------------------------


def _telemetry_geometry(result) -> tuple[str, dict]:
    assert result.telemetry_path is not None
    payload = json.loads(result.telemetry_path.read_text())
    return payload["geometry_scope"], payload["geometry"]


def _assert_child_blackjax_is_the_parents(result) -> None:
    """Pin the child interpreter's blackjax to the one this process verified.

    The emitted capability guard proves the child could call the joint
    controller; it says nothing about WHICH build answered.  The launcher
    records the child's own resolution (``child_packages``, scoped
    ``child_interpreter``), so identity is measured rather than assumed.

    Asserting child == parent is deliberately environment-agnostic: whatever
    pins the parent also pins the child.  In the dedicated CI job the parent is
    pinned to an immutable SHA by .github/scripts/verify_joint_controller_env.py
    (module path under the pinned checkout AND checkout HEAD == that SHA), so
    parent-pinned plus child-equals-parent closes the chain to the SHA without
    the test needing to know anything about CI.
    """
    environment = result.receipt.environment
    assert environment["child_packages_scope"] == "child_interpreter"
    child = environment["child_packages"]
    assert "error" not in child, f"child provenance probe failed: {child}"

    child_blackjax = child["packages"]["blackjax"]
    parent_origin = Path(_blackjax.__file__).resolve()
    assert child_blackjax["origin"] is not None, child_blackjax
    assert Path(child_blackjax["origin"]).resolve() == parent_origin, (
        f"child resolved blackjax at {child_blackjax['origin']}, parent at "
        f"{parent_origin}: the generated program did not import the build this "
        "test verified"
    )
    assert child_blackjax["version"] == _blackjax.__version__


@requires_joint_controller
@pytest.mark.e2e
def test_small_joint_run_executes_and_samples_cleanly(tmp_path) -> None:
    """A joint run at an adequate n_warmup, executed end to end.

    Complements the pinned short-warmup runaway case: the controller is not
    merely reproducible, it produces a usable shared step size and a clean
    sample when the warmup is long enough for its schedule.
    """
    from tuningfork.catalog import execute_recipe

    result = execute_recipe(
        _recipe(num_chains=6, n_warmup=200, max_grad_budget=20_000),
        tmp_path / "runs",
        num_samples=20,
        progress_bar=False,
        timeout=600,
    )
    assert result.returncode == 0
    assert result.artifact_path is not None
    _assert_child_blackjax_is_the_parents(result)

    scope, geometry = _telemetry_geometry(result)
    # The joint controller publishes ONE payload for all six chains.
    assert scope == "shared"
    step_size = geometry["step_size"]
    assert isinstance(step_size, float)
    assert 0.05 < step_size < 20.0, f"joint controller published {step_size}"
    imm = geometry["inverse_mass_matrix"]
    assert imm["type"] == "low_rank_inverse_mass_matrix"
    # Shared scope forbids a batched marker; sigma must be a flat vector.
    assert not isinstance(imm["sigma"][0], list)
    assert len(imm["U"]) == len(imm["sigma"])
    assert len(imm["U"][0]) == len(imm["lam"])

    stdout = result.stdout_path.read_text()
    assert "n_divergences=0" in stdout, stdout[-500:]


@requires_joint_controller
@pytest.mark.e2e
def test_joint_run_with_per_chain_init_strategy_executes(tmp_path) -> None:
    """Dispersed per-chain starts, executed -- not merely admitted by the plan.

    The cross-chain gates in the controller are designed for dispersed starts,
    so this pairing is the intended one, and plan resolution admitting it is
    not evidence that it runs.

    Note on ``_ENSEMBLE_FRIENDLY_WARMUPS``: this warmup was added to that
    frozenset in the same change, but that is a CONSISTENCY fix, not a defect
    fix.  ``validate_init_strategy_warmup_compatibility`` has no production
    caller -- emission and launching never consult it -- so per-chain init
    emitted correctly with or without the entry.  Verified, not assumed.
    """
    from tuningfork.catalog import execute_recipe

    recipe = replace(
        _recipe(num_chains=6, n_warmup=200, max_grad_budget=20_000),
        init_strategy={"type": "uniform_perchain", "low": -2.0, "high": 2.0},
    )
    result = execute_recipe(
        recipe, tmp_path / "runs", num_samples=20, progress_bar=False, timeout=600
    )
    assert result.returncode == 0
    _assert_child_blackjax_is_the_parents(result)
    assert result.manifest.executable_config["init_strategy"] == {
        "type": "uniform_perchain",
        "low": -2.0,
        "high": 2.0,
    }
    source = result.source_path.read_text()
    # Pre-batched positions feed the controller directly; no broadcast of one
    # shared start, which would defeat the dispersion this strategy exists for.
    assert "_init_positions = init_position" in source
    assert "_init_position_is_prebatched = True" in source

    scope, geometry = _telemetry_geometry(result)
    assert scope == "shared"
    assert 0.05 < geometry["step_size"] < 20.0
    assert "n_divergences=0" in result.stdout_path.read_text()


@pytest.mark.e2e
def test_w1_generated_program_runs_on_any_supported_blackjax(tmp_path) -> None:
    """W=1 portability: no n_chains kwarg, so no capability requirement.

    This is deliberately NOT capability-gated -- it is the released-dependency
    half of the coverage and must execute against the ordinary bench sync.
    """
    from tuningfork.catalog import execute_recipe

    result = execute_recipe(
        _recipe(num_chains=4, n_warmup=200, max_grad_budget=20_000),
        tmp_path / "runs",
        num_samples=20,
        warmup_num_chains=[1],
        progress_bar=False,
        timeout=600,
    )
    assert result.returncode == 0
    _assert_child_blackjax_is_the_parents(result)
    source = result.source_path.read_text()
    assert "    n_chains=1," not in source
    assert "_sa_inspect" not in source

    scope, geometry = _telemetry_geometry(result)
    assert scope == "shared"
    assert 0.05 < geometry["step_size"] < 20.0
    assert geometry["inverse_mass_matrix"]["type"] == "low_rank_inverse_mass_matrix"


@requires_released_without_n_chains
@pytest.mark.e2e
def test_joint_program_fails_explicitly_without_the_capability(tmp_path) -> None:
    """On a released blackjax the joint program must name the real reason.

    Without the emitted guard, n_chains falls through ``**extra_parameters``
    into the sampling kernel and surfaces as an unrelated TypeError after
    tracing.
    """
    from tuningfork.catalog import execute_recipe
    from tuningfork.recipes._launcher import GeneratedProgramError

    with pytest.raises(GeneratedProgramError) as excinfo:
        execute_recipe(
            _recipe(num_chains=6, n_warmup=200, max_grad_budget=20_000),
            tmp_path / "runs",
            num_samples=5,
            progress_bar=False,
            timeout=600,
        )
    stderr = excinfo.value.result.stderr_path.read_text()
    assert "staged_adaptation_auto with warmup_num_chains>1 requires a blackjax" in (
        stderr
    )
    assert "accepts n_chains" in stderr
    assert "TypeError" not in stderr


@pytest.mark.fast
def test_child_identity_assertion_detects_a_divergent_child() -> None:
    """The identity check must be able to fail, not just pass everywhere.

    Guards against the mechanism silently degrading into a tautology if the
    receipt shape changes -- a missing key, a failed probe, or a child that
    resolved a different build must all be caught rather than skipped over.
    """
    from types import SimpleNamespace

    def _result(child: dict, scope: str = "child_interpreter"):
        return SimpleNamespace(
            receipt=SimpleNamespace(
                environment={"child_packages_scope": scope, "child_packages": child}
            )
        )

    matching = {
        "packages": {
            "blackjax": {
                "origin": _blackjax.__file__,
                "version": _blackjax.__version__,
            }
        }
    }
    # Sanity: the honest case passes, so the failures below are meaningful.
    _assert_child_blackjax_is_the_parents(_result(matching))

    # A child that resolved some other build.
    with pytest.raises(AssertionError, match="did not import the build"):
        _assert_child_blackjax_is_the_parents(
            _result(
                {
                    "packages": {
                        "blackjax": {
                            "origin": "/somewhere/else/blackjax/__init__.py",
                            "version": _blackjax.__version__,
                        }
                    }
                }
            )
        )
    # A probe that failed must not pass as "no mismatch observed".
    with pytest.raises(AssertionError, match="child provenance probe failed"):
        _assert_child_blackjax_is_the_parents(_result({"error": "probe exited 1"}))
    # An unresolvable module must not pass either.
    with pytest.raises(AssertionError):
        _assert_child_blackjax_is_the_parents(
            _result({"packages": {"blackjax": {"origin": None, "version": None}}})
        )
    # Wrong scope label means the provenance is not the child's.
    with pytest.raises(AssertionError):
        _assert_child_blackjax_is_the_parents(
            _result(matching, scope="launcher_process")
        )
