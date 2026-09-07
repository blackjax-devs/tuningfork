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
import sys
from dataclasses import replace

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
_BLACKJAX_FILE = ""
_BLACKJAX_ORIGIN = "<import failed>"
try:  # pragma: no cover - import guard, not a branch under test
    import blackjax as _blackjax

    _HAS_N_CHAINS = (
        "n_chains" in inspect.signature(_blackjax.staged_adaptation).parameters
    )
    _BLACKJAX_FILE = _blackjax.__file__
    _BLACKJAX_ORIGIN = f"{_blackjax.__version__} from {_BLACKJAX_FILE}"
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

# Tie THIS process's blackjax to the pin, in this process.  The CI verify step
# pins its own interpreter, but that is a separate process; without this, the
# in-test assertion that the child ran under this interpreter would chain to a
# parent nothing here had pinned.  BLACKJAX_PINNED_PATH is a job-level env var,
# so it is already visible to the pytest step; outside that job it is unset and
# this check is inert.
_PINNED_PATH = os.environ.get("BLACKJAX_PINNED_PATH")
if _REQUIRE_JOINT and _PINNED_PATH:  # pragma: no cover - CI gate path
    import pathlib as _pathlib

    _pinned_root = _pathlib.Path(_PINNED_PATH).resolve()
    _origin = _pathlib.Path(_BLACKJAX_FILE).resolve()
    if _pinned_root not in _origin.parents:
        raise RuntimeError(
            f"this process imported blackjax from {_origin}, which is not under "
            f"the pinned checkout {_pinned_root}"
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
@pytest.mark.parametrize("n_warmup", [137, 200])
def test_joint_emission_is_one_shared_controller_not_a_vmap(n_warmup: int) -> None:
    """Topology and material arguments only.

    The distinction this must hold is structural: ONE controller call carrying
    the joint chain count, versus the window family's vmap over independent
    warmups.  n_warmup is parametrised because it is material -- upstream
    derives num_steps from max_grad_budget when it is omitted, which would let
    max_grad_budget silently override the recipe, so a hardcoded constant must
    fail here.
    """
    source = _emit(num_chains=6, n_warmup=n_warmup)
    ast.parse(source)

    # One controller construction, carrying the material arguments.
    assert source.count("blackjax.staged_adaptation(") == 1
    assert '    metric="auto",' in source
    assert "    n_chains=6," in source
    assert "    max_grad_budget=20000," in source
    assert f"_warmup.run(_warmup_key, _init_positions, {n_warmup})" in source

    # Not a vmap over independent per-chain warmups.
    assert "_run_one_warmup" not in source

    # One shared published payload feeds every sampling chain.
    assert "_warmup_is_perchain = False" in source
    assert '_shared_step_size = _adapted_params["step_size"]' in source
    assert '_batched_step_size = _adapted_params["step_size"]' not in source

    # Gradient accounting is the joint one.  That it is emitted exactly once is
    # asserted for every exact route by
    # test_generated_warmup_accounting.test_exact_routes_emit_accounting_once.
    assert "jointly adapted warmup chains" in source

    # W>1 cannot silently fall through to a blackjax without n_chains.
    assert '"n_chains" not in _sa_inspect.signature(' in source


@pytest.mark.fast
def test_single_chain_emission_stays_portable_and_broadcasts() -> None:
    """W=1 passes no n_chains, so it needs no capability and no guard."""
    source = _emit(num_chains=6, warmup_num_chains=[1])
    ast.parse(source)
    assert "    n_chains=1," not in source
    assert "_sa_inspect" not in source
    assert "_warmup.run(_warmup_key, init_position, 200)" in source
    assert "_warmup_is_perchain = False" in source
    assert "jnp.broadcast_to(x[None], (num_chains,) + x.shape)" in source


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
    """Compare generated warmup payloads with the direct public call.

    This measures only warmup-payload fidelity (step size, shared low-rank
    metric, and gradient count). Sampling metric consumption is checked by the
    emission assertion, not this runtime measurement; this is not sampling
    parity or convergence evidence. Equality is empirical, not guaranteed, for
    one model and seed with identical CPU choreography; classify any failure
    before changing it, since genuine divergence differs from a last-bit
    execution difference.
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
    _assert_child_ran_under_this_interpreter(result)
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
    assert len(generated_imm["U"]) == len(generated_imm["sigma"])
    assert len(generated_imm["U"][0]) == len(generated_imm["lam"])
    leaves = jax.tree.leaves(states.position)
    assert leaves and all(np.shape(leaf)[0] == num_chains for leaf in leaves)

    # This configuration is a SUCCESSFUL warmup, not merely a reproducible one:
    # a usable shared step size and a clean sample. Parity against a runaway
    # would prove fidelity while saying nothing about the route working.
    assert 0.05 < generated["step_size"] < 20.0
    assert "n_divergences=0" in result.stdout_path.read_text()


# ---------------------------------------------------------------------------
# Executed generated-program coverage
#
# These run the emitted standalone program in a subprocess via the real
# launcher.  A successful joint run proves the child had the n_chains
# CAPABILITY -- it does not by itself prove the child imported any particular
# build, so identity is asserted separately from the execution receipt's
# child-interpreter provenance.  See _assert_child_ran_under_this_interpreter.
# ---------------------------------------------------------------------------


def _telemetry_geometry(result) -> tuple[str, dict]:
    assert result.telemetry_path is not None
    payload = json.loads(result.telemetry_path.read_text())
    return payload["geometry_scope"], payload["geometry"]


def _assert_child_ran_under_this_interpreter(result) -> None:
    """Tie the generated program's interpreter to this test process's.

    What this proves: the launcher invoked the same Python executable this
    process is running, so the generated program resolved imports from the same
    environment whose blackjax the caller has verified.  It reads only fields
    the execution receipt already records.

    What it does NOT prove: that the child's import could not be shadowed by a
    ``sys.path[0]`` difference -- the child runs with its work directory as
    cwd, this process with the repo root.  The dedicated CI job closes that
    separately, by importing blackjax under a child-like invocation (same
    interpreter, foreign cwd) and asserting the resolved file.
    """
    environment = result.receipt.environment
    assert environment["child_python_executable"] == sys.executable, (
        f"generated program ran under {environment['child_python_executable']}, "
        f"this process under {sys.executable}"
    )
    assert environment["launcher_python"]["executable"] == sys.executable


@requires_joint_controller
@pytest.mark.e2e
def test_joint_run_with_per_chain_init_strategy_executes(tmp_path) -> None:
    """Dispersed per-chain starts, executed -- not merely admitted by the plan.

    The cross-chain gates in the controller are designed for dispersed starts,
    so this pairing is the intended one, and plan resolution admitting it is
    not evidence that it runs.

    Note: ``validate_init_strategy_warmup_compatibility`` has no production
    caller -- emission and launching never consult it -- so this warmup's entry
    in ``_ENSEMBLE_FRIENDLY_WARMUPS`` keeps that declared list consistent and
    does not gate this path.
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
    _assert_child_ran_under_this_interpreter(result)
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
    _assert_child_ran_under_this_interpreter(result)
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
def test_dual_topology_warmups_are_declared_ensemble_friendly() -> None:
    """The two lists must not drift apart.

    A warmup that supports W=S consumes one initial position per warmup chain,
    which is exactly what the per-chain init strategies produce.  This is the
    direction that holds: every dual-topology warmup should be declared
    ensemble-friendly, not the converse -- ``_ENSEMBLE_FRIENDLY_WARMUPS`` is
    the broader list and is currently broader than plan resolution allows.
    """
    from tuningfork.recipes._init_strategy import _ENSEMBLE_FRIENDLY_WARMUPS
    from tuningfork.recipes._resolve_execution_plan import _DUAL_TOPOLOGY_WARMUPS

    assert _DUAL_TOPOLOGY_WARMUPS <= _ENSEMBLE_FRIENDLY_WARMUPS
