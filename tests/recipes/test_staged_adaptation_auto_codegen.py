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
try:  # pragma: no cover - import guard, not a branch under test
    import blackjax as _blackjax

    _HAS_N_CHAINS = (
        "n_chains" in inspect.signature(_blackjax.staged_adaptation).parameters
    )
except Exception:  # pragma: no cover - blackjax always present in this suite
    _HAS_N_CHAINS = False

requires_joint_controller = pytest.mark.skipif(
    not _HAS_N_CHAINS,
    reason=(
        "installed blackjax.staged_adaptation does not accept n_chains "
        "(multi-chain metric='auto' controller)"
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
@pytest.mark.slow
def test_generated_joint_warmup_matches_the_direct_public_call() -> None:
    """The generated program must publish exactly what the public call does.

    This is the semantic-fidelity gate for the joint topology: same seed
    choreography, same shared step size, same shared low-rank metric, same
    summed warmup gradient count.  It is not a convergence claim.
    """
    import blackjax
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tuningfork.model._numpyro import build_logdensity_fn

    num_chains, n_warmup, budget = 6, 200, 20_000
    recipe = _recipe(num_chains=num_chains, n_warmup=n_warmup, max_grad_budget=budget)
    source = emit_script(recipe, num_samples=2)

    # Reproduce the emitted seed choreography with the unchanged public API.
    init_position, logdensity_fn, _ = build_logdensity_fn(
        jax.random.key(0), MODELS["mvn_10"]
    )
    assert "jax.random.fold_in(jax.random.key(0), 0)" in source
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

    # Topology: one shared step size, one shared low-rank metric, one warmup
    # state per chain.
    assert np.shape(params["step_size"]) == ()
    imm = params["inverse_mass_matrix"]
    assert getattr(imm, "_fields", None) == ("sigma", "U", "lam")
    # position may be a pytree; every leaf carries the joint chain axis, and
    # the shared metric is sized by the flattened dimension, not by chains.
    leaves = jax.tree.leaves(states.position)
    assert leaves and all(np.shape(leaf)[0] == num_chains for leaf in leaves)
    flat_dim = sum(int(np.size(leaf)) // num_chains for leaf in leaves)
    assert np.shape(imm.sigma) == (flat_dim,)
    assert np.shape(imm.U)[0] == flat_dim
    assert np.shape(imm.U)[1] == np.shape(imm.lam)[0]

    # Gradient accounting matches what the emitted program computes.
    nis = np.asarray(info.info.num_integration_steps)
    assert nis.shape == (n_warmup, num_chains)
    assert f"if _warmup_nis.shape != ({n_warmup}, {num_chains}):" in source


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
