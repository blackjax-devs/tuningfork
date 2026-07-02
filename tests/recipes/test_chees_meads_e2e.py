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
"""End-to-end wiring regression tests for the ChEES-HMC and MEADS warmups
through the real ``_recipe_runner.py`` pipeline.

Both warmups have full wrappers + registry entries but were never exercised
end-to-end through ``emit_low_recipe_for_cell`` — the generic dispatch was
built and validated only against the ``window_adaptation`` output contract
(step_size + inverse_mass_matrix, per-chain broadcast, no callables). These
tests pin the wiring defects a statistician A/B investigation located so
they cannot silently regress. The tests assert an HONEST gate verdict is
reached (PASS, REVIEW, or FAIL are all acceptable outcomes) — the point is
that the pipeline runs to completion without TypeError/NaN-crash, not that
the cell PASSES the statistician gate (a later statistician pass tunes for
that).
"""

from pathlib import Path

import pytest

# Not a blanket module-level pytestmark: the emit_low_recipe_for_cell tests
# are phase-level gates (warmup + sampling + auto-gate, >10s) -> e2e; the
# _reduce_and_broadcast_warmup_output unit test is pure array math (<100ms)
# -> fast; the _build_shared_kwargs/_reinit_batched_state unit tests are
# single JAX-compiled warmup calls (<5s) -> slow. Marked individually below.


@pytest.mark.e2e
def test_chees_dynamic_hmc_e2e_no_crash(tmp_path: Path) -> None:
    """emit_low_recipe_for_cell(mvn_10, chees, dynamic_hmc) runs to completion.

    Regression guard for bug 2 (target_acceptance_rate=None TypeError in
    upstream chees_adaptation.py: `target_acceptance_rate - harmonic_mean`)
    and bug 3 (np.isfinite TypeError on the Python-callable leaves CHEES
    legitimately returns in adapted_params: next_random_arg_fn,
    integration_steps_fn). Both bugs fire deterministically on every CHEES
    cell through the default (no-override) call path — this is the exact
    call shape used in production recipe emission.
    """
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "chees",
        "dynamic_hmc",
        n_warmup=50,
        n_samples=50,
        num_chains=4,
        catalog_root=tmp_path,
        verbose=False,
    )
    assert result.verdict in ("PASS", "REVIEW", "FAIL"), (
        f"Expected an honest gate verdict, got {result.verdict!r} "
        f"(note={result.note!r})"
    )
    # FAIL from a genuine gate reason (rhat/ess/div/NaN-in-sampler) is fine at
    # this tiny n_warmup/n_samples; a crash-derived FAIL (TypeError/KeyError
    # text in the note) is what we're guarding against.
    if result.verdict == "FAIL":
        assert "Error" not in (
            result.note or ""
        ), f"FAIL note looks crash-derived, not gate-derived: {result.note}"


@pytest.mark.e2e
def test_meads_ghmc_e2e_no_crash(tmp_path: Path) -> None:
    """emit_low_recipe_for_cell(mvn_10, meads, ghmc) runs to completion.

    Regression guard for bug 1 (MEADS identical-init NaN, see
    warmup/meads.py's _replicate_with_init_jitter) and for the GHMC
    momentum_inverse_scale/inverse_mass_matrix kwarg-name mismatch:
    blackjax.ghmc.as_top_level_api has no inverse_mass_matrix parameter at
    all (no **kwargs catch-all either) -- the generic dispatch
    (_build_vmapped_inference) unconditionally called
    base_method.factory(..., inverse_mass_matrix=imm, ...) for every
    sampler, which TypeErrors immediately for ghmc. MEADS's adapted_params
    also uses the key "momentum_inverse_scale", not "inverse_mass_matrix",
    so batched_params.get("inverse_mass_matrix") returned None before the
    kwarg-name TypeError was even reached. Both defects are independent of
    the 5 statistician-diagnosed items and block MEADS+GHMC unconditionally
    (any warmup paired with ghmc would hit the kwarg-name mismatch).

    Uses num_chains=16 (not the recipe-runner default of 4): MEADS requires
    num_chains // num_folds >= 2 to have any within-fold cross-chain
    dispersion to estimate (see warmup/meads.py ENTRY.notes) -- an honest
    reflection of the algorithm's structural requirement, not gate-gaming.
    """
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "meads",
        "ghmc",
        n_warmup=50,
        n_samples=50,
        num_chains=16,
        catalog_root=tmp_path,
        verbose=False,
    )
    assert result.verdict in ("PASS", "REVIEW", "FAIL"), (
        f"Expected an honest gate verdict, got {result.verdict!r} "
        f"(note={result.note!r})"
    )
    if result.verdict == "FAIL":
        assert "NaN" not in (result.note or "") and "Error" not in (
            result.note or ""
        ), f"FAIL note looks crash/NaN-derived, not gate-derived: {result.note}"


@pytest.mark.e2e
def test_meads_recipe_pins_inverse_mass_matrix(tmp_path: Path) -> None:
    """A PASS/REVIEW MEADS recipe must pin a real (non-null) inverse_mass_matrix.

    Regression guard for the momentum_inverse_scale -> inverse_mass_matrix
    alias: the recipe-schema emission code
    (imm_raw = batched_params.get("inverse_mass_matrix", None)) reads the
    canonical key. Before the alias, this returned None for MEADS -- any
    emitted MEADS recipe would have silently pinned a NULL mass matrix.
    Runs at a generous n_warmup so the cell has a realistic chance to reach
    PASS/REVIEW (where a recipe is actually emitted); skips the pin
    assertion on FAIL since no recipe is written in that case.
    """
    import json

    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "meads",
        "ghmc",
        n_warmup=1000,
        n_samples=1000,
        num_chains=64,
        catalog_root=tmp_path,
        verbose=False,
    )
    if result.verdict == "FAIL":
        pytest.skip(
            f"Gate FAIL at this seed/config ({result.note}); no recipe emitted "
            "to check -- tuning for PASS is a later statistician pass, not "
            "this wiring test's job."
        )
    recipe_path = tmp_path / "mvn_10" / "recipes" / "low__ghmc__meads.json"
    assert recipe_path.exists(), f"Expected recipe at {recipe_path}"
    recipe = json.loads(recipe_path.read_text())
    imm = recipe["base_method_params"].get("inverse_mass_matrix")
    assert imm is not None, (
        "MEADS recipe pinned a null inverse_mass_matrix -- the "
        "momentum_inverse_scale alias regressed."
    )


@pytest.mark.fast
def test_reduce_and_broadcast_key_name_aware_for_meads() -> None:
    """HARD-KEEP regression guard for bug 5: reduce-broadcast is key-name-aware.

    _reduce_and_broadcast_warmup_output (used by run_recipe_to_idata's
    warmup_num_chains adapt-many/sample-few path) hardcoded
    warmup_params["inverse_mass_matrix"] -- a KeyError on MEADS's
    "momentum_inverse_scale" key. Fixed via an explicit imm_kwarg_name
    parameter (callers pass base_method.imm_kwarg_name -- the single source
    of truth on the BaseMethod descriptor, ghmc: "momentum_inverse_scale",
    everything else: the default "inverse_mass_matrix") rather than sniffing
    which key is present; reduce+broadcast happens under that key name.
    """
    import jax.numpy as jnp

    from tuningfork.recipes._recipe_runner import _reduce_and_broadcast_warmup_output

    # MEADS-shaped warmup_params: momentum_inverse_scale, not inverse_mass_matrix.
    num_warmup_chains, num_sampling_chains, d = 16, 4, 5
    warmup_state = {"position": jnp.zeros((num_warmup_chains, d))}
    warmup_params = {
        "step_size": jnp.linspace(0.1, 0.2, num_warmup_chains),
        "momentum_inverse_scale": jnp.ones((num_warmup_chains, d)) * 2.0,
        "alpha": jnp.full((num_warmup_chains,), 0.5),
        "delta": jnp.full((num_warmup_chains,), 0.25),
    }
    broadcasted_state, broadcasted_params = _reduce_and_broadcast_warmup_output(
        warmup_state,
        warmup_params,
        num_warmup_chains,
        num_sampling_chains,
        imm_kwarg_name="momentum_inverse_scale",
    )
    assert "momentum_inverse_scale" in broadcasted_params, (
        "Expected the reduce step to preserve the momentum_inverse_scale key "
        f"name; got keys: {list(broadcasted_params)}"
    )
    assert "inverse_mass_matrix" not in broadcasted_params, (
        "reduce_and_broadcast should not invent an inverse_mass_matrix key "
        "when told imm_kwarg_name=momentum_inverse_scale."
    )
    reduced_imm = jnp.asarray(broadcasted_params["momentum_inverse_scale"])
    assert reduced_imm.shape == (num_sampling_chains, d)
    assert bool(jnp.allclose(reduced_imm, 2.0))

    # Sibling check: standard inverse_mass_matrix-keyed input still works via
    # the default imm_kwarg_name (no regression for window_adaptation-style
    # warmups, which don't pass imm_kwarg_name explicitly).
    warmup_params_std = {
        "step_size": jnp.linspace(0.1, 0.2, num_warmup_chains),
        "inverse_mass_matrix": jnp.ones((num_warmup_chains, d)) * 3.0,
    }
    _, broadcasted_std = _reduce_and_broadcast_warmup_output(
        warmup_state, warmup_params_std, num_warmup_chains, num_sampling_chains
    )
    assert "inverse_mass_matrix" in broadcasted_std
    assert bool(jnp.allclose(jnp.asarray(broadcasted_std["inverse_mass_matrix"]), 3.0))


@pytest.mark.slow
def test_chees_own_trajectory_length_threaded_into_shared_kwargs() -> None:
    """HARD-KEEP regression guard for bug 4 (item 4, the design-not-mechanical one).

    Before the fix, _build_shared_kwargs unconditionally set
    integration_steps_fn = build_step_policy(_effective_step_policy); for
    chees, _effective_step_policy is None -> build_step_policy(None) returns
    the V0 library default (`lambda key: randint(1, 10)`). CHEES's OWN
    adapted integration_steps_fn (the entire point of ChEES-HMC) was
    computed, pinned in adapted_params, and never read -- per_chain_param_keys
    omits it, and shared_kwargs overwrote it unconditionally. This test
    proves the fix by identity: shared_kwargs["integration_steps_fn"] must
    literally BE the callable CHEES returned, not a fresh V0-default closure.
    Gated on the explicit warmup_name="chees" kwarg (not batched_params
    sniffing) per TL review -- explicit identity preferred when reachable.
    """
    import jax

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.recipes._recipe_runner import _build_shared_kwargs
    from tuningfork.warmup.chees import ENTRY as CHEES_ENTRY

    key = jax.random.key(555)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, MODELS["mvn_10"])
    _, adapted_params = CHEES_ENTRY.runner(
        jax.random.fold_in(key, 1),
        init_pos,
        100,
        BASE_METHODS["dynamic_hmc"],
        logdensity_fn=logdensity_fn,
        num_chains=4,
    )
    shared_kwargs, _ = _build_shared_kwargs(
        BASE_METHODS["dynamic_hmc"],
        "dynamic_hmc",
        adapted_params,
        None,  # batched_warmup_info
        None,  # warmup_inner_kernel
        None,  # step_policy
        None,  # params_override
        warmup_name="chees",
    )
    assert (
        shared_kwargs["integration_steps_fn"] is adapted_params["integration_steps_fn"]
    ), (
        "shared_kwargs['integration_steps_fn'] is not CHEES's own adapted "
        "callable -- the V0 build_step_policy(None) default leaked back in."
    )
    assert (
        shared_kwargs["next_random_arg_fn"] is adapted_params["next_random_arg_fn"]
    ), "next_random_arg_fn was not threaded through alongside integration_steps_fn."
    assert (
        shared_kwargs["integration_steps_params"]
        == adapted_params["integration_steps_params"]
    ), "integration_steps_params was not threaded through."


@pytest.mark.slow
def test_chees_reinit_preserves_random_generator_arg_counter() -> None:
    """HARD-KEEP: dynamic_hmc reinit must be SKIPPED for chees, not applied.

    dynamic_hmc.reinit_state=True exists because most dynamic_hmc-pairing
    warmups produce a plain HMCState lacking random_generator_arg -- reinit
    via kernel.init(position, reinit_key) is required to add it, and that
    call sets random_generator_arg = reinit_key (a raw PRNGKey, per
    blackjax.dynamic_hmc.init's pass_rng_key_to_init=True convention).
    CHEES is the exception: its own AdaptationResults.state is ALREADY a
    correctly-shaped DynamicHMCState with random_generator_arg as an
    INTEGER counter (CHEES inits it at 0, increments by 1 per adaptation
    step) -- CHEES's adapted integration_steps_fn calls
    dynamic_hmc.halton_sequence(random_generator_arg, max_bits) internally,
    which raises ValueError("Invalid integer data type 'O'") on a PRNGKey.
    Reinit-by-default would silently clobber the correct counter with the
    wrong type. This test proves the fix: the reinit-skip path (gated on the
    explicit warmup_name="chees" kwarg, not batched_params sniffing, per TL
    review) preserves CHEES's own counter bit-for-bit rather than replacing
    it with reinit_key.
    """
    import jax

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.recipes._recipe_runner import (
        _build_shared_kwargs,
        _reinit_batched_state,
    )
    from tuningfork.warmup.chees import ENTRY as CHEES_ENTRY

    key = jax.random.key(9)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, MODELS["mvn_10"])
    base_method = BASE_METHODS["dynamic_hmc"]
    states, adapted_params = CHEES_ENTRY.runner(
        jax.random.fold_in(key, 1),
        init_pos,
        50,
        base_method,
        logdensity_fn=logdensity_fn,
        num_chains=4,
    )
    # Sanity: CHEES's own counter is an int32 array (== n_warmup after 50
    # +1-increments from init_random_arg=0), NOT a PRNGKey.
    import jax.numpy as jnp

    assert jnp.issubdtype(states.random_generator_arg.dtype, jnp.integer), (
        f"Expected CHEES's random_generator_arg to be an integer counter, "
        f"got dtype {states.random_generator_arg.dtype}"
    )

    shared_kwargs, _ = _build_shared_kwargs(
        base_method,
        "dynamic_hmc",
        adapted_params,
        None,
        None,
        None,
        None,
        warmup_name="chees",
    )
    reinit_keys = jax.random.split(jax.random.key(1234), 4)
    run_states = _reinit_batched_state(
        states,
        adapted_params["step_size"],
        adapted_params["inverse_mass_matrix"],
        None,
        reinit_keys,
        logdensity_fn=logdensity_fn,
        base_method=base_method,
        shared_kwargs=shared_kwargs,
        laplace_log_joint_fn=None,
        laplace_theta_init=None,
        warmup_name="chees",
    )
    assert bool(
        (run_states.random_generator_arg == states.random_generator_arg).all()
    ), (
        "_reinit_batched_state overwrote CHEES's own random_generator_arg "
        "counter -- reinit was NOT skipped for the chees+dynamic_hmc case."
    )
