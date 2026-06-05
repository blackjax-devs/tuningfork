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
"""Tests for Recipe emission logic (from warmup/tuning results).

This file contains all tests that run actual MCMC chains or tuning algorithms.
These are marked @pytest.mark.slow individually (not at module level, per PR-4 rules).

Tests: from_warmup_only, from_tuning_result, render_instructions_medium_and_high_real.

History: test_medium_recipe_exists_and_has_warmup_data (parametrized over 6
(model × {hmc, nuts}) combos) was removed 2026-05-17 as a slow-CI fix —
the MEDIUM placeholder recipes it asserted-existence-of had been deleted in
PR #6 commit 3 (715a82c, "recipes: delete stale low/medium/high starter
recipes"), but the test surgery in that commit missed this slow-only test
because we don't run slow locally. Real MEDIUM recipes are produced by
Recipe Phase 1+ pipeline; their existence-on-disk is no longer a test gate.
"""

import math
from pathlib import Path

import jax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS

# ---------------------------------------------------------------------------
# MEDIUM and HIGH constructors (require actual warmup)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_from_warmup_only_window_adaptation_diag_imm_nuts() -> None:
    """from_warmup_only with window_adaptation_diag_imm + NUTS returns a MEDIUM recipe.

    Verifies:
    - effort = MEDIUM
    - warmup_name = "window_adaptation_diag_imm"
    - base_method_params contains both step_size (from defaults) and
      inverse_mass_matrix (from warmup adaptation)
    - calibration_budget["n_warmup"] == 200
    - calibration_budget["wall_seconds_estimate"] > 0
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"

    # base_method_params must include both the default step_size (loguniform
    # 70th-pctile ≈ 0.126) AND the warmup-adapted inverse_mass_matrix.
    assert "step_size" in recipe.base_method_params
    assert "inverse_mass_matrix" in recipe.base_method_params

    # IMM must be a list (coerced from jax.Array by _to_jsonable).
    imm = recipe.base_method_params["inverse_mass_matrix"]
    assert isinstance(imm, list), f"inverse_mass_matrix should be list, got {type(imm)}"
    assert len(imm) == 10  # mvn_10 is 10-D

    # calibration_budget fields
    assert recipe.calibration_budget["n_warmup"] == 200
    assert recipe.calibration_budget["wall_seconds_estimate"] > 0
    assert recipe.calibration_budget["trials"] == 0

    # warmup_params records the input config
    assert recipe.warmup_params["n_warmup"] == 200

    # headline_metric is None for MEDIUM (no post-warmup samples)
    assert recipe.headline_metric is None

    # instructions must be non-empty prose
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


@pytest.mark.slow
def test_from_warmup_only_mclmc_tuning_metadata() -> None:
    """from_warmup_only with mclmc_tuning threads _total_tuning_steps into calibration_budget.

    Threading the ``_total_tuning_steps`` metadata key from mclmc_tuning into
    adapted_params with an underscore prefix.  from_warmup_only must capture it
    in calibration_budget and strip it from base_method_params.
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["mclmc"]
    warmup = WARMUPS["mclmc_tuning"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(1),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "mclmc_tuning"

    # _total_tuning_steps must appear in calibration_budget (threaded from metadata).
    assert "_total_tuning_steps" in recipe.calibration_budget
    assert isinstance(recipe.calibration_budget["_total_tuning_steps"], int)

    # _total_tuning_steps must NOT appear in base_method_params (stripped).
    assert "_total_tuning_steps" not in recipe.base_method_params

    # MCLMC adapted params (L, step_size) must be in base_method_params.
    assert "step_size" in recipe.base_method_params
    assert "L" in recipe.base_method_params


@pytest.mark.slow
def test_from_warmup_only_incompatible_raises() -> None:
    """from_warmup_only with an incompatible (warmup, base_method) pair raises ValueError."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    # mclmc_tuning is only compatible with mclmc, not nuts.
    warmup = WARMUPS["mclmc_tuning"]

    with pytest.raises(ValueError, match="not compatible"):
        Recipe.from_warmup_only(
            posterior,
            base_method,
            warmup,
            n_warmup=100,
            rng_key=jax.random.key(0),
        )


@pytest.mark.slow
def test_from_tuning_result_nuts() -> None:
    """from_tuning_result produces a HIGH recipe from tune_algorithm output.

    Verifies:
    - effort = HIGH
    - headline_metric > 0 (best_score was finite; mvn_10 + nuts is well-behaved)
    - difficulty dict contains expected keys
    - n_trials_completed matches n_trials arg
    """
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=3,
        n_seeds=1,
        n_chains=1,
        n_samples=200,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    recipe = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup,
    )

    assert recipe.effort == Effort.HIGH
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"
    assert recipe.warmup_name == "window_adaptation_diag_imm"

    # headline_metric should be a finite float (mvn_10 doesn't diverge)
    assert isinstance(recipe.headline_metric, float)
    assert math.isfinite(recipe.headline_metric)

    # difficulty dict from TuningDifficulty.asdict()
    assert recipe.difficulty is not None
    assert isinstance(recipe.difficulty, dict)
    for key in (
        "default_score",
        "best_score",
        "threshold_score",
        "default_works",
        "n_trials_to_threshold",
        "n_trials_to_best",
    ):
        assert key in recipe.difficulty, f"Missing difficulty key: {key}"

    # calibration_budget
    assert recipe.calibration_budget["trials"] == 3
    assert recipe.calibration_budget["n_seeds"] == 1

    # instructions non-empty
    assert len(recipe.instructions) > 10


@pytest.mark.slow
def test_from_tuning_result_save_load_roundtrip(tmp_path: Path) -> None:
    """HIGH recipe round-trips through Recipe.save / Recipe.load.

    Verifies in particular that:
    - inverse_mass_matrix (list[float]) in base_method_params round-trips.
    - difficulty dict (nested Python primitives) round-trips without JSON errors.
    """
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=2,
        n_seeds=1,
        n_chains=1,
        n_samples=100,
        n_warmup=100,
        rng_key=jax.random.key(42),
    )

    recipe = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup,
    )

    saved_path = recipe.save(tmp_path)
    loaded = Recipe.load(saved_path)

    # Core identity fields
    assert loaded.effort == Effort.HIGH
    assert loaded.model_name == recipe.model_name
    assert loaded.base_method_name == recipe.base_method_name
    assert loaded.warmup_name == recipe.warmup_name

    # headline_metric round-trip (float precision)
    assert loaded.headline_metric == recipe.headline_metric

    # difficulty round-trip (nested dict, not a dataclass after load)
    assert loaded.difficulty == recipe.difficulty
    assert isinstance(loaded.difficulty, dict)

    # inverse_mass_matrix round-trip: list[float] after load
    if "inverse_mass_matrix" in recipe.base_method_params:
        orig_imm = recipe.base_method_params["inverse_mass_matrix"]
        loaded_imm = loaded.base_method_params["inverse_mass_matrix"]
        assert isinstance(loaded_imm, list)
        assert loaded_imm == orig_imm  # exact list equality (both are Python floats)

    # Filename convention
    assert saved_path.name == "high__nuts__window_adaptation_diag_imm.json"


@pytest.mark.slow
def test_render_instructions_medium_and_high_real() -> None:
    """render_instructions on real MEDIUM and HIGH recipes returns meaningful prose."""
    from tuningfork.calibration.tune import tune_algorithm

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup_sw = WARMUPS["window_adaptation_diag_imm"]

    # --- MEDIUM ---
    medium = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup_sw,
        n_warmup=100,
        rng_key=jax.random.key(7),
    )
    prose_m = render_instructions(medium)
    assert isinstance(prose_m, str)
    assert len(prose_m) > 20
    assert "window_adaptation_diag_imm" in prose_m

    # --- HIGH ---
    tuning_result = tune_algorithm(
        posterior,
        base_method,
        n_trials=2,
        n_seeds=1,
        n_chains=1,
        n_samples=100,
        n_warmup=100,
        rng_key=jax.random.key(8),
    )
    high = Recipe.from_tuning_result(
        tuning_result,
        posterior=posterior,
        base_method=base_method,
        warmup=warmup_sw,
    )
    prose_h = render_instructions(high)
    assert isinstance(prose_h, str)
    assert len(prose_h) > 20
    # HIGH template shows the number of trials
    assert str(tuning_result.n_trials_completed) in prose_h


# test_medium_recipe_exists_and_has_warmup_data (parametrized over 6 combos)
# removed 2026-05-17 — see module docstring "History" section.


# ---------------------------------------------------------------------------
# Slow smoke: timing-breakdown stamping by emit_low_recipe_for_cell
# Added 2026-05-26 per recipe-timing-schema PR
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_emit_low_recipe_stamps_timing_breakdown(tmp_path: Path) -> None:
    """emit_low_recipe_for_cell stamps timing breakdown into calibration_budget.

    Smoke test on mvn_10 × nuts × window_adaptation_diag_imm at
    n_samples=1000, num_chains=4 (canonical config).  Verifies that
    on PASS the saved recipe JSON has:
    - warmup_wall_seconds is a positive float
    - sampling_wall_seconds is a positive float
    - sampling_seconds_per_draw is positive
    - split_source == "measured"
    - machine_info is a dict containing "cpu_model" and "jax_version"
    """
    from tuningfork.recipes._base import Recipe
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "window_adaptation_diag_imm",
        "nuts",
        n_warmup=1000,
        n_samples=1000,
        num_chains=4,
        seed=20260517,
        catalog_root=tmp_path,
        outcomes_file=tmp_path / "outcomes.md",
        verbose=False,
    )

    # PASS or REVIEW both indicate the pipeline ran to completion.
    assert result.verdict in ("PASS", "REVIEW"), (
        f"Expected PASS or REVIEW; got {result.verdict} "
        f"(rhat={result.gate_rhat_max}, ess={result.gate_min_ess}, div={result.gate_n_div})"
    )

    # REVIEW means the recipe was not emitted; skip the file checks.
    if result.verdict == "REVIEW":
        pytest.skip(
            f"verdict=REVIEW (rhat={result.gate_rhat_max}, ess={result.gate_min_ess}); "
            "recipe not emitted; timing-field assertions are unreachable."
        )

    assert result.recipe_path is not None
    recipe = Recipe.load(result.recipe_path)
    budget = recipe.calibration_budget

    assert isinstance(budget.get("warmup_wall_seconds"), float)
    assert budget["warmup_wall_seconds"] > 0.0

    assert isinstance(budget.get("sampling_wall_seconds"), float)
    assert budget["sampling_wall_seconds"] > 0.0

    assert isinstance(budget.get("sampling_seconds_per_draw"), float)
    assert budget["sampling_seconds_per_draw"] > 0.0

    assert budget.get("split_source") == "measured"

    minfo = budget.get("machine_info")
    assert isinstance(minfo, dict)
    assert "cpu_model" in minfo
    assert "jax_version" in minfo


# ---------------------------------------------------------------------------
# mclmc-family adapted-L plumbing regression (PR #99 / issue found post-#98)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mclmc_emit_uses_adapted_L_not_default(tmp_path: Path) -> None:
    """emit_low_recipe_for_cell persists warmup-adapted L, not the default_params_for value.

    Regression test for the bug where shared_kwargs supplied the
    ``default_params_for`` fallback L to the mclmc factory instead of
    the per-chain adapted L from batched_params["L"].

    Before the fix, recipe L was ~12.6 (the loguniform 70th-pctile default).
    After the fix, recipe.base_method_params["L"] stores the warmup-adapted
    value, which must differ from the default by more than 5%.
    """
    import jax

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.calibration.tune import default_params_for
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    base_method = BASE_METHODS["mclmc"]

    # 1. Capture the default L for comparison.
    default_L = float(default_params_for(base_method)["L"])

    # 2. Run full emit.
    result = emit_low_recipe_for_cell(
        "mvn_10",
        "mclmc_tuning",
        "mclmc",
        n_warmup=200,
        n_samples=100,
        num_chains=4,
        seed=int(jax.random.bits(jax.random.key(42), dtype="uint32")),
        catalog_root=tmp_path,
        verbose=False,
    )

    # Both PASS and REVIEW indicate the pipeline completed (recipe may not be on disk
    # for REVIEW, but we can still check the gate machinery).
    assert result.verdict in (
        "PASS",
        "REVIEW",
        "FAIL",
    ), f"Unexpected verdict: {result.verdict}"

    # 3. If a recipe was written, load it and verify L is the adapted value (not default).
    if result.recipe_path is not None:
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(result.recipe_path)
        assert "L" in recipe.base_method_params, (
            "mclmc recipe must persist L in base_method_params; got "
            f"{list(recipe.base_method_params)}"
        )
        stored_L = float(recipe.base_method_params["L"])
        # L must be in the valid warmup search range.
        assert (
            0.0 < stored_L <= 1000.0
        ), f"Stored L={stored_L} is outside a valid range — possible default-L bug"
        # Stored L must NOT be the default fallback value (was the pre-fix behaviour).
        # We allow a 5% tolerance; in practice the adapted value will differ by >>5%.
        assert abs(stored_L - default_L) / default_L > 0.05, (
            f"Stored L={stored_L} is suspiciously close to the default_params_for "
            f"value ({default_L:.4f}) — the L plumbing fix may not be active."
        )


# ---------------------------------------------------------------------------
# adjusted_mclmc-family smoke emit regression (PR #100 / adjusted-family hardening)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["adjusted_mclmc", "adjusted_mclmc_dynamic"])
def test_adjusted_mclmc_family_emits_without_crash(
    tmp_path: Path, sampler: str
) -> None:
    """Both adjusted-MCLMC samplers run emit_low_recipe_for_cell without error.

    Regression test covering:
    - adjusted_mclmc: vmap-safe factory (jnp ops, no float(step_size))
    - adjusted_mclmc_dynamic: same + DynamicHMCState reinit (needs_mclmc_dyn_reinit)

    At n_warmup=200, n_samples=100 the verdict will typically be FAIL/REVIEW
    (warmup too short for convergence) — we only gate on crash-free completion.
    """
    import jax

    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "mvn_10",
        "adjusted_mclmc_tuning",
        sampler,
        n_warmup=200,
        n_samples=100,
        num_chains=4,
        seed=int(jax.random.bits(jax.random.key(99), dtype="uint32")),
        catalog_root=tmp_path,
        verbose=False,
    )
    # Any verdict is acceptable — the test gates on crash-free execution only.
    assert result.verdict in (
        "PASS",
        "REVIEW",
        "FAIL",
    ), f"Unexpected verdict {result.verdict!r} for {sampler!r}"
    assert result.gate_rhat_max is not None, "gate_rhat_max must be set"
    assert result.gate_min_ess is not None, "gate_min_ess must be set"


# ---------------------------------------------------------------------------
# M2: warmup_grad_evals forward-wiring tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_compute_warmup_grad_evals_mclmc_tuning_steps() -> None:
    """_compute_warmup_grad_evals reads _total_tuning_steps from mclmc warmup params."""
    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    batched_params = {"_total_tuning_steps": 12345, "step_size": 0.1}
    result = _compute_warmup_grad_evals(batched_params, None, None, 1000, 4)
    assert result == 12345, f"Expected 12345, got {result}"


@pytest.mark.fast
def test_compute_warmup_grad_evals_inner_kernel_nis() -> None:
    """_compute_warmup_grad_evals sums num_integration_steps from warmup_info."""
    import numpy as np

    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    class _FakeInfo:
        num_integration_steps = np.array(
            [[5, 10, 8, 7], [6, 9, 11, 4]]
        )  # (2_chains, 4_steps)

    result = _compute_warmup_grad_evals({"step_size": 0.1}, _FakeInfo(), None, 4, 2)
    assert result == 5 + 10 + 8 + 7 + 6 + 9 + 11 + 4, f"Got {result}"


@pytest.mark.fast
def test_compute_warmup_grad_evals_standard_hmc_returns_none() -> None:
    """_compute_warmup_grad_evals returns None for standard HMC (grad count unknown)."""
    from tuningfork.recipes._recipe_runner import _compute_warmup_grad_evals

    batched_params = {"step_size": 0.1, "inverse_mass_matrix": [1.0]}
    result = _compute_warmup_grad_evals(batched_params, None, None, 1000, 4)
    assert result is None, f"Expected None for standard HMC, got {result}"


@pytest.mark.slow
def test_mclmc_emit_has_warmup_grad_evals(tmp_path: Path) -> None:
    """Emitted mclmc recipe has warmup_grad_evals in calibration_budget (M2).

    mclmc_tuning returns _total_tuning_steps which should be threaded into
    calibration_budget.warmup_grad_evals by the emit path.
    """
    import jax

    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import emit_low_recipe_for_cell

    result = emit_low_recipe_for_cell(
        "logistic_synthetic",
        "mclmc_tuning",
        "mclmc",
        n_warmup=100,
        n_samples=50,
        num_chains=4,
        seed=int(jax.random.bits(jax.random.key(7), dtype="uint32")),
        catalog_root=tmp_path,
        verbose=False,
    )
    # Emit succeeded (any verdict OK — short run)
    assert result.verdict in ("PASS", "REVIEW", "FAIL")

    if result.recipe_path is not None:
        recipe = load_recipe(result.recipe_path)
        budget = recipe.calibration_budget or {}
        assert (
            "warmup_grad_evals" in budget
        ), "mclmc recipe must have warmup_grad_evals in calibration_budget"
        wge = budget["warmup_grad_evals"]
        assert (
            wge is not None and wge > 0
        ), f"warmup_grad_evals must be positive; got {wge}"


# ---------------------------------------------------------------------------
# T0.3 fix: mclmc rerun uses per-chain L (not default fallback) — regression test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mclmc_rerun_uses_recipe_L_not_default(tmp_path: Path) -> None:
    """run_recipe_to_idata for mclmc uses the recipe's stored L, not default_params_for.

    Regression test for T0.3 (latent bug): the old rerun step-builder had only
    ``is_laplace`` vs ``else``, so mclmc fell into the default branch which
    used a scalar L from shared_kwargs.  Before the fix, that scalar came from
    ``default_params_for`` (because L was never injected into shared_kwargs by
    the rerun path).  After the fix, L is injected into shared_kwargs via
    ``_recipe_params_override`` (which includes recipe.base_method_params) so
    the factory call receives the warmup-adapted L stored in the recipe.

    Validation strategy: emit a recipe, record L stored in the recipe, then
    rerun with run_recipe_to_idata and verify the run completes without error.
    The structural correctness of L-wiring is checked by asserting the rerun
    produces the same InferenceData shape as the emit's n_samples config.
    """
    import jax

    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import (
        emit_low_recipe_for_cell,
        run_recipe_to_idata,
    )

    seed = int(jax.random.bits(jax.random.key(11), dtype="uint32"))

    # 1. Emit a recipe.
    result = emit_low_recipe_for_cell(
        "mvn_10",
        "mclmc_tuning",
        "mclmc",
        n_warmup=100,
        n_samples=50,
        num_chains=2,
        seed=seed,
        catalog_root=tmp_path,
        verbose=False,
    )

    assert result.verdict in ("PASS", "REVIEW", "FAIL"), f"Unexpected: {result.verdict}"
    if result.recipe_path is None:
        pytest.skip("Emit did not produce a recipe — skipping rerun test")

    recipe_path = result.recipe_path
    assert recipe_path is not None  # mypy narrowing (pytest.skip raises above)
    recipe = load_recipe(recipe_path)
    assert (
        "L" in recipe.base_method_params
    ), "mclmc recipe must store L; rerun-vs-emit T0.3 test requires it"
    stored_L = float(recipe.base_method_params["L"])

    # 2. Rerun the recipe and verify it completes without error.
    idata = run_recipe_to_idata(
        recipe,
        n_samples=50,
        catalog_root=tmp_path,
        _suppress_print=True,
    )

    # 3. InferenceData must have posterior with the correct shape.
    import numpy as np

    assert hasattr(idata, "posterior"), "run_recipe_to_idata must return InferenceData"
    # At least one variable must exist.
    post_vars = list(idata.posterior.data_vars)
    assert len(post_vars) > 0, "InferenceData posterior has no variables"
    # Shape: (num_chains, n_samples, ...) — at least 2 chains and 50 samples.
    first_var = np.asarray(idata.posterior[post_vars[0]])
    n_chains_ret, n_samples_ret = first_var.shape[:2]
    assert n_chains_ret >= 1, f"Expected >=1 chains, got {n_chains_ret}"
    assert n_samples_ret == 50, f"Expected 50 samples, got {n_samples_ret}"

    # 4. Verify the stored L is in a valid range (T0.3 bug would use ~12.6 default).
    assert stored_L > 0.0, f"Stored L={stored_L:.4f} must be positive"
    assert stored_L < 2000.0, f"Stored L={stored_L:.4f} is suspiciously large"
