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
"""Tests for skip_warmup=True path in run_recipe_to_idata.

Tests cover:
- Fast: stationary init helper produces correct shape and per-chain offsets
- Fast: skip_warmup validation rejects laplace/mclmc/no-step_size recipes
- Slow: skip_warmup=True on eight_schools_ncp × nuts produces sane R̂ and
  zero divergences (verifying stored params + stationary init work end-to-end)
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pytest

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


# ---------------------------------------------------------------------------
# Fast: _build_stationary_init_positions unit tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_stationary_init_shape() -> None:
    """_build_stationary_init_positions returns batched dict with leading dim num_chains."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=4, catalog_root=_CATALOG_ROOT
    )
    # eight_schools_ncp has keys: mu (scalar), tau (scalar), theta_raw (8-vector)
    assert "mu" in positions
    assert "tau" in positions
    assert "theta_raw" in positions
    assert positions["mu"].shape == (4,), f"mu shape: {positions['mu'].shape}"
    assert positions["tau"].shape == (4,), f"tau shape: {positions['tau'].shape}"
    assert positions["theta_raw"].shape == (
        4,
        8,
    ), f"theta_raw shape: {positions['theta_raw'].shape}"


@pytest.mark.fast
def test_stationary_init_offsets_correct() -> None:
    """Chains are at gt_mean ± offsets * gt_std, not all at the same point."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=4, catalog_root=_CATALOG_ROOT
    )
    summary = json.loads(
        (_CATALOG_ROOT / "eight_schools_ncp" / "reference" / "summary.json").read_text()
    )
    gt_mean_mu = float(summary["mean"]["mu"])
    gt_std_mu = float(summary["std"]["mu"])

    mu_chains = np.asarray(positions["mu"])
    expected_offsets = [0.1, -0.1, 0.05, -0.05]
    for i, offset in enumerate(expected_offsets):
        expected = gt_mean_mu + offset * gt_std_mu
        assert (
            abs(float(mu_chains[i]) - expected) < 1e-5
        ), f"Chain {i}: expected mu={expected:.6f}, got {float(mu_chains[i]):.6f}"


@pytest.mark.fast
def test_stationary_init_num_chains_cycling() -> None:
    """num_chains=2 cycles offsets: chain 0 = +0.1σ, chain 1 = -0.1σ."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=2, catalog_root=_CATALOG_ROOT
    )
    assert positions["mu"].shape == (2,)
    # Chains should differ (not all at the same point)
    assert float(positions["mu"][0]) != float(positions["mu"][1])


@pytest.mark.fast
def test_stationary_init_missing_summary_raises() -> None:
    """FileNotFoundError raised when reference/summary.json is absent."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    with pytest.raises(FileNotFoundError, match="Reference summary not found"):
        _build_stationary_init_positions(
            "nonexistent_model", num_chains=4, catalog_root=_CATALOG_ROOT
        )


# ---------------------------------------------------------------------------
# Fast: skip_warmup validation checks
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_skip_warmup_rejects_laplace() -> None:
    """skip_warmup=True raises ValueError for laplace_* samplers."""
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = Recipe(
        model_name="eight_schools_ncp",
        base_method_name="laplace_dhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0, 1.0]},
        warmup_params={"n_warmup": 100, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
    )
    with pytest.raises(ValueError, match="not supported for laplace"):
        run_recipe_to_idata(recipe, skip_warmup=True)


@pytest.mark.fast
def test_skip_warmup_rejects_missing_step_size() -> None:
    """skip_warmup=True raises ValueError when step_size absent from recipe params."""
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = Recipe(
        model_name="eight_schools_ncp",
        base_method_name="nuts",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={},  # missing step_size
        warmup_params={"n_warmup": 0, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
    )
    with pytest.raises(ValueError, match="step_size"):
        run_recipe_to_idata(recipe, skip_warmup=True)


@pytest.mark.fast
def test_skip_warmup_rejects_missing_imm() -> None:
    """skip_warmup=True raises ValueError when inverse_mass_matrix absent."""
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = Recipe(
        model_name="eight_schools_ncp",
        base_method_name="nuts",
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.5},  # missing inverse_mass_matrix
        warmup_params={"n_warmup": 0, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
    )
    with pytest.raises(ValueError, match="inverse_mass_matrix"):
        run_recipe_to_idata(recipe, skip_warmup=True)


# ---------------------------------------------------------------------------
# Slow: end-to-end correctness — skip_warmup=True on a committed recipe
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_skip_warmup_eight_schools_nuts_sane_diagnostics() -> None:
    """skip_warmup=True on eight_schools_ncp nuts yields R̂<1.1, n_div=0.

    Uses the committed low__nuts__window_adaptation_diag_imm.json recipe with
    n_samples=200 to keep wall time short.  Stationary init from GT-means
    ensures chains start near the posterior so sampling is valid without any
    warmup.

    Validates:
    - R̂ max < 1.1 (lenient; 200 samples on a simple model)
    - divergence rate == 0 (stored step_size from a good warmup should stay safe)
    - warmup wall time ≈ 0 (< 1 s; verifiable via _return_timing)
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(recipe_path)

    idata, t_warmup, t_sample = run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=200,
        _return_timing=True,
    )

    # Warmup should be essentially free
    assert t_warmup < 1.0, f"skip_warmup should have t_warmup≈0, got {t_warmup:.3f}s"

    # Posterior group must exist
    assert hasattr(idata, "posterior"), "InferenceData missing posterior group"

    # R-hat sanity check (lenient threshold for 200 samples on a simple model)
    rhat = az.rhat(idata)
    rhat_values: list[float] = []
    for var in rhat.data_vars:
        rhat_values.extend(float(v) for v in rhat[var].values.ravel())
    rhat_max = max(rhat_values) if rhat_values else 0.0
    assert rhat_max < 1.1, (
        f"R̂ max={rhat_max:.4f} exceeds 1.1 — stationary init may be too far "
        "from posterior or stored step_size is stale."
    )

    # Divergence check
    if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
        n_div = int(idata.sample_stats.diverging.values.sum())
        assert n_div == 0, (
            f"Expected 0 divergences with skip_warmup=True on eight_schools_ncp, "
            f"got {n_div}"
        )


@pytest.mark.slow
def test_skip_warmup_lotka_volterra_x64_dtype() -> None:
    """skip_warmup=True on lotka_volterra (x64 model) uses float64 throughout.

    Lotka_volterra requires x64 precision (requires_x64=True).  This test is
    the dtype lock: if _build_stationary_init_positions hard-casts to float32,
    the kernel init will fail with a dtype mismatch or silently lose precision.

    Validates:
    - x64 is auto-enabled by run_recipe_to_idata before the skip_warmup path
    - stationary init positions are float64 (not float32)
    - sampling produces R̂ < 1.2 and n_div == 0 (more lenient; stiff ODE model)
    - warmup wall time < 1 s
    """
    import jax
    import jax.numpy as jnp

    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import (
        _build_stationary_init_positions,
        run_recipe_to_idata,
    )

    recipe_path = (
        _CATALOG_ROOT
        / "lotka_volterra"
        / "recipes"
        / "medium__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(recipe_path)

    idata, t_warmup, _t_sample = run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=100,
        _return_timing=True,
    )

    # x64 should have been auto-enabled
    assert jax.config.read(
        "jax_enable_x64"
    ), "run_recipe_to_idata should have auto-enabled x64 for lotka_volterra"

    # Stationary init must be float64 when x64 is enabled
    positions = _build_stationary_init_positions(
        "lotka_volterra", num_chains=4, catalog_root=_CATALOG_ROOT
    )
    for key, arr in positions.items():
        assert arr.dtype == jnp.float64, (
            f"Position '{key}' has dtype {arr.dtype}, expected float64 "
            "when jax_enable_x64 is True."
        )

    # Warmup should be essentially free
    assert t_warmup < 1.0, f"skip_warmup should have t_warmup≈0, got {t_warmup:.3f}s"

    # Posterior group must exist
    assert hasattr(idata, "posterior"), "InferenceData missing posterior group"

    # R-hat sanity (more lenient; stiff ODE model + only 100 samples)
    rhat = az.rhat(idata)
    rhat_values: list[float] = []
    for var in rhat.data_vars:
        rhat_values.extend(float(v) for v in rhat[var].values.ravel())
    rhat_max = max(rhat_values) if rhat_values else 0.0
    assert rhat_max < 1.2, (
        f"R̂ max={rhat_max:.4f} exceeds 1.2 for lotka_volterra skip_warmup — "
        "stationary init may be far from posterior or stored step_size is stale."
    )

    # Divergence check
    if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
        n_div = int(idata.sample_stats.diverging.values.sum())
        assert n_div == 0, (
            f"Expected 0 divergences with skip_warmup=True on lotka_volterra, "
            f"got {n_div}"
        )


# ---------------------------------------------------------------------------
# Slow: regression — Bug 1 (sidecar IMM) + Bug 2 (num_integration_steps)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_skip_warmup_sidecar_imm_irt_2pl_nuts() -> None:
    """skip_warmup=True on irt_2pl × nuts (sidecar IMM) loads sidecar and runs.

    Regression for Bug 1: recipes with inverse_mass_matrix='sidecar' previously
    failed with jnp.asarray('sidecar').  This test verifies the sidecar is loaded
    transparently and the run produces valid InferenceData.

    irt_2pl is a D>50 model whose diag-IMM recipe stores the IMM as a sidecar
    .imm.npz to keep the JSON small.  nuts does not require num_integration_steps
    so this test isolates Bug 1.
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = (
        _CATALOG_ROOT
        / "irt_2pl"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(recipe_path)

    # Sidecar token must be present for this test to be meaningful
    assert recipe.base_method_params["inverse_mass_matrix"] == "sidecar", (
        "Expected irt_2pl nuts recipe to have inverse_mass_matrix='sidecar'; "
        "test precondition failed — recipe may have changed."
    )

    idata, t_warmup, _t_sample = run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=100,
        _return_timing=True,
    )

    assert t_warmup < 1.0, f"skip_warmup should have t_warmup≈0, got {t_warmup:.3f}s"
    assert hasattr(idata, "posterior"), "InferenceData missing posterior group"

    rhat = az.rhat(idata)
    rhat_values: list[float] = []
    for var in rhat.data_vars:
        rhat_values.extend(float(v) for v in rhat[var].values.ravel())
    rhat_max = max(rhat_values) if rhat_values else 0.0
    assert rhat_max < 1.2, (
        f"R̂ max={rhat_max:.4f} exceeds 1.2 for irt_2pl skip_warmup — "
        "sidecar IMM may have been loaded incorrectly or is stale."
    )


@pytest.mark.slow
def test_skip_warmup_sidecar_imm_and_nis_radon_mhmc() -> None:
    """skip_warmup=True on radon × mhmc (sidecar IMM + num_integration_steps).

    Regression for Bug 1 + Bug 2 combined:
    - Bug 1: inverse_mass_matrix='sidecar' must be resolved from .imm.npz.
    - Bug 2: mhmc factory requires num_integration_steps; the skip_warmup path
      previously built the init kernel without it, raising
      "as_top_level_api() missing 1 required positional argument: 'num_integration_steps'".

    radon × mhmc (low__mhmc__window_adaptation_diag_imm.json) has both
    inverse_mass_matrix='sidecar' and num_integration_steps=64 in
    base_method_params — the minimal committed recipe that exercises both fixes.
    """
    from tuningfork.catalog.inspect import load_recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe_path = (
        _CATALOG_ROOT
        / "radon"
        / "recipes"
        / "low__mhmc__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(recipe_path)

    # Preconditions: both sidecar token and num_integration_steps must be present
    assert recipe.base_method_params["inverse_mass_matrix"] == "sidecar", (
        "Expected radon mhmc recipe to have inverse_mass_matrix='sidecar'; "
        "test precondition failed."
    )
    assert "num_integration_steps" in recipe.base_method_params, (
        "Expected radon mhmc recipe to carry num_integration_steps; "
        "test precondition failed."
    )

    idata, t_warmup, _t_sample = run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=100,
        _return_timing=True,
    )

    assert t_warmup < 1.0, f"skip_warmup should have t_warmup≈0, got {t_warmup:.3f}s"
    assert hasattr(idata, "posterior"), "InferenceData missing posterior group"

    rhat = az.rhat(idata)
    rhat_values: list[float] = []
    for var in rhat.data_vars:
        rhat_values.extend(float(v) for v in rhat[var].values.ravel())
    rhat_max = max(rhat_values) if rhat_values else 0.0
    assert rhat_max < 1.2, (
        f"R̂ max={rhat_max:.4f} exceeds 1.2 for radon skip_warmup × mhmc — "
        "sidecar load or num_integration_steps injection may be broken."
    )
