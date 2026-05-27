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
"""Tests for emit_script x64 injection and timing split.

Fast tests only (no JAX tracing, no subprocess execution):
- x64 config line present for requires_x64=True models (gp_regression).
- x64 config line absent for float32 models (eight_schools_ncp, logistic_synthetic).
- x64 line appears before the first model/JAX computation.
- Emitted script contains warmup_wall_seconds and sampling_wall_seconds prints.
- Warmup timing fence (_warmup_t0, _warmup_t1, _warmup_wall) present in emitted script.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe
from tuningfork.recipes._emit_script import _X64_CONFIG_LINE

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"

pytestmark = pytest.mark.fast

# ── Fixtures: lightweight synthetic recipes ────────────────────────────────


def _make_recipe(
    model_name: str,
    base_method_name: str = "nuts",
    warmup_name: str = "window_adaptation_diag_imm",
):
    """Build a minimal in-memory Recipe for the given model."""
    from tuningfork.recipes._base import Effort, Recipe

    return Recipe(
        model_name=model_name,
        base_method_name=base_method_name,
        warmup_name=warmup_name,
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 50, "target_acceptance_rate": 0.8},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )


# ── x64 injection tests ───────────────────────────────────────────────────


def test_x64_line_present_for_gp_regression() -> None:
    """emit_script for gp_regression (requires_x64=True) emits the x64 config line.

    The emitted preamble must contain ``jax.config.update("jax_enable_x64", True)``
    so the script runs correctly out of the box without the user manually setting
    JAX_ENABLE_X64=1.  Absence of this line caused Cholesky NaN failures before
    this fix (user-reported, 2026-05-27).
    """
    if not (_CATALOG_ROOT / "gp_regression").exists():
        # Fallback: construct a synthetic recipe when catalog artifacts are absent.
        recipe = _make_recipe("gp_regression")
    else:
        recipe_path = (
            _CATALOG_ROOT
            / "gp_regression"
            / "recipes"
            / "high__laplace_mhmc__window_adaptation_dense_imm__inner_laplace_hmc.json"
        )
        if recipe_path.exists():
            recipe = load_recipe(recipe_path)
        else:
            recipe = _make_recipe("gp_regression")

    script = emit_script(recipe, num_samples=10, num_chains=1)

    assert 'jax.config.update("jax_enable_x64", True)' in script, (
        "gp_regression requires_x64=True but the emitted script is missing the "
        "``jax.config.update('jax_enable_x64', True)`` line.\n"
        "This means the script will produce Cholesky NaN failures on float32.\n"
        f"Script preamble (first 600 chars):\n{script[:600]}"
    )


def test_x64_line_absent_for_eight_schools_ncp() -> None:
    """emit_script for eight_schools_ncp (float32) does NOT emit the x64 config line.

    Float32 models must not get the x64 config line — enabling x64 unconditionally
    would break reproducibility for models that were certified at float32 and
    change JAX's default behaviour for the user's session.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    if recipe_path.exists():
        recipe = load_recipe(recipe_path)
    else:
        recipe = _make_recipe("eight_schools_ncp")

    script = emit_script(recipe, num_samples=10)

    assert 'jax.config.update("jax_enable_x64", True)' not in script, (
        "eight_schools_ncp does NOT require x64, but the emitted script "
        "contains the x64 config line.\n"
        "Float32 models must not trigger x64 mode.\n"
        f"Script preamble (first 600 chars):\n{script[:600]}"
    )


@pytest.mark.parametrize(
    "model_name",
    ["mvn_10", "eight_schools_ncp"],
)
def test_x64_line_absent_for_float32_models(model_name: str) -> None:
    """Parametric: float32 models (mvn_10, eight_schools_ncp) must not get x64 line."""
    recipe = _make_recipe(model_name)
    script = emit_script(recipe, num_samples=10)

    assert 'jax.config.update("jax_enable_x64", True)' not in script, (
        f"Model '{model_name}' has requires_x64=False but the emitted script "
        "contains the x64 config line.\n"
        f"Script preamble (first 600 chars):\n{script[:600]}"
    )


def test_x64_line_precedes_model_computation() -> None:
    """The x64 config line appears before the first model/JAX computation.

    ``jax.config.update("jax_enable_x64", True)`` MUST appear before
    ``build_logdensity_fn`` (and before any ``jax.random.key`` call) — JAX
    commits to dtype precision on first use.  This test asserts the line is
    positioned right after ``import jax`` and before the ``from tuningfork.model``
    block.
    """
    recipe = _make_recipe("gp_regression")
    script = emit_script(recipe, num_samples=10, num_chains=1)
    lines = script.split("\n")

    # Find line numbers for anchor tokens.
    import_jax_line = next(
        (i for i, l in enumerate(lines) if l.strip() == "import jax"), None
    )
    x64_line = next((i for i, l in enumerate(lines) if "jax_enable_x64" in l), None)
    model_import_line = next(
        (i for i, l in enumerate(lines) if "from tuningfork.model import MODELS" in l),
        None,
    )

    assert import_jax_line is not None, "Could not find 'import jax' in emitted script"
    assert (
        x64_line is not None
    ), "gp_regression emitted script is missing 'jax_enable_x64' config line"
    assert (
        model_import_line is not None
    ), "Could not find 'from tuningfork.model import MODELS' in emitted script"

    assert (
        x64_line > import_jax_line
    ), f"x64 config line (line {x64_line}) must come AFTER 'import jax' (line {import_jax_line})"
    assert x64_line < model_import_line, (
        f"x64 config line (line {x64_line}) must come BEFORE model import "
        f"(line {model_import_line}) — JAX locks precision on first use.\n"
        "Ensure the x64 line is in the preamble, before build_logdensity_fn."
    )


def test_x64_config_line_is_syntactically_valid() -> None:
    """The _X64_CONFIG_LINE module constant is syntactically valid Python."""
    ast.parse(_X64_CONFIG_LINE)


# ── Timing split tests ────────────────────────────────────────────────────


def test_emitted_script_contains_warmup_timing_variables() -> None:
    """Emitted script defines the warmup timing variables needed for split reporting.

    Checks that the timing fence (_warmup_t0, _warmup_wall, _warmup_t1) is
    present in the emitted script — these feed into warmup_wall_seconds and
    sampling_wall_seconds in the postamble.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    if recipe_path.exists():
        recipe = load_recipe(recipe_path)
    else:
        recipe = _make_recipe("eight_schools_ncp")

    script = emit_script(recipe, num_samples=10)

    assert "_warmup_t0" in script, (
        "Emitted script is missing '_warmup_t0' (warmup timer start).\n"
        "The preamble must define _warmup_t0 after model setup so warmup "
        "wall time can be measured separately."
    )
    assert "_warmup_wall" in script, (
        "Emitted script is missing '_warmup_wall' (computed warmup duration).\n"
        "The warmup timing fence (inserted between warmup_body and sampler_body) "
        "must compute _warmup_wall = perf_counter() - _warmup_t0."
    )
    assert "_warmup_t1" in script, (
        "Emitted script is missing '_warmup_t1' (sampling phase timer start).\n"
        "The warmup timing fence must set _warmup_t1 for the postamble to "
        "compute sampling_wall_seconds."
    )


def test_emitted_script_prints_timing_split() -> None:
    """Postamble prints warmup_wall_seconds and sampling_wall_seconds separately.

    This is the honest-timing objective: the user should see how long warmup
    took vs how long sampling took, so they can evaluate amortized ESS/s
    correctly.  Both values feed the recipe re-stamp (future work).
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    if recipe_path.exists():
        recipe = load_recipe(recipe_path)
    else:
        recipe = _make_recipe("eight_schools_ncp")

    script = emit_script(recipe, num_samples=10)

    assert "warmup_wall_seconds=" in script, (
        "Emitted script postamble is missing 'warmup_wall_seconds=' print.\n"
        "The timing split is required for honest recipe timing."
    )
    assert "sampling_wall_seconds=" in script, (
        "Emitted script postamble is missing 'sampling_wall_seconds=' print.\n"
        "The timing split is required for honest recipe timing."
    )
    assert "wall_seconds=" in script, (
        "Emitted script postamble is missing 'wall_seconds=' (total wall time) print.\n"
        "Backward-compat: total wall time must still be reported."
    )


def test_emitted_script_has_block_until_ready_in_postamble() -> None:
    """Postamble calls jax.block_until_ready(_samples) before measuring sampling time.

    Without block_until_ready, JAX async dispatch means perf_counter() measures
    kernel dispatch latency (microseconds) rather than actual computation time.
    This test ensures the postamble has the barrier call before the timing stamp.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    if recipe_path.exists():
        recipe = load_recipe(recipe_path)
    else:
        recipe = _make_recipe("eight_schools_ncp")

    script = emit_script(recipe, num_samples=10)

    assert "jax.block_until_ready(_samples)" in script, (
        "Emitted script postamble is missing 'jax.block_until_ready(_samples)'.\n"
        "Without this barrier, the sampling_wall_seconds measurement is unreliable "
        "(JAX async dispatch makes perf_counter() measure dispatch, not compute time)."
    )


def test_emitted_script_timing_split_is_valid_python() -> None:
    """Emitted script with timing split is syntactically valid Python."""
    recipe = _make_recipe("eight_schools_ncp")
    script = emit_script(recipe, num_samples=10)
    ast.parse(script)  # raises SyntaxError on malformed output


def test_emitted_script_gp_regression_timing_split_is_valid_python() -> None:
    """Emitted gp_regression script (requires_x64) with timing split is valid Python."""
    recipe = _make_recipe("gp_regression")
    script = emit_script(recipe, num_samples=10, num_chains=1)
    ast.parse(script)  # raises SyntaxError on malformed output


# ── Progress bar tests ───────────────────────────────────────────────────


def test_emitted_window_adaptation_uses_progress_bar_true() -> None:
    """window_adaptation calls in emitted scripts use progress_bar=True.

    This makes warmup progress visible when running standalone (previously
    hardcoded to False, causing the script to appear hung during ~10 min
    Laplace warmup runs).
    """
    recipe = _make_recipe(
        "eight_schools_ncp",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
    )
    script = emit_script(recipe, num_samples=10)

    assert "progress_bar=True" in script, (
        "window_adaptation in emitted script should use progress_bar=True "
        "so warmup progress is visible in standalone runs.\n"
        f"Script warmup section:\n{script[script.find('# === WARMUP:'):script.find('# === SAMPLER:')]}"
    )
    assert "progress_bar=False" not in script, (
        "progress_bar=False must not appear in window_adaptation calls "
        "(warmup should show progress for observability).\n"
        f"Script warmup section:\n{script[script.find('# === WARMUP:'):script.find('# === SAMPLER:')]}"
    )


def test_emitted_dense_window_adaptation_uses_progress_bar_true() -> None:
    """window_adaptation_dense_imm emitted template uses progress_bar=True."""
    recipe = _make_recipe(
        "eight_schools_ncp",
        base_method_name="nuts",
        warmup_name="window_adaptation_dense_imm",
    )
    script = emit_script(recipe, num_samples=10)

    assert "progress_bar=True" in script, (
        "window_adaptation in emitted window_adaptation_dense_imm script "
        "should use progress_bar=True."
    )
