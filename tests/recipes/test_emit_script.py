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
"""Tests for emit_script -- recipe to reproduction Python script.

The core **inference choreography** (warmup + sampler + inference loop) is
emitted inline. The **model definition** is imported via
``from tuningfork.model import MODELS``: the canonical NumPyro code lives
upstream and is not duplicated in the generated program. The optional tap diagnostics
wrapper imports its canonical compatibility and artifact policy.

The tests below enforce:

- Emitted script is syntactically valid Python.
- The specific tuningfork imports allowed are the two model modules and the
  opt-in tap module; no recipe schema, calibration code, or sampler/warmup
  wrappers may be imported.
- The emitted script executes end-to-end and reports divergence count.

NOTE: e2e emit-execute tests run a minimal 10-sample / minimal-warmup config;
they assert the emitted script executes (structure correct, no vmap/io_callback
errors), NOT inference quality. This keeps the e2e gate fast and memory-safe.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"

# Top-level packages the emitted script may import.  tuningfork imports are
# limited to canonical model and diagnostics modules.
# stdlib modules used by the emitted preamble/postamble (e.g. ``time`` for
# wall-clock timing, ``json`` for machine-readable timing, diagnostics cleanup,
# and ``warnings`` for the progress_bar=True single-chain warmup advisory) are
# also allowlisted here.
# VI programs additionally import ``optax`` (VI optimizer) and ``collections``
# (stdlib; for inline NamedTuple definition without the typing module).
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "jax",
        "numpy",
        "numpyro",
        "blackjax",
        "arviz",
        "tuningfork",
        "time",
        "json",
        "atexit",
        "contextlib",
        "logging",
        "os",
        "warnings",
        "optax",
        "collections",
    }
)


@pytest.mark.fast
def test_emit_script_rmhmc_generated_valid_python() -> None:
    """The generated RMHMC program is valid and calls the intended API.

    This uses an existing eight_schools_ncp HMC recipe as a scaffold,
    substituting ``base_method_name`` with ``rmhmc`` so the generated emitter
    produces the corresponding sampler call. It checks:

    1. The emitted script is syntactically valid Python.
    2. It calls ``blackjax.rmhmc`` (not ``blackjax.hmc``).
    3. It does NOT call ``blackjax.hmc``.
    4. It contains the ``_imm_to_mass_matrix`` helper required by the upstream
       RMHMC API.
    """
    recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__hmc__window_adaptation_diag_imm.json"
    )
    if not recipe_path.exists():
        pytest.skip("eight_schools_ncp hmc recipe not in catalog — run emit first")

    hmc_recipe = load_recipe(recipe_path)
    # Swap only the method name; the recipe parameters are structurally
    # compatible with the RMHMC emitter.
    rmhmc_recipe = dataclasses.replace(hmc_recipe, base_method_name="rmhmc")

    script = emit_script(rmhmc_recipe)

    # 1. Syntactically valid Python
    ast.parse(script)

    # 2. Calls the right upstream API
    assert "blackjax.rmhmc(" in script, (
        "generated RMHMC program must call blackjax.rmhmc(...) — got:\n"
        + script[max(0, script.find("blackjax.")) : script.find("blackjax.") + 80]
    )

    # 3. No stray blackjax.hmc calls.
    assert (
        "blackjax.hmc(" not in script
    ), "generated RMHMC program must not contain blackjax.hmc() calls"

    # 4. IMM→mass_matrix helper present (required because upstream rmhmc takes
    #    mass_matrix not inverse_mass_matrix)
    assert (
        "_imm_to_mass_matrix" in script
    ), "generated RMHMC program must define _imm_to_mass_matrix"


@pytest.mark.fast
def test_emit_script_imports_only_allowed_modules() -> None:
    """Emitted script imports only the allowlisted top-level packages."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    tree = ast.parse(script)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in _ALLOWED_TOP_LEVEL, f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            top = node.module.split(".")[0]
            assert top in _ALLOWED_TOP_LEVEL, f"Disallowed from-import: {node.module}"


@pytest.mark.fast
def test_emit_script_num_samples_defaults_to_calibration_budget() -> None:
    """emit_script reads calibration_budget['n_samples'] when num_samples not passed.

    Regression gate for the bug where emit_script defaulted to 2000 samples
    regardless of the recipe's validated calibration_budget. Calling
    emit_script(recipe) without num_samples must set ``_NUM_SAMPLES`` from
    ``calibration_budget['n_samples']`` (re-stamped to the 1000×4 production
    config 2026-05-27), not a hardcoded fallback.
    """
    recipe_path = (
        _CATALOG_ROOT
        / "gp_regression"
        / "recipes"
        / "high__laplace_mhmc__window_adaptation_dense_imm__inner_laplace_hmc.json"
    )
    if not recipe_path.exists():
        pytest.skip("HIGH laplace_mhmc recipe not in catalog")
    recipe = load_recipe(recipe_path)

    # Sanity: recipe carries an integer calibration_budget["n_samples"]
    # (re-stamped to the 1000×4 production config 2026-05-27 — assert dynamically
    # against whatever the recipe currently declares, not a hardcoded value).
    n_samples = recipe.calibration_budget.get("n_samples")
    assert isinstance(
        n_samples, int
    ), f"Unexpected n_samples in calibration_budget: {recipe.calibration_budget}"

    # emit_script with no num_samples arg → must use calibration_budget value
    script = emit_script(recipe)
    needle = "_NUM_SAMPLES"
    excerpt = script[script.find(needle) : script.find(needle) + 40]
    assert f"_NUM_SAMPLES = {n_samples}" in script, (
        "emit_script(recipe) must set _NUM_SAMPLES from calibration_budget['n_samples'] "
        f"(expected {n_samples}). Got script excerpt:\n{excerpt}"
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "warmup_name,base_method_name",
    [
        # Conventional gradient samplers with compatible warmups.
        ("window_adaptation_diag_imm", "hmc"),
        ("no_warmup", "dynamic_hmc"),
        ("no_warmup", "mhmc"),
        ("no_warmup", "dmhmc"),
        ("no_warmup", "ghmc"),
        # Random-walk / Langevin samplers.
        ("no_warmup", "mala"),
        ("no_warmup", "barker"),
        ("no_warmup", "rwm"),
    ],
)
def test_emit_script_executes_for_cell(
    warmup_name: str, base_method_name: str, tmp_path: Path
) -> None:
    """Synthetic recipe for (mvn_10, warmup, sampler) cell; verify emit + exec works.

    Each parametrised cell covers a distinct warmup × sampler pairing.
    Tests that the assembled script is executable end-to-end (exit 0, prints DONE,
    reports n_divergences).  Uses mvn_10 (well-behaved 10-D Gaussian) for speed.
    Lightweight config: num_samples=10, num_warmup=10 (just enough to verify structure).
    """
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.recipes._base import Recipe

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS[base_method_name]
    recipe = Recipe.from_default_config(posterior, base_method)
    if warmup_name != "no_warmup":
        recipe = dataclasses.replace(
            recipe,
            warmup_name=warmup_name,
            warmup_params={"n_warmup": 10, "num_chains": 1},
            warmups=[
                {"name": warmup_name, "params": {"n_warmup": 10, "num_chains": 1}}
            ],
        )

    script = emit_script(recipe, num_samples=10, num_warmup=10)
    script_path = tmp_path / "test_emitted.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Emitted script failed for {warmup_name}+{base_method_name}:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout
    assert "n_divergences=" in result.stdout


# ---------------------------------------------------------------------------
# Multi-chain (num_chains) tests (emit-script-num-chains feature)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_script_num_chains_derived_from_4chain_recipe() -> None:
    """emit_script(recipe, num_chains=None) derives num_chains=4 for a LOW recipe.

    LOW/MEDIUM recipes record ``num_chains=4`` in warmup_params (and
    calibration_budget).  When num_chains is not passed explicitly, emit_script
    reads it from the recipe metadata.
    """
    low_recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(low_recipe_path)
    script = emit_script(recipe)
    # The preamble must declare num_chains = 4.
    assert "num_chains = 4" in script, (
        "Expected 'num_chains = 4' in the emitted script preamble for a "
        f"LOW recipe with num_chains=4 in warmup_params.\nScript start:\n{script[:800]}"
    )


@pytest.mark.fast
def test_emit_script_num_chains_defaults_to_1_for_groundtruth() -> None:
    """emit_script(recipe, num_chains=None) falls back to 1 for groundtruth recipes.

    Groundtruth recipes pre-date the num_chains field; both warmup_params and
    calibration_budget omit it.  The fallback must be 1 (single-chain
    reproduction).
    """
    gt_recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(gt_recipe_path)
    script = emit_script(recipe)
    assert "num_chains = 1" in script, (
        "Expected 'num_chains = 1' in the emitted script preamble for a "
        "groundtruth recipe that has no num_chains field.\n"
        f"Script start:\n{script[:800]}"
    )


@pytest.mark.e2e
def test_emit_script_perchain_warmup_adapted_params_shape(tmp_path: Path) -> None:
    """Per-chain warmup (progress_bar=False) produces vector _adapted_params["step_size"].

    Uses the LOW recipe for eight_schools_ncp x nuts x window_adaptation_diag_imm
    which encodes num_chains=4.  Verifies that:
    - _adapted_params["step_size"] is shape (num_chains,) — ndim=1 (per-chain warmup)
    - _samples has shape (4, n_samples, ...)  (multi-chain output via scan(vmap))

    With progress_bar=False, warmup runs per-chain via jax.vmap, so each chain
    gets its own adapted step_size.  _adapted_params["step_size"] is therefore
    shape (num_chains,), not scalar.

    Lightweight config: num_samples=10 for e2e speed.
    """
    low_recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(low_recipe_path)
    _NUM_SAMPLES = 10
    _NUM_CHAINS = 4
    # progress_bar=False → per-chain warmup via jax.vmap → step_size shape (num_chains,).
    script = emit_script(recipe, num_samples=_NUM_SAMPLES, progress_bar=False)
    # Inject verification prints after the warmup and after the inference loop.
    verification = (
        "\nimport jax as _jax\n"
        "import numpy as _np\n"
        # Check _adapted_params["step_size"] shape — must be (num_chains,) for per-chain warmup.
        "_ss = _adapted_params['step_size']\n"
        'print("STEP_SIZE_NDIM=" + str(int(_np.asarray(_ss).ndim)))\n'
        # Check _samples shape — first leaf must start with (4, _NUM_SAMPLES, ...).
        "_first_leaf = _jax.tree.leaves(_samples)[0]\n"
        'print("SAMPLES_SHAPE=" + str(_first_leaf.shape))\n'
    )
    script += verification
    script_path = tmp_path / "test_perchain_warmup.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Per-chain warmup emitted script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout

    # _adapted_params["step_size"] must be shape (num_chains,) ndim=1 — per-chain warmup.
    assert "STEP_SIZE_NDIM=1" in result.stdout, (
        "Expected _adapted_params['step_size'] to be shape (num_chains,) (ndim=1) "
        "for per-chain warmup (progress_bar=False path uses jax.vmap).\n"
        f"stdout:\n{result.stdout}"
    )
    # _samples must have shape (4, n_samples, ...).
    assert f"SAMPLES_SHAPE=({_NUM_CHAINS}, {_NUM_SAMPLES}" in result.stdout, (
        f"Expected _samples shape starting with ({_NUM_CHAINS}, {_NUM_SAMPLES}, ...) "
        f"but got different output.\nstdout:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Warmup algorithm correctness tests (emit-script-warmup-algo-fix)
# ---------------------------------------------------------------------------


def _extract_warmup_section(script: str) -> str:
    """Return the substring from `# === WARMUP:` to the next `# === SAMPLER:`."""
    lines = script.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("# === WARMUP:"))
    end = next(
        i
        for i, line in enumerate(lines[start:], start=start)
        if line.startswith("# === SAMPLER:")
    )
    return "\n".join(lines[start:end])


@pytest.mark.fast
@pytest.mark.parametrize(
    "sampler",
    [
        "nuts",
        "hmc",
        "mhmc",
        "dynamic_hmc",
        "dmhmc",
    ],
)
@pytest.mark.parametrize(
    "warmup",
    [
        "window_adaptation_diag_imm",
        "window_adaptation_dense_imm",
        "window_adaptation_low_rank_imm",
    ],
)
def test_emit_script_warmup_protocol_selects_expected_algorithm(
    sampler: str, warmup: str
) -> None:
    """The generated warmup protocol selects the expected BlackJAX algorithm.

    Catches the class of bug where templates hardcode an algorithm name
    (e.g., `blackjax.nuts`) regardless of the recipe's actual sampler.
    Discovered 2026-05-20 on `medium__mhmc__window_adaptation_dense_imm`.

    Laplace samplers use their dedicated generated protocol.
    """
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._warmup_protocol import WARMUP_SUBSTITUTE_METHOD_NAMES
    from tuningfork.warmup import WARMUPS

    # Skip incompatible pairs (e.g., low_rank_window_adaptation may not support
    # all sampler families).
    if not WARMUPS[warmup].is_compatible(sampler):
        pytest.skip(f"{warmup} is not compatible with {sampler}")

    # Create a synthetic Recipe in-memory with minimal required fields.
    # The emit_script function only uses: model_name, base_method_name, warmup_name,
    # effort, base_method_params, warmup_params, tuning_seed, calibration_budget,
    # and gate_evidence. Most of these can be stubbed for the syntax check.
    # window_adaptation_low_rank_imm requires max_rank in warmup_params.
    _wp = {"n_warmup": 100, "target_acceptance_rate": 0.8}
    if warmup == "window_adaptation_low_rank_imm":
        _wp["max_rank"] = 3

    recipe = Recipe(
        model_name="eight_schools_ncp",
        base_method_name=sampler,
        warmup_name=warmup,
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params=_wp,
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=50)

    # Substitute-family methods use the protocol's NUTS warmup kernel.
    expected_algo = "nuts" if sampler in WARMUP_SUBSTITUTE_METHOD_NAMES else sampler
    warmup_section = _extract_warmup_section(script)

    assert f"blackjax.{expected_algo}" in warmup_section, (
        f"Emitted script for {sampler} × {warmup}: warmup section "
        f"should call `blackjax.{expected_algo}` but doesn't. "
        f"Section:\n{warmup_section[:500]}"
    )


@pytest.mark.fast
def test_hmc_warmup_uses_recipe_pinned_trajectory_length() -> None:
    """A material HMC override must reach both generated warmup and sampling."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "num_integration_steps": 3},
        warmup_params={"n_warmup": 10, "target_acceptance_rate": 0.8},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    warmup_source = _extract_warmup_section(
        emit_script(recipe, num_samples=2, num_chains=1)
    )
    assert "num_integration_steps=3" in warmup_source


@pytest.mark.fast
def test_emit_script_warmup_inner_kernel_override_emits_correct_algo() -> None:
    """Schema extension: warmup_inner_kernel=nuts overrides hmc recipe to emit blackjax.nuts.

    When recipe.warmup_inner_kernel is explicitly set AND differs from the
    implicit default (hmc -> hmc is implicit; hmc + inner_nuts overrides to nuts),
    the emitted script's warmup section must reference blackjax.nuts, not
    blackjax.hmc.

    This mirrors the generated warmup protocol's inner-kernel selection.
    """
    from tuningfork.recipes._base import Effort, Recipe

    # hmc with explicit warmup_inner_kernel="nuts" (non-default override)
    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1, "num_integration_steps": 10},
        warmup_params={"n_warmup": 200, "target_acceptance_rate": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 200, "target_acceptance_rate": 0.8},
            }
        ],
        warmup_inner_kernel="nuts",  # Schema extension explicit override
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=50)
    warmup_section = _extract_warmup_section(script)

    # hmc's implicit default is "hmc" (not in WARMUP_SUBSTITUTE_METHOD_NAMES);
    # the override to "nuts" must appear in the warmup section.
    assert "blackjax.nuts" in warmup_section, (
        "Schema extension: warmup_inner_kernel='nuts' override should make the emitted "
        f"script use blackjax.nuts in the warmup section.\nSection:\n{warmup_section[:500]}"
    )
    # hmc itself should NOT appear as the warmup algorithm
    # (it would be blackjax.hmc if the override didn't work).
    assert "blackjax.hmc" not in warmup_section, (
        "Schema extension: blackjax.hmc should NOT appear in the warmup section when "
        f"warmup_inner_kernel='nuts' overrides it.\nSection:\n{warmup_section[:500]}"
    )


@pytest.mark.fast
def test_adapted_hmc_sidecar_sentinel_is_not_emitted_inline() -> None:
    """Adapted runs leave sidecar-backed IMM resolution to warmup."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={
            "step_size": 0.1,
            "num_integration_steps": 7,
            "inverse_mass_matrix": "sidecar",
        },
        warmup_params={"n_warmup": 1},
        calibration_budget={},
        headline_metric=None,
        sample_quality=None,
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=1, num_chains=1)
    assert "jnp.asarray('sidecar')" not in script
    assert "_default_imm = jnp.ones(_n_params)" in script


@pytest.mark.fast
def test_no_warmup_replay_rejects_sidecar_imm_sentinel() -> None:
    """Pinned replay cannot defer IMM resolution to a sidecar at runtime."""
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="",
        effort=Effort.LOW,
        base_method_params={
            "step_size": 0.1,
            "num_integration_steps": 7,
            "inverse_mass_matrix": "sidecar",
        },
        warmup_params={},
        calibration_budget={
            "baked_from": {"warmup_name": "window_adaptation_diag_imm"}
        },
        headline_metric=None,
        sample_quality=None,
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    with pytest.raises(ValueError, match="numeric inline .*inverse_mass_matrix"):
        emit_script(recipe, num_samples=1, num_chains=1)


@pytest.mark.fast
def test_meads_low_rank_configuration_fails_closed() -> None:
    """Codegen must reject MEADS metric structures it cannot preserve."""
    from dataclasses import replace

    recipe = load_recipe(_CATALOG_ROOT / "mvn_10" / "recipes" / "low__ghmc__meads.json")
    recipe = replace(
        recipe,
        warmup_params={**recipe.warmup_params, "low_rank_rank": 2},
        warmups=[
            {
                "name": "meads",
                "params": {**recipe.warmup_params, "low_rank_rank": 2},
            }
        ],
    )

    with pytest.raises(NotImplementedError, match="MEADS low-rank"):
        emit_script(recipe, num_samples=1, num_warmup=1, num_chains=8)


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("recipe_name", "num_chains"),
    [
        ("low__dynamic_hmc__chees.json", 4),
        ("low__ghmc__meads.json", 8),
    ],
)
def test_generated_ensemble_warmup_executes(
    tmp_path: Path, recipe_name: str, num_chains: int
) -> None:
    """CHEES and MEADS execute through the same generated-program boundary."""
    import numpy as np

    from tuningfork.catalog import execute_recipe

    recipe = load_recipe(_CATALOG_ROOT / "mvn_10" / "recipes" / recipe_name)
    result = execute_recipe(
        recipe,
        tmp_path / recipe_name,
        num_samples=2,
        num_warmup=1,
        num_chains=num_chains,
        timeout=120,
    )

    assert result.artifact_path is not None
    with np.load(result.artifact_path, allow_pickle=False) as artifact:
        assert artifact["x"].shape == (num_chains, 2, 10)
    assert result.timings is not None


@pytest.mark.e2e
def test_generated_pinned_reference_summary_replay_executes(tmp_path: Path) -> None:
    """Pinned no-warmup replay initializes and samples every chain in codegen."""
    import hashlib
    import json
    from dataclasses import replace

    import numpy as np

    from tuningfork.catalog import execute_recipe

    recipe = load_recipe(
        _CATALOG_ROOT
        / "mvn_10"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    summary_path = _CATALOG_ROOT / "mvn_10" / "reference" / "summary.json"
    raw_summary = summary_path.read_bytes()
    summary = json.loads(raw_summary)
    replay = replace(
        recipe.normalize_pinned_replay(),
        init_strategy={
            "type": "reference_summary",
            "mean": summary["mean"],
            "std": summary["std"],
            "offsets": [0.1, -0.1],
            "source_path": "mvn_10/reference/summary.json",
            "source_sha256": hashlib.sha256(raw_summary).hexdigest(),
        },
    )

    result = execute_recipe(
        replay,
        tmp_path,
        num_samples=3,
        num_chains=2,
        timeout=120,
    )

    assert result.artifact_path is not None
    with np.load(result.artifact_path, allow_pickle=False) as artifact:
        assert artifact["x"].shape == (2, 3, 10)
    assert result.timings is not None
    assert result.timings.warmup_seconds >= 0
    assert result.timings.sampling_seconds >= 0


@pytest.mark.e2e
def test_emit_script_warmup_execution_contract_mhmc_dense(tmp_path: Path) -> None:
    """Generated warmup execution produces valid adapted parameters.

    Target recipe: eight_schools_ncp × mhmc × window_adaptation_dense_imm.
    User-reported bug case (2026-05-20). MHMC is in the standard
    (non-HMC-substitute) warmup path; the fix puts blackjax.mhmc in
    the emitted warmup call.

    This test verifies that generated warmup completes without error and
    produces finite adapted parameters of the correct structure.

    Lightweight config: num_samples=10 minimal warmup (recipe default overridden).
    """
    recipe = load_recipe(
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "medium__mhmc__window_adaptation_dense_imm.json"
    )

    # Emitted script's warmup (executed in subprocess for isolation).
    # Use num_samples=10, num_warmup=10 to keep the test fast.
    # Changed 2026-07-08 (blackjax #964 stage 2): progress_bar=True no longer
    # forces single-chain emit (topology is unaffected by the flag). Use
    # warmup_num_chains=[1] -- the one knob that still selects single-chain --
    # to preserve the single-chain emit this test was designed for.
    script = emit_script(recipe, num_samples=10, num_warmup=10, warmup_num_chains=[1])

    epilogue = """
import json
import numpy as np

# adapted_params after warmup: persist for the test process to read.
_emitted = {
    "step_size": float(np.asarray(_adapted_params["step_size"])),
    "imm_ndim": int(np.asarray(_adapted_params["inverse_mass_matrix"]).ndim),
    "imm_shape": list(np.asarray(_adapted_params["inverse_mass_matrix"]).shape),
    "step_size_finite": bool(np.isfinite(np.asarray(_adapted_params["step_size"]))),
    "imm_finite": bool(np.all(np.isfinite(np.asarray(_adapted_params["inverse_mass_matrix"])))),
}
with open("/tmp/test_emit_script_mhmc_dense_params.json", "w") as _f:
    json.dump(_emitted, _f)
print("GENERATED PROGRAM OK")
"""

    script_path = tmp_path / "test_mhmc_dense.py"
    script_path.write_text(script + "\n\n" + epilogue)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Emitted script crashed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "GENERATED PROGRAM OK" in result.stdout

    import json

    with open("/tmp/test_emit_script_mhmc_dense_params.json") as f:
        emitted = json.load(f)

    # Verify structure: single-chain warmup produces scalar step_size (ndim=0).
    assert emitted[
        "step_size_finite"
    ], f"Emitted _adapted_params['step_size'] is not finite: {emitted['step_size']}"
    assert emitted[
        "imm_finite"
    ], "Emitted _adapted_params['inverse_mass_matrix'] has non-finite values"
    # Dense IMM for eight_schools_ncp (9 free params): shape should be (9, 9).
    assert emitted["imm_ndim"] == 2, (
        f"Expected dense IMM (ndim=2) for window_adaptation_dense_imm, "
        f"got ndim={emitted['imm_ndim']}"
    )


# ---------------------------------------------------------------------------
# Laplace recipe emission tests
# ---------------------------------------------------------------------------

_LAPLACE_HIGH_RECIPE_PATH = (
    _CATALOG_ROOT
    / "gp_regression"
    / "recipes"
    / "high__laplace_mhmc__window_adaptation_dense_imm__inner_laplace_hmc.json"
)


@pytest.mark.fast
def test_emit_script_laplace_high_recipe_multiphase_warmup_structure() -> None:
    """Emitted laplace HIGH script contains the two-phase warmup structure.

    The HIGH gp_regression × laplace_mhmc recipe uses a two-phase warmup:
    Phase 1 (diag IMM, maxiter=100) + Phase 2 (dense IMM, maxiter=400, maxcor=20).
    Verifies that both phases appear in the emitted script, that the
    LaplaceMarginal factories are distinct (different maxiter values), and that
    the per-model maxcor=20 is reproduced in all factory calls.
    """
    if not _LAPLACE_HIGH_RECIPE_PATH.exists():
        pytest.skip("HIGH laplace_mhmc recipe not in catalog — generate first")

    recipe = load_recipe(_LAPLACE_HIGH_RECIPE_PATH)
    script = emit_script(recipe, num_samples=10, num_chains=2)

    # Both phases must appear in the warmup section.
    assert (
        "_warmup_p1" in script
    ), "Phase 1 warmup (`_warmup_p1`) not found in emitted script"
    assert (
        "_warmup_p2" in script
    ), "Phase 2 warmup (`_warmup_p2`) not found in emitted script"

    # Phase 1 uses maxiter=100, Phase 2 uses maxiter=400 (from the recipe JSON).
    assert "maxiter=100" in script, "Expected maxiter=100 (Phase 1) in emitted script"
    assert "maxiter=400" in script, "Expected maxiter=400 (Phase 2) in emitted script"

    # maxcor=20 must be reproduced (per-model optimizer kwarg for 203-d L-BFGS convergence).
    assert "maxcor=20" in script, (
        "Expected maxcor=20 in emitted script — per-model optimizer kwarg for "
        "gp_regression (203-d Cholesky L-BFGS needs maxcor=20 to converge)."
    )

    # Phase 2 uses dense IMM (is_mass_matrix_diagonal=False).
    assert "is_mass_matrix_diagonal=False" in script, (
        "Expected `is_mass_matrix_diagonal=False` in Phase 2 warmup "
        "(Phase 2 should use dense IMM for gp_regression ridge geometry capture)."
    )

    # initial_step_size_p2 should be seeded from Phase 1 (warm-start DA).
    assert "_initial_step_size_p2" in script, (
        "Expected `_initial_step_size_p2` in emitted script "
        "(Phase 2 DA should be warm-started from Phase 1 adapted step_size)."
    )


@pytest.mark.fast
def test_emit_script_laplace_optimizer_kwargs_persisted() -> None:
    """emit_script reproduces all optimizer_kwargs stored in base_method_params.

    Regression for the optimizer_kwargs plumbing: recipes that store maxcor (or
    other minimize_lbfgs kwargs) in base_method_params must reproduce them in the
    emitted script's sampler section and LaplaceMarginal factory calls.

    Uses a synthetic gp_regression recipe with maxiter=400 + maxcor=20 to verify
    that both are in the emitted script — the same config needed for honest gp
    HIGH convergence (203-d Cholesky L-BFGS needs maxcor=20, maxiter=400).
    """
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._emit_script import emit_script

    recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={
            "num_integration_steps": 5,
            "step_size": 0.5,
            "inverse_mass_matrix": [1.0, 1.0, 1.0],
            "maxiter": 400,
            "maxcor": 20,
        },
        warmup_params={"n_warmup": 100, "num_chains": 4, "target_acceptance": 0.8},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
    )

    script = emit_script(recipe, num_samples=10, num_chains=2)

    # sampler template must use optimizer_kwargs dict (not individual maxiter arg)
    assert "_optimizer_kwargs" in script, (
        "Emitted script missing _optimizer_kwargs — laplace sampler template "
        "should use $bm_optimizer_kwargs_expr to pass all optimizer kwargs."
    )
    # Both maxiter and maxcor must appear in the sampler section
    assert "maxiter" in script, "maxiter not reproduced in emitted script"
    assert "maxcor" in script, "maxcor not reproduced in emitted script"
    # The exact values must be present (not just the key names)
    assert "400" in script, "maxiter=400 value not in emitted script"
    assert "20" in script, "maxcor=20 value not in emitted script"
    # LaplaceMarginal factory call must include the kwargs
    assert (
        "_lmf(log_joint_fn, theta_init" in script
    ), "LaplaceMarginal factory call missing from emitted script"


# ---------------------------------------------------------------------------
# emit_script override params: num_warmup, progress_bar
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_script_num_warmup_multiphase_list() -> None:
    """emit_script(recipe, num_warmup=[100, 10]) maps to per-phase n_warmup slots.

    Uses a synthetic 2-phase laplace_mhmc recipe for gp_regression.
    Phase 1 must get n_warmup=100, Phase 2 must get n_warmup=10.
    """
    from tuningfork.catalog import emit_script
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.HIGH,
        base_method_params={
            "num_integration_steps": 2,
            "step_size": 0.5,
            "inverse_mass_matrix": [
                [0.23, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.0, 0.002],
            ],
            "maxiter": 5,
        },
        warmup_params={"n_warmup": 500, "num_chains": 1, "target_acceptance": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {
                    "n_warmup": 500,
                    "target_acceptance": 0.8,
                    "num_integration_steps": 2,
                    "maxiter": 5,
                },
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {
                    "n_warmup": 200,
                    "target_acceptance": 0.8,
                    "num_integration_steps": 2,
                    "maxiter": 5,
                    "initial_step_size_from_phase1": True,
                },
            },
        ],
        warmup_inner_kernel="laplace_hmc",
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1},
        difficulty=None,
        instructions="",
        tuning_seed=42,
    )

    script = emit_script(recipe, num_samples=5, num_chains=1, num_warmup=[100, 10])

    # Phase 1 should use 100 steps; Phase 2 should use 10.
    # The run() call is multi-line so check the comment header which has n_warmup=<int>.
    # e.g. "# Phase 1: window_adaptation_diag_imm (n_warmup=100, ...)"
    assert (
        "n_warmup=100" in script
    ), "Expected 'n_warmup=100' (Phase 1 comment) in emitted script with num_warmup=[100, 10]."
    assert (
        "n_warmup=10," in script
    ), "Expected 'n_warmup=10,' (Phase 2 comment) in emitted script with num_warmup=[100, 10]."
    # The recipe's original values (500/200) should NOT appear in the warmup comments.
    assert (
        "n_warmup=500" not in script
    ), "Recipe's Phase 1 n_warmup=500 should be overridden by num_warmup=[100, 10]."
    assert (
        "n_warmup=200" not in script
    ), "Recipe's Phase 2 n_warmup=200 should be overridden by num_warmup=[100, 10]."


@pytest.mark.fast
def test_emit_script_num_warmup_wrong_list_length_raises() -> None:
    """emit_script raises ValueError when num_warmup list length != num phases.

    2-phase recipe + 3-element list → ValueError with a clear message.
    """
    from tuningfork.catalog import emit_script
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.HIGH,
        base_method_params={"step_size": 0.5, "maxiter": 5},
        warmup_params={"n_warmup": 500, "num_chains": 1, "target_acceptance": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 500, "maxiter": 5},
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {"n_warmup": 200, "maxiter": 5},
            },
        ],
        warmup_inner_kernel="laplace_hmc",
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1},
        difficulty=None,
        instructions="",
        tuning_seed=42,
    )

    with pytest.raises(ValueError, match="list length"):
        emit_script(recipe, num_warmup=[100, 10, 50])  # 3 entries for a 2-phase recipe


# ---------------------------------------------------------------------------
# VI template round-trip tests (Phase 8B.2 -- meanfield_vi + fullrank_vi)
# ---------------------------------------------------------------------------


def _make_vi_sampler_recipe(base_method_name: str):  # type: ignore[return]
    """Synthetic recipe for VI-as-inference (Track A)."""
    from tuningfork.recipes._base import Effort, Recipe

    return Recipe(
        model_name="mvn_10",
        base_method_name=base_method_name,
        warmup_name="no_warmup",
        effort=Effort.LOW,
        base_method_params={"num_optimization_steps": 200},
        warmup_params={"n_warmup": 0},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"num_chains": 1, "n_samples": 5},
        difficulty=None,
        instructions="vi template smoke",
        tuning_seed=42,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("base_method_name", ["meanfield_vi", "fullrank_vi"])
def test_emit_script_vi_sampler_executes(base_method_name: str, tmp_path: Path) -> None:
    """VI-as-inference (Track A) emitted script runs end-to-end: exit 0 + DONE.

    Phase 8B.2 e2e gate: {meanfield,fullrank}_vi + no_warmup on mvn_10 with
    200 VI opt steps, 5 samples. Exit 0 + DONE + n_divergences= verified.
    """
    recipe = _make_vi_sampler_recipe(base_method_name)
    script = emit_script(recipe, num_samples=5, num_chains=1)
    script_path = tmp_path / "test_vi_sampler.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"VI sampler ({base_method_name}) failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout
    assert "n_divergences=" in result.stdout


# ── C5 regression guard: unadjusted mclmc info-fields ────────────────────────


@pytest.mark.fast
def test_laplace_emits_lbfgs_diagnostics_and_legacy_chain_default() -> None:
    """Generated Laplace scripts retain solver stats and the 4-chain default."""
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._emit_script import _build_draws_ss_block

    laplace_stats = _build_draws_ss_block("laplace_hmc")
    assert '_draws_dict["_ss_lbfgs_iter_num"]' in laplace_stats
    assert '_draws_dict["_ss_lbfgs_hit_maxiter"]' in laplace_stats

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 10},
        warmups=[{"name": "window_adaptation_diag_imm", "params": {"n_warmup": 10}}],
        calibration_budget={"n_samples": 20},
        headline_metric=None,
        sample_quality=None,
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )
    script = emit_script(recipe, num_samples=2)
    assert "num_chains = 4" in script


@pytest.mark.fast
def test_sample_stats_matrix_preserves_safe_sampler_fields() -> None:
    """Per-step scalar/bool info fields are explicit and nested pytrees stay out."""
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.recipes._emit_script import (
        _build_draws_ss_block,
        _build_info_diagnostics_block,
    )
    from tuningfork.recipes._sample_stats import SAMPLE_STATS_CONTRACTS

    assert set(SAMPLE_STATS_CONTRACTS) == set(BASE_METHODS)

    nuts = _build_draws_ss_block("nuts")
    for field in (
        "num_trajectory_expansions",
        "is_turning",
        "num_integration_steps",
    ):
        assert f'_draws_dict["_ss_{field}"]' in nuts
    assert "trajectory_leftmost_state" not in nuts
    assert "momentum" not in nuts

    laplace = _build_draws_ss_block("laplace_hmc")
    for field in (
        "lbfgs_iter_num",
        "lbfgs_error",
        "lbfgs_converged",
        "lbfgs_hit_maxiter",
    ):
        assert f'_draws_dict["_ss_{field}"]' in laplace

    mclmc = _build_draws_ss_block("mclmc")
    for field in ("logdensity", "kinetic_change", "energy_change", "nonans"):
        assert f'_draws_dict["_ss_{field}"]' in mclmc
    assert "acceptance_rate" not in mclmc

    adjusted = _build_draws_ss_block("adjusted_mclmc_dynamic")
    for field in (
        "is_divergent",
        "energy",
        "num_integration_steps",
        "acceptance_rate",
        "is_accepted",
    ):
        assert f'_draws_dict["_ss_{field}"]' in adjusted

    for sampler in ("orbital_hmc", "elliptical_slice"):
        diagnostics = _build_info_diagnostics_block(sampler)
        assert "_infos.acceptance_rate" not in diagnostics


@pytest.mark.fast
def test_generated_artifact_fails_closed_on_reserved_position_names() -> None:
    recipe = load_recipe(_CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json")
    script = emit_script(recipe, num_samples=2)

    assert "_reserved_positions = sorted(" in script
    assert "position names use reserved generated-stat prefix _ss_" in script
