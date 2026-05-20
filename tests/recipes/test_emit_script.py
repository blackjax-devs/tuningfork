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

Phase R3.5 round-trip CI gate (locked decision D10).

Post-R3.5-MVP clarification (2026-05-17): the **inference choreography**
(warmup + sampler + inference loop) is STRICT — zero ``import tuningfork`` in
those parts of the emitted script.  The **model definition** is imported via
``from tuningfork.model import MODELS``: the canonical NumPyro code lives
upstream and is not duplicated in a template (no template-drift risk on the
largest, most-stable code surface).

The tests below enforce:

- Emitted script is syntactically valid Python.
- The two specific tuningfork imports allowed are ``tuningfork.model`` and
  ``tuningfork.model._numpyro``; no other tuningfork modules may be imported
  (no recipe schema, no calibration code, no sampler/warmup wrappers — the
  inference choreography must stand alone, auditable inline).
- The emitted script executes end-to-end and reports divergence count.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"

# Exactly the tuningfork modules an emitted script is allowed to import.
# The inference choreography (warmup/sampler/loop) must NOT depend on any
# other tuningfork module — only the model definition is sourced upstream.
_ALLOWED_TUNINGFORK_IMPORTS = frozenset(
    {"tuningfork.model", "tuningfork.model._numpyro"}
)

# Top-level packages the emitted script may import.  tuningfork imports are
# further restricted by _ALLOWED_TUNINGFORK_IMPORTS (checked separately).
_ALLOWED_TOP_LEVEL = frozenset(
    {"jax", "numpy", "numpyro", "blackjax", "arviz", "tuningfork"}
)


@pytest.mark.fast
def test_emit_script_returns_valid_python() -> None:
    """emit_script output is syntactically valid Python."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    ast.parse(script)  # raises SyntaxError on malformed output


@pytest.mark.fast
def test_emit_script_inference_choreography_has_no_tuningfork() -> None:
    """The inference choreography (warmup + sampler + loop) has zero ``import tuningfork``.

    The only tuningfork imports allowed in the whole script are
    ``tuningfork.model`` and ``tuningfork.model._numpyro`` (model definition
    + logdensity_fn builder).  Everything else — warmup call, sampler call,
    inference loop — must go directly through BlackJAX so the emitted
    choreography is auditable inline.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    tree = ast.parse(script)

    tuningfork_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tuningfork"):
                    tuningfork_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("tuningfork"):
                tuningfork_imports.append(node.module)

    disallowed = [
        name for name in tuningfork_imports if name not in _ALLOWED_TUNINGFORK_IMPORTS
    ]
    assert not disallowed, (
        "Emitted script imports tuningfork modules outside the model-import "
        f"allowlist {_ALLOWED_TUNINGFORK_IMPORTS}.\nFound: {disallowed!r}\n"
        "The inference choreography (warmup + sampler + loop) must be inline "
        "and tuningfork-free; only the model definition is sourced upstream."
    )


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "warmup_name,base_method_name",
    [
        # Conventional gradient samplers with compatible warmups (R3.5b Commit 2).
        ("window_adaptation_diag_imm", "hmc"),
        ("no_warmup", "dynamic_hmc"),
        ("no_warmup", "mhmc"),
        ("no_warmup", "dmhmc"),
        ("no_warmup", "ghmc"),
        # Random-walk / Langevin samplers (R3.5b Commit 3).
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
    """
    import jax

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.recipes._base import Recipe
    from tuningfork.warmup import WARMUPS

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS[base_method_name]
    warmup = WARMUPS[warmup_name]

    if warmup_name == "no_warmup":
        recipe = Recipe.from_default_config(posterior, base_method)
    else:
        recipe = Recipe.from_warmup_only(
            posterior,
            base_method,
            warmup,
            n_warmup=200,
            rng_key=jax.random.key(0),
        )

    script = emit_script(recipe, num_samples=200)
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


@pytest.mark.slow
def test_emit_script_executes_and_completes(tmp_path: Path) -> None:
    """Emitted script runs end-to-end via subprocess and prints DONE.

    Verifies:
    - Subprocess exit code 0 (no Python errors).
    - Postamble prints ``n_divergences=<int>`` and ``DONE``.
    - The recipe's pinned warmup + sampler params produce a runnable kernel
      against the imported NumPyro model.

    Acts as the round-trip CI gate (per D10): any drift in the warmup or
    sampler template that produces a runtime error or non-zero exit code is
    caught here.  Model definition drift is impossible by construction
    (model imported from tuningfork.model — single source).
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    # Small sample count so the test runs in ~30-60s instead of multiple minutes.
    script = emit_script(recipe, num_samples=200)
    script_path = tmp_path / "test_emitted.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert (
        result.returncode == 0
    ), f"Emitted script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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


@pytest.mark.fast
def test_emit_script_num_chains_override() -> None:
    """emit_script(recipe, num_chains=8) overrides recipe-derived value.

    Callers can force a specific chain count regardless of what the recipe
    metadata records.
    """
    gt_recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(gt_recipe_path)
    script = emit_script(recipe, num_chains=8)
    assert "num_chains = 8" in script, (
        "Expected 'num_chains = 8' in emitted script when num_chains=8 override "
        f"is passed.\nScript start:\n{script[:800]}"
    )


@pytest.mark.slow
def test_emit_script_multichain_output_shape(tmp_path: Path) -> None:
    """Emitted 4-chain script produces _samples with shape (4, num_samples, ...).

    Runs the emitted script via subprocess and checks that the printed shape
    matches the expected (4, 100, ...) protocol. Uses the eight_schools_ncp
    groundtruth recipe with num_chains=4 override and num_samples=100 so the
    test completes quickly (~60 s).

    The shape verification relies on a print statement injected into the
    emitted script after the inference loop.
    """
    gt_recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(gt_recipe_path)
    _NUM_SAMPLES = 100
    _NUM_CHAINS = 4
    script = emit_script(recipe, num_samples=_NUM_SAMPLES, num_chains=_NUM_CHAINS)
    # Append a shape-verification line that prints the first-leaf shape of _samples.
    # Use string concat (not f-string) to avoid escaping braces inside the snippet.
    shape_check = (
        "\nimport jax as _jax\n"
        "_first_leaf = _jax.tree.leaves(_samples)[0]\n"
        'print("SHAPE=" + str(_first_leaf.shape))\n'
    )
    script += shape_check
    script_path = tmp_path / "test_multichain.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Emitted multi-chain script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout
    # Shape must start with (num_chains, num_samples, ...)
    assert f"SHAPE=({_NUM_CHAINS}, {_NUM_SAMPLES}" in result.stdout, (
        f"Expected _samples shape starting with ({_NUM_CHAINS}, {_NUM_SAMPLES}, ...) "
        f"but got different output.\nstdout:\n{result.stdout}"
    )


@pytest.mark.slow
def test_emit_script_perchain_warmup_adapted_params_shape(tmp_path: Path) -> None:
    """Per-chain warmup produces _adapted_params["step_size"] shape (4,) in emitted script.

    Uses the LOW recipe for eight_schools_ncp x nuts x window_adaptation_diag_imm
    which encodes num_chains=4.  Verifies that:
    - _adapted_params["step_size"] has shape (4,)  (NOT scalar — per-chain warmup)
    - _samples has shape (4, n_samples, ...)        (multi-chain output)

    This is the regression test for the warmup vmap fix: before the fix,
    warmup templates did single-chain warmup and broadcast the state, so
    _adapted_params["step_size"] was a scalar even when num_chains=4.
    """
    low_recipe_path = (
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "low__nuts__window_adaptation_diag_imm.json"
    )
    recipe = load_recipe(low_recipe_path)
    _NUM_SAMPLES = 50
    _NUM_CHAINS = 4
    script = emit_script(recipe, num_samples=_NUM_SAMPLES)
    # Inject verification prints after the warmup and after the inference loop.
    verification = (
        "\nimport jax as _jax\n"
        "import numpy as _np\n"
        # Check _adapted_params["step_size"] shape — must be (4,) for per-chain warmup.
        "_ss = _adapted_params['step_size']\n"
        'print("STEP_SIZE_SHAPE=" + str(tuple(_np.shape(_ss))))\n'
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

    # _adapted_params["step_size"] must be shape (4,) — one value per chain.
    assert f"STEP_SIZE_SHAPE=({_NUM_CHAINS},)" in result.stdout, (
        f"Expected _adapted_params['step_size'] shape ({_NUM_CHAINS},) "
        f"but got different output (scalar would indicate single-chain warmup bug).\n"
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
def test_emit_script_warmup_algorithm_matches_runner(sampler: str, warmup: str) -> None:
    """The emitted script's warmup section references the SAME blackjax
    algorithm that `resolve_warmup_algorithm` picks.

    Catches the class of bug where templates hardcode an algorithm name
    (e.g., `blackjax.nuts`) regardless of the recipe's actual sampler.
    Discovered 2026-05-20 on `medium__mhmc__window_adaptation_dense_imm`.

    Note: laplace_* samplers are deferred to R3.5b-2 (no templates yet).
    """
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.warmup import WARMUPS
    from tuningfork.warmup._laplace_adapter import HMC_SUBSTITUTE_METHOD_NAMES

    # Skip incompatible pairs (e.g., low_rank_window_adaptation may not support
    # all sampler families).
    if not WARMUPS[warmup].is_compatible(sampler):
        pytest.skip(f"{warmup} is not compatible with {sampler}")

    # Create a synthetic Recipe in-memory with minimal required fields.
    # The emit_script function only uses: model_name, base_method_name, warmup_name,
    # effort, base_method_params, warmup_params, tuning_seed, calibration_budget,
    # and gate_evidence. Most of these can be stubbed for the syntax check.
    recipe = Recipe(
        model_name="eight_schools_ncp",
        base_method_name=sampler,
        warmup_name=warmup,
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 100, "target_acceptance_rate": 0.8},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=50)

    # Determine the expected warmup algorithm following resolve_warmup_algorithm logic.
    expected_algo = "hmc" if sampler in HMC_SUBSTITUTE_METHOD_NAMES else sampler
    warmup_section = _extract_warmup_section(script)

    assert f"blackjax.{expected_algo}" in warmup_section, (
        f"Emitted script for {sampler} × {warmup}: warmup section "
        f"should call `blackjax.{expected_algo}` but doesn't. "
        f"Section:\n{warmup_section[:500]}"
    )


@pytest.mark.slow
def test_emit_script_warmup_imm_matches_runner_mhmc_dense() -> None:
    """L2 fidelity test: emit_script's warmup produces the SAME
    (step_size, IMM) as the recipe-runner.

    Target recipe: eight_schools_ncp × mhmc × window_adaptation_dense_imm.
    User-reported bug case (2026-05-20). MHMC is in the standard
    (non-HMC-substitute) warmup path; the fix puts blackjax.mhmc in
    the emitted warmup call. Running both paths with the same seed and
    asserting numerical equivalence on adapted_params confirms the fix
    closes the protocol-fidelity gap.
    """
    import json

    import jax
    import numpy as np

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.model import MODELS
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.warmup import WARMUPS

    recipe = load_recipe(
        _CATALOG_ROOT
        / "eight_schools_ncp"
        / "recipes"
        / "medium__mhmc__window_adaptation_dense_imm.json"
    )

    # === Path A: recipe-runner's warmup (ground truth) ===
    posterior = MODELS[recipe.model_name]
    init_position, logdensity_fn, _ = build_logdensity_fn(
        jax.random.key(recipe.tuning_seed), posterior
    )

    warmup = WARMUPS[recipe.warmup_name]
    base_method = BASE_METHODS[recipe.base_method_name]
    n_warmup = recipe.warmup_params["n_warmup"]
    num_chains = recipe.warmup_params["num_chains"]
    target_acceptance_rate = recipe.warmup_params.get(
        "target_acceptance_rate", recipe.warmup_params.get("target_acceptance", 0.8)
    )

    runner_state, runner_params = warmup.runner(
        jax.random.key(recipe.tuning_seed),
        init_position,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        num_chains=num_chains,
        target_acceptance_rate=target_acceptance_rate,
    )

    # === Path B: emitted script's warmup (executed in subprocess for isolation) ===
    # Use num_samples=1 to keep the test fast (sampling does almost no work);
    # we only care about adapted_params, set BEFORE sampling.
    script = emit_script(recipe, num_samples=1)

    epilogue = """
import json
import numpy as np

# adapted_params after warmup: persist for the test process to read.
_emitted = {
    "step_size": np.asarray(_adapted_params["step_size"]).tolist(),
    "inverse_mass_matrix": np.asarray(_adapted_params["inverse_mass_matrix"]).tolist(),
}
with open("/tmp/test_emit_script_l2_emitted_params.json", "w") as _f:
    json.dump(_emitted, _f)
print("L2 EMITTED OK")
"""

    tmp_path = Path("/tmp/test_emit_script_l2_mhmc_dense.py")
    tmp_path.write_text(script + "\n\n" + epilogue)

    result = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "tuningfork",
            "--with",
            "jax",
            "--with",
            "blackjax",
            "--with",
            "numpyro",
            "python",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Emitted script crashed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    with open("/tmp/test_emit_script_l2_emitted_params.json") as f:
        emitted = json.load(f)

    # === Compare ===
    runner_step_size = np.asarray(runner_params["step_size"])
    emitted_step_size = np.asarray(emitted["step_size"])
    np.testing.assert_allclose(
        runner_step_size,
        emitted_step_size,
        rtol=1e-4,
        err_msg=f"step_size mismatch: runner={runner_step_size}, "
        f"emitted={emitted_step_size}",
    )

    runner_imm = np.asarray(runner_params["inverse_mass_matrix"])
    emitted_imm = np.asarray(emitted["inverse_mass_matrix"])
    np.testing.assert_allclose(
        runner_imm,
        emitted_imm,
        rtol=1e-4,
        err_msg=f"IMM mismatch: runner shape={runner_imm.shape}, "
        f"emitted shape={emitted_imm.shape}",
    )
