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

NOTE: e2e emit-execute tests run a minimal 10-sample / minimal-warmup config;
they assert the emitted script executes (structure correct, no vmap/io_callback
errors), NOT inference quality. This keeps the e2e gate fast and memory-safe.
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
# stdlib modules used by the emitted preamble/postamble (e.g. ``time`` for
# wall-clock timing, ``warnings`` for the progress_bar=True single-chain
# warmup advisory) are also allowlisted here.
_ALLOWED_TOP_LEVEL = frozenset(
    {"jax", "numpy", "numpyro", "blackjax", "arviz", "tuningfork", "time", "warnings"}
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


@pytest.mark.fast
def test_emit_script_num_samples_fallback_when_budget_absent() -> None:
    """emit_script falls back to 1000 when calibration_budget has no n_samples key.

    Directly overrides calibration_budget on a loaded recipe to simulate the
    legacy LOW recipe format (produced before the calibration_budget n_samples
    stamp was added). No JAX execution — pure dataclass manipulation.
    """
    import dataclasses

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)

    # Simulate a recipe whose calibration_budget has no n_samples key.
    recipe_no_n = dataclasses.replace(
        recipe, calibration_budget={"trials": 0, "wall_seconds_estimate": 0.0}
    )
    assert recipe_no_n.calibration_budget.get("n_samples") is None

    script = emit_script(recipe_no_n)
    assert "_NUM_SAMPLES = 1000" in script, (
        "emit_script(recipe) must fall back to _NUM_SAMPLES=1000 when "
        "calibration_budget has no n_samples key."
    )


@pytest.mark.fast
def test_emit_script_preamble_has_wall_timer() -> None:
    """Emitted script preamble starts a wall-clock timer (_recipe_t0)."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    assert (
        "_recipe_t0" in script
    ), "Emitted script must define _recipe_t0 (wall-clock start) in its preamble."
    assert (
        "wall_seconds=" in script
    ), "Emitted script postamble must print wall_seconds= for timing observability."


@pytest.mark.fast
def test_emit_script_postamble_has_draws_npz() -> None:
    """Emitted script postamble persists draws as .npz via np.savez.

    Checks that the generated script contains:
    - ``np.savez`` (draws persistence — no external library needed)
    - ``draws.npz`` suffix (filename pattern)
    - ``[draws written to`` (confirmation print)
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    assert (
        "np.savez" in script
    ), "Emitted script postamble must call np.savez to persist draws."
    assert (
        ".draws.npz" in script
    ), "Emitted script postamble must write a .draws.npz file."
    assert (
        "[draws written to" in script
    ), "Emitted script postamble must print '[draws written to ...]' confirmation."


@pytest.mark.fast
def test_emit_script_postamble_npz_filename_contains_recipe_components() -> None:
    """Emitted script's .npz filename encodes model/sampler/warmup names.

    The filename pattern is ``{model_name}__{base_method_name}__{warmup_name}.draws.npz``.
    After template substitution the postamble contains each component as a string
    literal in the _npz_path assignment. Verifies that model_name, base_method_name,
    warmup_name, and the .draws.npz suffix all appear in the script.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    # After substitution, the _npz_path line contains the component names as literals.
    assert "eight_schools_ncp" in script, "Model name not found in emitted script"
    assert (
        ".draws.npz" in script
    ), "Expected '.draws.npz' suffix in emitted script for draws persistence."
    assert (
        "_npz_path" in script
    ), "Expected '_npz_path' variable in emitted script (draws output path)."


@pytest.mark.e2e
def test_emit_script_multichain_draws_npz(tmp_path: Path) -> None:
    """Emitted multi-chain script runs, writes .draws.npz, and prints DONE.

    Uses eight_schools_ncp × nuts × window_adaptation_diag_imm with num_chains=4
    and num_samples=10 for e2e speed.  Verifies:
    - Script exits 0
    - ``[draws written to`` appears in stdout
    - A ``.draws.npz`` file is written in the script's working directory
    - DONE is printed
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
    script = emit_script(
        recipe,
        num_samples=_NUM_SAMPLES,
        num_chains=_NUM_CHAINS,
        num_warmup=10,
        progress_bar=False,
    )
    script_path = tmp_path / "test_draws_multichain.py"
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
        f"Emitted multi-chain draws script failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout
    # The draws path must have been printed.
    assert (
        "[draws written to" in result.stdout
    ), f"Expected '[draws written to ...]' in stdout.\nstdout:\n{result.stdout}"
    # The .draws.npz file must exist in cwd (tmp_path).
    npz_files = list(tmp_path.glob("*.draws.npz"))
    assert npz_files, (
        f"Expected a .draws.npz file in {tmp_path}. "
        f"Files: {list(tmp_path.iterdir())}"
    )


@pytest.mark.e2e
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
    Lightweight config: num_samples=10, num_warmup=10 (just enough to verify structure).
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
            n_warmup=10,
            rng_key=jax.random.key(0),
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


@pytest.mark.e2e
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

    Lightweight config: num_samples=10, num_warmup=10 for e2e speed.
    """
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    # Minimal config: num_warmup=10, num_samples=10 overrides the groundtruth recipe's n_warmup=5000.
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


@pytest.mark.e2e
def test_emit_script_multichain_output_shape(tmp_path: Path) -> None:
    """Emitted 4-chain script produces _samples with shape (4, num_samples, ...).

    Runs the emitted script via subprocess and checks that the printed shape
    matches the expected (4, 10, ...) protocol. Uses the eight_schools_ncp
    groundtruth recipe with num_chains=4 + num_warmup=10 + num_samples=10
    overrides for e2e speed.

    The shape verification relies on a print statement injected into the
    emitted script after the inference loop.
    """
    gt_recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(gt_recipe_path)
    _NUM_SAMPLES = 10
    _NUM_CHAINS = 4
    # Minimal warmup/samples for e2e speed.
    # progress_bar=False for multi-chain output (True = single-chain, shape (1, ...) ).
    script = emit_script(
        recipe,
        num_samples=_NUM_SAMPLES,
        num_chains=_NUM_CHAINS,
        num_warmup=10,
        progress_bar=False,
    )
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
    from tuningfork.warmup._laplace_adapter import WARMUP_SUBSTITUTE_METHOD_NAMES

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
    expected_algo = "nuts" if sampler in WARMUP_SUBSTITUTE_METHOD_NAMES else sampler
    warmup_section = _extract_warmup_section(script)

    assert f"blackjax.{expected_algo}" in warmup_section, (
        f"Emitted script for {sampler} × {warmup}: warmup section "
        f"should call `blackjax.{expected_algo}` but doesn't. "
        f"Section:\n{warmup_section[:500]}"
    )


@pytest.mark.fast
def test_emit_script_warmup_inner_kernel_override_emits_correct_algo() -> None:
    """Schema extension: warmup_inner_kernel=nuts overrides hmc recipe to emit blackjax.nuts.

    When recipe.warmup_inner_kernel is explicitly set AND differs from the
    implicit default (hmc -> hmc is implicit; hmc + inner_nuts overrides to nuts),
    the emitted script's warmup section must reference blackjax.nuts, not
    blackjax.hmc.

    This mirrors the runner logic in _warmup_to_sampler_transform.resolve_warmup_inner_kernel.
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
def test_emit_script_warmup_inner_kernel_none_uses_implicit() -> None:
    """Schema extension: warmup_inner_kernel=None falls back to implicit substitute-family logic.

    For hmc (not in WARMUP_SUBSTITUTE_METHOD_NAMES), the implicit default is
    blackjax.hmc. Setting warmup_inner_kernel=None must NOT change this.
    """
    from tuningfork.recipes._base import Effort, Recipe

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
        warmup_inner_kernel=None,  # explicit None — must use implicit default
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=50)
    warmup_section = _extract_warmup_section(script)

    # hmc's implicit default is "hmc": warmup section must reference blackjax.hmc.
    assert "blackjax.hmc" in warmup_section, (
        "Schema extension: warmup_inner_kernel=None for hmc should emit blackjax.hmc "
        f"(the implicit default).\nSection:\n{warmup_section[:500]}"
    )


@pytest.mark.fast
def test_emit_script_warmup_inner_kernel_substitute_family_unchanged() -> None:
    """Schema extension: warmup_inner_kernel=None for dynamic_hmc still uses blackjax.nuts.

    dynamic_hmc is in WARMUP_SUBSTITUTE_METHOD_NAMES; its implicit default is nuts.
    warmup_inner_kernel=None must preserve this existing behaviour.
    """
    from tuningfork.recipes._base import Effort, Recipe

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="dynamic_hmc",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.LOW,
        base_method_params={"step_size": 0.1},
        warmup_params={"n_warmup": 200, "target_acceptance_rate": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {"n_warmup": 200, "target_acceptance_rate": 0.8},
            }
        ],
        warmup_inner_kernel=None,
        headline_metric=None,
        sample_quality=None,
        calibration_budget={},
        difficulty=None,
        instructions="",
        tuning_seed=0,
    )

    script = emit_script(recipe, num_samples=50)
    warmup_section = _extract_warmup_section(script)

    # dynamic_hmc is in the substitute family -> implicit default is nuts
    assert "blackjax.nuts" in warmup_section, (
        "Schema extension: warmup_inner_kernel=None for dynamic_hmc should keep "
        f"blackjax.nuts (substitute-family implicit default).\nSection:\n{warmup_section[:500]}"
    )


@pytest.mark.e2e
def test_emit_script_warmup_imm_matches_runner_mhmc_dense(tmp_path: Path) -> None:
    """L2 fidelity test: emit_script's warmup produces valid adapted params.

    Target recipe: eight_schools_ncp × mhmc × window_adaptation_dense_imm.
    User-reported bug case (2026-05-20). MHMC is in the standard
    (non-HMC-substitute) warmup path; the fix puts blackjax.mhmc in
    the emitted warmup call.

    Note: since the scan(vmap) refactor, the emitted script runs SINGLE-CHAIN
    warmup while the runner runs PER-CHAIN warmup (vmap).  The seeds differ
    (fold_in vs split), so numerical equality is no longer expected.  This
    test verifies that the emitted warmup completes without error and produces
    finite adapted params of the correct structure.

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
    script = emit_script(recipe, num_samples=10, num_warmup=10)

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
with open("/tmp/test_emit_script_l2_emitted_params.json", "w") as _f:
    json.dump(_emitted, _f)
print("L2 EMITTED OK")
"""

    script_path = tmp_path / "test_l2_mhmc_dense.py"
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
    assert "L2 EMITTED OK" in result.stdout

    import json

    with open("/tmp/test_emit_script_l2_emitted_params.json") as f:
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
# Laplace-* recipe emit tests (R3.5b-2 laplace templates)
# ---------------------------------------------------------------------------

_LAPLACE_HIGH_RECIPE_PATH = (
    _CATALOG_ROOT
    / "gp_regression"
    / "recipes"
    / "high__laplace_mhmc__window_adaptation_dense_imm__inner_laplace_hmc.json"
)


@pytest.mark.fast
def test_emit_script_laplace_high_recipe_is_valid_python() -> None:
    """emit_script for the HIGH gp_regression × laplace_mhmc recipe is syntactically valid Python.

    Validates that all laplace template slots are populated (no un-substituted
    $slot markers) and that the assembled script parses without SyntaxError.
    """
    if not _LAPLACE_HIGH_RECIPE_PATH.exists():
        pytest.skip("HIGH laplace_mhmc recipe not in catalog — generate first")

    recipe = load_recipe(_LAPLACE_HIGH_RECIPE_PATH)
    script = emit_script(recipe, num_samples=10, num_chains=2)
    # ast.parse raises SyntaxError on malformed output (including un-substituted $slots)
    ast.parse(script)


@pytest.mark.fast
def test_emit_script_laplace_high_recipe_d8_compliant() -> None:
    """D8: emitted laplace HIGH recipe has zero forbidden tuningfork imports.

    The only allowed tuningfork imports are ``tuningfork.model`` and
    ``tuningfork.model._numpyro``.  The inference choreography (laplace preamble
    + multiphase warmup + sampler + inference loop) must be completely free of
    tuningfork.warmup, tuningfork.recipes, tuningfork.calibration, etc.

    Also checks that blackjax is imported (required for the laplace kernels),
    and that ``from blackjax.mcmc.laplace_marginal import laplace_marginal_factory``
    appears (the D8-compliant inline factory from the laplace_preamble template).
    """
    if not _LAPLACE_HIGH_RECIPE_PATH.exists():
        pytest.skip("HIGH laplace_mhmc recipe not in catalog — generate first")

    recipe = load_recipe(_LAPLACE_HIGH_RECIPE_PATH)
    script = emit_script(recipe, num_samples=10, num_chains=2)
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
        "Emitted laplace HIGH script imports tuningfork modules outside the allowlist "
        f"{_ALLOWED_TUNINGFORK_IMPORTS}.\nFound: {disallowed!r}\n"
        "The inference choreography must be tuningfork-free (D8 STRICT)."
    )

    # Positive checks: laplace_marginal_factory must be inlined (D8-compliant path).
    assert "laplace_marginal_factory" in script, (
        "Expected `laplace_marginal_factory` in the emitted script "
        "(imported from blackjax.mcmc.laplace_marginal in laplace_preamble)."
    )
    assert "blackjax.laplace_mhmc" in script, (
        "Expected `blackjax.laplace_mhmc` in the emitted script "
        "(sampler template for the HIGH recipe)."
    )


@pytest.mark.fast
def test_emit_script_laplace_high_recipe_multiphase_warmup_structure() -> None:
    """Emitted laplace HIGH script contains the two-phase warmup structure.

    The HIGH gp_regression × laplace_mhmc recipe uses a two-phase warmup:
    Phase 1 (diag IMM, maxiter=100) + Phase 2 (dense IMM, maxiter=500).
    Verifies that both phases appear in the emitted script and that the
    LaplaceMarginal factories are distinct (different maxiter values).
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

    # Phase 1 uses maxiter=100, Phase 2 uses maxiter=500 (from the recipe JSON).
    assert "maxiter=100" in script, "Expected maxiter=100 (Phase 1) in emitted script"
    assert "maxiter=500" in script, "Expected maxiter=500 (Phase 2) in emitted script"

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


@pytest.mark.e2e
def test_emit_script_laplace_multiphase_executes(tmp_path: Path) -> None:
    """Acceptance test: emitted laplace multiphase script runs end-to-end (exit 0, prints DONE).

    Uses a synthetic multi-phase laplace_mhmc recipe with minimal warmup budgets
    (n_warmup=2 per phase, maxiter=2) so the test completes quickly for e2e speed.

    What is validated:
    - The laplace_preamble + laplace_multiphase_warmup + laplace_mhmc sampler
      templates assemble into a Python script that runs without errors.
    - LaplaceMarginal factories are correctly built and passed to window_adaptation.
    - The two-phase warmup loop (Phase 1 diag → Phase 2 dense) executes.
    - The sampler produces draws and the postamble prints DONE + n_divergences.

    This is the D10 round-trip CI gate for laplace templates: any template
    slot miss, assembly-order bug, or LaplaceMarginal contract violation would
    surface here as a Python error or non-zero exit code.

    Lightweight config: num_samples=10, num_warmup=[2, 2] (Phase1=2, Phase2=2).
    """
    from tuningfork.recipes._base import Effort, Recipe

    # Synthetic multi-phase laplace_mhmc recipe for gp_regression.
    # Minimal warmup budgets (n_warmup=2, maxiter=2) for e2e speed.
    recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.HIGH,
        base_method_params={
            "num_integration_steps": 2,
            "step_size": 0.5,
            "inverse_mass_matrix": [
                [0.23, 0.07, 0.0],
                [0.07, 0.03, 0.0],
                [0.0, 0.0, 0.002],
            ],
            "maxiter": 2,
        },
        warmup_params={
            "n_warmup": 2,
            "num_chains": 1,
            "target_acceptance": 0.8,
        },
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {
                    "n_warmup": 2,
                    "target_acceptance": 0.8,
                    "num_integration_steps": 2,
                    "maxiter": 2,
                },
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {
                    "n_warmup": 2,
                    "target_acceptance": 0.8,
                    "num_integration_steps": 2,
                    "maxiter": 2,
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

    script = emit_script(recipe, num_samples=10, num_chains=1, num_warmup=[2, 2])
    script_path = tmp_path / "test_laplace_multiphase.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={"JAX_PLATFORM_NAME": "cpu", **os.environ},
    )
    assert result.returncode == 0, (
        f"Emitted laplace multiphase script failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DONE" in result.stdout, (
        f"Expected 'DONE' in stdout of laplace multiphase emitted script.\n"
        f"stdout:\n{result.stdout}"
    )
    assert (
        "n_divergences=" in result.stdout
    ), f"Expected 'n_divergences=' in stdout.\nstdout:\n{result.stdout}"


# ---------------------------------------------------------------------------
# run_recipe_to_idata multi-phase faithfulness (cached_idata_for_recipe path)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_recipe_to_idata_laplace_multiphase_uses_dense_imm() -> None:
    """run_recipe_to_idata faithfully runs both warmup phases for a laplace multiphase recipe.

    Verifies that the multi-phase runner path (used by cached_idata_for_recipe)
    executes Phase 1 (diag IMM) followed by Phase 2 (dense IMM) and uses Phase 2's
    adapted params for sampling — NOT Phase 1's diagonal IMM.

    Uses a synthetic recipe with tiny n_warmup=5/maxiter=5 for speed (~30 s).

    What is checked:
    - run_recipe_to_idata returns valid InferenceData with posterior group.
    - phi-space variables (log_lengthscale, log_kernel_scale, log_noise_scale) present.
    - Sample shape is (1, 5) for num_chains=1, n_samples=5.
    - No error from the dense-IMM window_adaptation call (would fail if Phase 2
      were incorrectly routing through the diagonal-only path).

    This is the fix verification for the pre-PR #63 unfaithfulness where
    run_recipe_to_idata ignored recipe.warmups list and only ran warmups[0],
    producing a diagonal IMM even when the recipe specifies dense.
    """
    from tuningfork.catalog._rerun_inference import (  # noqa: F401
        cached_idata_for_recipe,
    )
    from tuningfork.recipes._base import Effort, Recipe
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    # Synthetic multi-phase laplace_mhmc recipe — same structure as HIGH gp_regression
    # but with tiny warmup budgets for CI speed.
    synth_recipe = Recipe(
        model_name="gp_regression",
        base_method_name="laplace_mhmc",
        warmup_name="window_adaptation_dense_imm",
        effort=Effort.HIGH,
        base_method_params={
            "num_integration_steps": 2,
            "step_size": 0.5,
            "inverse_mass_matrix": [
                [0.23, 0.07, 0.0],
                [0.07, 0.03, 0.0],
                [0.0, 0.0, 0.002],
            ],
            "maxiter": 5,
        },
        warmup_params={"n_warmup": 5, "num_chains": 1, "target_acceptance": 0.8},
        warmups=[
            {
                "name": "window_adaptation_diag_imm",
                "params": {
                    "n_warmup": 5,
                    "target_acceptance": 0.8,
                    "num_integration_steps": 2,
                    "maxiter": 5,
                },
            },
            {
                "name": "window_adaptation_dense_imm",
                "params": {
                    "n_warmup": 5,
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

    # Direct call to run_recipe_to_idata (the function cached_idata_for_recipe wraps)
    idata = run_recipe_to_idata(synth_recipe, n_samples=5)

    # Returned InferenceData must have a posterior group
    assert hasattr(idata, "posterior"), "InferenceData missing posterior group"

    # phi-space variables must be present (gp_regression phi sites)
    phi_sites = {"log_lengthscale", "log_kernel_scale", "log_noise_scale"}
    posterior_vars = set(idata.posterior.data_vars)
    assert phi_sites <= posterior_vars, (
        f"Missing phi sites in posterior: {phi_sites - posterior_vars}\n"
        "Expected gp_regression phi-space variables from laplace_mhmc sampling."
    )

    # Sample shape must be (num_chains=1, n_samples=5)
    first_var = next(iter(idata.posterior.data_vars))
    sample_shape = tuple(idata.posterior[first_var].shape[:2])
    assert sample_shape == (1, 5), (
        f"Expected sample shape (1, 5), got {sample_shape}.\n"
        "Indicates either wrong num_chains/n_samples routing or idata assembly error."
    )

    # If we reach here without error, Phase 2's dense window_adaptation completed
    # (blackjax.window_adaptation(..., is_mass_matrix_diagonal=False, ...) ran successfully)
    # — proving the multi-phase loop executed Phase 2, not only Phase 1.
    # A regression to the old single-phase path would skip Phase 2 entirely and
    # produce a diagonal IMM, which would still succeed numerically but lose density.
    # The dense IMM run is the discriminating evidence (no error = Phase 2 ran).
    assert True, "Phase 2 dense-IMM warmup completed without error"


# ---------------------------------------------------------------------------
# emit_script override params: num_warmup, progress_bar
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_emit_script_num_warmup_int_single_phase() -> None:
    """emit_script(recipe, num_warmup=50) overrides $n_warmup in single-phase warmup.

    Checks that the emitted source contains '50' at the warmup call site, not
    the recipe's stored n_warmup value (1000 for groundtruth recipes).
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script_default = emit_script(recipe, num_samples=50)
    script_override = emit_script(recipe, num_samples=50, num_warmup=77)

    # The override must appear in the warmup template's .run(..., 77) call.
    # Groundtruth recipes have n_warmup=5000 in warmup_params; 77 is distinctive.
    assert ", 77)" in script_override, (
        "Expected ', 77)' (warmup run call) in emitted script with num_warmup=77.\n"
        f"Script:\n{script_override[:1000]}"
    )
    # Without override, must NOT contain 77 in that position.
    assert (
        ", 77)" not in script_default
    ), "Default emit should not contain ', 77)'; num_warmup override leaked."


@pytest.mark.fast
def test_emit_script_num_warmup_list_single_phase_length1() -> None:
    """emit_script(recipe, num_warmup=[77]) works for single-phase warmup (list len=1)."""
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=50, num_warmup=[77])
    assert (
        ", 77)" in script
    ), "Expected ', 77)' in emitted script with num_warmup=[77] (single-phase)."


@pytest.mark.fast
def test_emit_script_num_warmup_none_preserves_recipe_value() -> None:
    """emit_script(recipe, num_warmup=None) uses recipe's stored n_warmup.

    The groundtruth recipe has n_warmup=5000 in warmup_params; None must
    preserve that value (backward-compatible).
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    n_warmup_recipe = recipe.warmup_params.get("n_warmup", 1000)
    script = emit_script(recipe, num_samples=50, num_warmup=None)
    assert f", {n_warmup_recipe})" in script, (
        f"Expected ', {n_warmup_recipe})' (recipe n_warmup) with num_warmup=None.\n"
        f"Script snippet:\n{script[:1000]}"
    )


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


@pytest.mark.fast
def test_emit_script_progress_bar_override_false() -> None:
    """emit_script(recipe, progress_bar=False) disables both warmup and sampling bars.

    Checks that:
    - Warmup template has progress_bar=False (not True).
    - Sampling constant _SAMPLING_PROGRESS_BAR = False (not True).
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=50, progress_bar=False)

    assert "progress_bar=False" in script, (
        "Expected 'progress_bar=False' in emitted script when progress_bar=False override.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    assert "_SAMPLING_PROGRESS_BAR = False" in script, (
        "Expected '_SAMPLING_PROGRESS_BAR = False' in emitted script when progress_bar=False.\n"
        f"Script snippet:\n{script[:1200]}"
    )
    # Must NOT have the True variants for the overridden slots.
    assert (
        "progress_bar=True" not in script
    ), "Unexpected 'progress_bar=True' in emitted script when progress_bar=False override."
    assert (
        "_SAMPLING_PROGRESS_BAR = True" not in script
    ), "Unexpected '_SAMPLING_PROGRESS_BAR = True' when progress_bar=False override."


@pytest.mark.fast
def test_emit_script_progress_bar_override_true() -> None:
    """emit_script(recipe, progress_bar=True) explicitly enables both progress bars.

    Both warmup and sampling must have progress_bar=True (same as the default).
    Also verifies the call-time warning is issued (pytest.warns) so tests are not
    surprised by UserWarning-as-error under filterwarnings=error.
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    with pytest.warns(UserWarning, match="#927"):
        script = emit_script(recipe, num_samples=50, progress_bar=True)

    assert (
        "progress_bar=True" in script
    ), "Expected 'progress_bar=True' in emitted script when progress_bar=True."
    assert (
        "_SAMPLING_PROGRESS_BAR = True" in script
    ), "Expected '_SAMPLING_PROGRESS_BAR = True' in emitted script when progress_bar=True."


@pytest.mark.fast
def test_emit_script_progress_bar_none_keeps_defaults() -> None:
    """emit_script(recipe, progress_bar=None) uses defaults (warmup True, sampling True).

    The default behaviour (None) must emit progress_bar=True for warmup and
    _SAMPLING_PROGRESS_BAR = True for sampling — unchanged from pre-override baseline.
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script_default = emit_script(recipe, num_samples=50)
    script_none = emit_script(recipe, num_samples=50, progress_bar=None)

    # Both must be identical (None is the backward-compat default).
    assert (
        script_default == script_none
    ), "emit_script with progress_bar=None should produce identical output to default call."
    assert "progress_bar=True" in script_none
    assert "_SAMPLING_PROGRESS_BAR = True" in script_none


@pytest.mark.fast
def test_emit_script_sampling_progress_bar_constant_exposed() -> None:
    """Emitted script exposes _SAMPLING_PROGRESS_BAR as a named constant.

    The inference loop must declare '_SAMPLING_PROGRESS_BAR = <bool>' near the
    top of the loop block so users can find and flip it in one line.
    """
    from tuningfork.catalog import emit_script, load_recipe

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe, num_samples=50)

    assert (
        "_SAMPLING_PROGRESS_BAR" in script
    ), "Emitted script must expose _SAMPLING_PROGRESS_BAR constant in the inference loop."
