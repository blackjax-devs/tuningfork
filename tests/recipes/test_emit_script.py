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
"""Tests for emit_script -- recipe to standalone Python script.

Phase R3.5 round-trip CI gate (locked decision D10): the slow tests run the
emitted script end-to-end via subprocess and cross-check the output against
the tuningfork-import control path.  Drift between the inlined model body in
the template and the canonical numpyro definition in
``tuningfork/model/eight_schools.py`` would show up as a divergence-count
mismatch in ``test_emit_script_cross_check_against_tuningfork_import``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tuningfork.catalog import emit_script, load_recipe

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


@pytest.mark.fast
def test_emit_script_returns_valid_python() -> None:
    """emit_script output is syntactically valid Python."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    # Must parse cleanly
    ast.parse(script)


@pytest.mark.fast
def test_emit_script_has_no_tuningfork_import() -> None:
    """Per locked decision D8 (STRICT), emitted script has zero `import tuningfork`."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    # Check both import forms
    assert "import tuningfork" not in script, (
        "STRICT decision D8: emitted script must not import tuningfork. "
        "Found 'import tuningfork' in emitted code."
    )
    assert "from tuningfork" not in script, (
        "STRICT decision D8: emitted script must not import from tuningfork. "
        "Found 'from tuningfork' in emitted code."
    )


@pytest.mark.fast
def test_emit_script_imports_only_allowed_modules() -> None:
    """Emitted script imports only the allowlisted dependencies."""
    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)
    script = emit_script(recipe)
    tree = ast.parse(script)
    allowed = {
        "jax",
        "jax.numpy",
        "numpy",
        "numpyro",
        "numpyro.distributions",
        "numpyro.infer.util",
        "blackjax",
        "arviz",
    }
    allowed_tops = {a.split(".")[0] for a in allowed}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in allowed_tops, f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            top = node.module.split(".")[0]
            assert top in allowed_tops, f"Disallowed from-import: {node.module}"


@pytest.mark.slow
def test_emit_script_executes_and_completes(tmp_path: Path) -> None:
    """Emitted script runs end-to-end via subprocess and prints DONE."""
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
        env={"JAX_PLATFORM_NAME": "cpu", **__import__("os").environ},
    )
    assert (
        result.returncode == 0
    ), f"Emitted script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "DONE" in result.stdout
    # Verify divergence count is reported
    assert "n_divergences=" in result.stdout


@pytest.mark.slow
def test_emit_script_cross_check_against_tuningfork_import(tmp_path: Path) -> None:
    """Per locked decision D8: emitted-script output matches tuningfork-import output.

    Run BOTH paths (strict-emit and from-tuningfork-import) in subprocesses at
    the same seed; compare the final n_divergences count.  The cross-check fails
    if the inlined model body in the template drifts from the canonical numpyro
    definition in ``tuningfork/model/eight_schools.py``.

    This is the round-trip CI gate (per locked decision D10): drift between
    templates and the canonical model code is caught by EXECUTION, not by
    syntactic match.
    """
    import os

    recipe_path = _CATALOG_ROOT / "eight_schools_ncp" / "groundtruth.json"
    recipe = load_recipe(recipe_path)

    # Path 1: STRICT emit (no tuningfork import).
    emitted = emit_script(recipe, num_samples=200)
    p1 = tmp_path / "strict_emit.py"
    p1.write_text(emitted)
    subprocess_env = {"JAX_PLATFORM_NAME": "cpu", **os.environ}
    r1 = subprocess.run(
        [sys.executable, str(p1)],
        capture_output=True,
        text=True,
        timeout=180,
        env=subprocess_env,
    )
    assert r1.returncode == 0, f"Strict-emit run failed:\n{r1.stderr}"

    # Path 2: equivalent but uses `from tuningfork.model import MODELS`.
    # Build a tiny control script that does the same thing minus the inlined model.
    control = f"""import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import blackjax
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn

posterior = MODELS["eight_schools_ncp"]
init_position, logdensity_fn, _ = build_logdensity_fn(
    jax.random.key({recipe.tuning_seed}), posterior
)

warmup = blackjax.window_adaptation(
    blackjax.nuts, logdensity_fn,
    target_acceptance_rate={recipe.warmup_params.get("target_acceptance_rate", recipe.warmup_params.get("target_acceptance", 0.8))},
    progress_bar=False,
)
(state_post_warmup, adapted_params), _ = warmup.run(
    jax.random.key({recipe.tuning_seed}), init_position,
    {recipe.warmup_params.get("n_warmup", 1000)},
)
kernel = blackjax.nuts(
    logdensity_fn,
    step_size=adapted_params["step_size"],
    inverse_mass_matrix=adapted_params["inverse_mass_matrix"],
    max_num_doublings={recipe.base_method_params.get("max_num_doublings", 10)},
).step

@jax.jit
def one_step(state, rng_key):
    state, info = kernel(rng_key, state)
    return state, (state, info)

keys = jax.random.split(jax.random.key({recipe.tuning_seed + 1}), 200)
_, (samples, infos) = jax.lax.scan(one_step, state_post_warmup, keys)
n_div = int(jnp.sum(infos.is_divergent))
print(f"[control] n_divergences={{n_div}}")
print("DONE")
"""
    p2 = tmp_path / "tuningfork_import_control.py"
    p2.write_text(control)
    r2 = subprocess.run(
        [sys.executable, str(p2)],
        capture_output=True,
        text=True,
        timeout=180,
        env=subprocess_env,
    )
    assert r2.returncode == 0, f"Control run failed:\n{r2.stderr}"

    # Parse divergence counts from both stdouts.
    def parse_n_div(stdout: str) -> int:
        for line in stdout.splitlines():
            if "n_divergences=" in line:
                return int(line.split("n_divergences=")[1].split()[0])
        raise AssertionError(f"Couldn't find n_divergences in stdout: {stdout!r}")

    n_div_strict = parse_n_div(r1.stdout)
    n_div_control = parse_n_div(r2.stdout)
    # At the same seed, both paths should produce IDENTICAL n_divergences.
    # Drift between template and canonical model would show up here.
    assert n_div_strict == n_div_control, (
        f"Cross-check FAILED: strict-emit n_div={n_div_strict}, "
        f"tuningfork-import n_div={n_div_control}. "
        "Template drift suspected -- re-verify the eight_schools_ncp.py.tmpl "
        "model body matches tuningfork/model/eight_schools.py."
    )
