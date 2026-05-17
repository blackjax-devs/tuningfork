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
