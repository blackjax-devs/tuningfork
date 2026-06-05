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
"""Regression tests for the ``tuningfork leaderboard`` CLI subcommand.

Guards against the SMC crash introduced before this PR: ``_cmd_leaderboard``
previously called ``Recipe.load()`` unconditionally on every recipe JSON file,
which raises ``ValueError`` for SMC recipes (they lack the 'effort' key).
After the fix, SMC recipes are silently skipped from the MCMC ranking and a
transparent note is printed.

Models with SMC recipes as of the current catalog:
  - gmm_25
  - logistic_synthetic
  - neals_funnel
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "tuningfork" / "catalog"


def _has_smc_recipes(model_name: str) -> bool:
    """Return True if the model has at least one smc__*.json recipe on disk."""
    recipes_dir = _CATALOG_ROOT / model_name / "recipes"
    if not recipes_dir.exists():
        return False
    return any(recipes_dir.glob("smc__*.json"))


def _models_with_smc() -> list[str]:
    """Collect all models that have at least one SMC recipe in the catalog."""
    if not _CATALOG_ROOT.exists():
        return []
    return [
        d.name
        for d in sorted(_CATALOG_ROOT.iterdir())
        if d.is_dir() and _has_smc_recipes(d.name)
    ]


# ---------------------------------------------------------------------------
# Core regression: leaderboard does NOT crash on models with SMC recipes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    _models_with_smc()
    or pytest.param(
        "gmm_25",
        marks=pytest.mark.skip(reason="No SMC catalog recipes found in this checkout"),
    ),
    ids=lambda m: m,
)
def test_leaderboard_does_not_crash_on_model_with_smc(model_name: str) -> None:
    """``_cmd_leaderboard`` must not raise on models that have SMC recipes.

    Regression test for the AttributeError / ValueError raised before this fix
    when SMCRecipe objects were mistakenly passed to the MCMC recipe filter
    (``r.effort.value``) and sort key (``recipe.base_method_name``).
    """
    import argparse

    from tuningfork.cli import _cmd_leaderboard

    args = argparse.Namespace(
        model=model_name,
        effort=None,
        format="markdown",
    )
    # Must not raise; return code 0 expected (some MCMC recipes present).
    rc = _cmd_leaderboard(args)
    assert rc == 0, f"_cmd_leaderboard returned {rc} for model {model_name!r}"


def test_leaderboard_smc_note_printed_in_markdown(capsys) -> None:
    """Markdown output includes the SMC note when SMC recipes are present.

    The note format is: "(N SMC recipe(s) present — not ranked; SMC uses a
    separate execution model)"
    """
    import argparse

    from tuningfork.cli import _cmd_leaderboard

    # Use any model known to have SMC recipes; fall back gracefully if absent.
    models_with_smc = _models_with_smc()
    if not models_with_smc:
        pytest.skip("No SMC catalog recipes found in this checkout")

    model_name = models_with_smc[0]
    args = argparse.Namespace(
        model=model_name,
        effort=None,
        format="markdown",
    )
    _cmd_leaderboard(args)
    out = capsys.readouterr().out
    assert "SMC" in out, (
        f"Expected SMC note in leaderboard output for {model_name!r}.\n" f"Got:\n{out}"
    )
    assert "not ranked" in out, (
        f"Expected 'not ranked' note for SMC in output for {model_name!r}.\n"
        f"Got:\n{out}"
    )


def test_leaderboard_smc_note_absent_without_smc(capsys) -> None:
    """Markdown output has NO SMC note when the model has no SMC recipes."""
    import argparse

    from tuningfork.cli import _cmd_leaderboard

    # Find a model with MCMC recipes but no SMC recipes
    models_with_smc = set(_models_with_smc())
    no_smc_candidates = [
        d.name
        for d in sorted(_CATALOG_ROOT.iterdir())
        if d.is_dir()
        and (d / "recipes").exists()
        and any((d / "recipes").glob("low__*.json"))
        and d.name not in models_with_smc
    ]
    if not no_smc_candidates:
        pytest.skip("No model without SMC recipes found in this checkout")

    model_name = no_smc_candidates[0]
    args = argparse.Namespace(
        model=model_name,
        effort=None,
        format="markdown",
    )
    _cmd_leaderboard(args)
    out = capsys.readouterr().out
    assert "not ranked" not in out, (
        f"Unexpected SMC note in leaderboard output for {model_name!r} "
        f"(a model with no SMC recipes).\nGot:\n{out}"
    )


def test_leaderboard_mcmc_recipes_still_listed_alongside_smc(capsys) -> None:
    """SMC exclusion must not also drop MCMC recipes from the listing."""
    import argparse

    from tuningfork.cli import _cmd_leaderboard

    models_with_smc = _models_with_smc()
    if not models_with_smc:
        pytest.skip("No SMC catalog recipes found in this checkout")

    # Find a model that also has MCMC recipes
    for model_name in models_with_smc:
        recipes_dir = _CATALOG_ROOT / model_name / "recipes"
        if recipes_dir.exists() and any(recipes_dir.glob("low__*.json")):
            break
    else:
        pytest.skip("No model with both SMC and MCMC recipes found")

    args = argparse.Namespace(
        model=model_name,
        effort=None,
        format="markdown",
    )
    _cmd_leaderboard(args)
    out = capsys.readouterr().out
    # The markdown table header row must be present (MCMC recipes rendered)
    assert "| effort |" in out, (
        f"MCMC leaderboard table header not found in output for {model_name!r}. "
        f"SMC-exclusion may have accidentally cleared MCMC recipes too.\nGot:\n{out}"
    )


def test_leaderboard_json_format_mcmc_only_on_smc_model() -> None:
    """JSON output for a model with SMC recipes must contain only MCMC entries."""
    import argparse
    import json

    from tuningfork.cli import _cmd_leaderboard

    models_with_smc = _models_with_smc()
    if not models_with_smc:
        pytest.skip("No SMC catalog recipes found in this checkout")

    model_name = models_with_smc[0]
    args = argparse.Namespace(
        model=model_name,
        effort=None,
        format="json",
    )

    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = _cmd_leaderboard(args)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    entries = json.loads(output)
    # All entries must have 'effort' and 'base_method_name' — SMCRecipe has neither.
    for entry in entries:
        assert "effort" in entry, f"Entry missing 'effort': {entry}"
        assert "base_method_name" in entry, f"Entry missing 'base_method_name': {entry}"
        # SMC entries would have 'smc_method_name' instead — assert absent
        assert (
            "smc_method_name" not in entry
        ), f"SMC entry leaked into JSON output: {entry}"
