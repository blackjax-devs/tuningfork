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
"""Recipe inspection helpers (load_recipe, summarize_recipe).

Provides ``load_recipe`` and ``summarize_recipe`` — the two functions a
statistician calls at the top of a diagnostics notebook to orient themselves
before looking at any plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from tuningfork.recipes._base import Recipe

__all__ = ["load_recipe", "summarize_recipe"]


def _repo_root() -> Path:
    """Return the tuningfork repo root (directory containing pyproject.toml).

    Resolution order:
    1. The installed tuningfork package path (works for editable installs).
    2. cwd / cwd.parent as fallbacks.
    """
    import tuningfork as _tf

    candidate = Path(_tf.__file__).parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    # Fallback: walk up from cwd
    for root in [Path.cwd(), Path.cwd().parent]:
        if (root / "pyproject.toml").exists():
            return root
    return candidate


def load_recipe(path: str | Path) -> Recipe:
    """Load a Recipe from a path.

    Resolves relative paths against the tuningfork repo root so that a
    notebook running in any working directory can load a recipe with a
    path like::

        recipe = load_recipe("tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json")

    Parameters
    ----------
    path
        Path to the recipe JSON. Absolute paths are used as-is. Relative
        paths are resolved against the tuningfork repo root (detected via
        the installed package location).

    Returns
    -------
    Recipe
        The loaded Recipe dataclass.

    Raises
    ------
    FileNotFoundError
        If the recipe file cannot be found after trying absolute and
        repo-root-relative resolution.
    """
    from tuningfork.recipes._base import Recipe

    recipe_path = Path(path)
    if recipe_path.is_absolute():
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe file not found at {recipe_path}. "
                "Check the path and ensure the file exists."
            )
        return Recipe.load(recipe_path)

    # Relative path: try as-is first, then against repo root
    if recipe_path.exists():
        return Recipe.load(recipe_path)

    candidate = _repo_root() / recipe_path
    if candidate.exists():
        return Recipe.load(candidate)

    raise FileNotFoundError(
        f"Recipe file not found: tried {recipe_path!r} (relative to cwd) "
        f"and {candidate!r} (relative to repo root). "
        "Starter recipes live under "
        "tuningfork/inference/recipes/starter/<model>/<effort>__<sampler>__<warmup>.json"
    )


def summarize_recipe(recipe: Recipe) -> pd.DataFrame:
    """Return a summary DataFrame for a Recipe.

    Returns a 2-column ``(Property, Value)`` DataFrame suitable for
    inline Jupyter rendering. The ``inverse_mass_matrix`` field is
    excluded (too verbose). Rows cover the most decision-relevant fields
    a statistician needs when verifying a recipe.

    Parameters
    ----------
    recipe
        A loaded Recipe object.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``Property`` and ``Value``. Auto-renders as HTML
        in Jupyter notebooks.

    Notes
    -----
    Fields included:

    - model, effort, sampler, warmup, dim (from gate_evidence)
    - stored gate verdict, R̂_max, min_bulk_ESS, n_divergences
    - tuning_seed
    - tuningfork / blackjax / jax versions
    - timestamp_utc
    """
    import pandas as pd

    auto_gate = recipe.gate_evidence.get("auto", {})
    rhat = auto_gate.get("rhat_max")
    min_ess = auto_gate.get("min_bulk_ess")
    n_diverg = auto_gate.get("n_divergences")
    verdict = auto_gate.get("verdict", "NOT_RUN")

    rows = [
        ("model", recipe.model_name),
        ("effort", recipe.effort.value),
        ("sampler", recipe.base_method_name),
        ("warmup", recipe.warmup_name),
        ("stored gate verdict", verdict),
        ("R_hat_max", f"{rhat:.4f}" if rhat is not None else "N/A"),
        ("min_bulk_ESS", f"{min_ess:.1f}" if min_ess is not None else "N/A"),
        ("n_divergences", str(int(n_diverg)) if n_diverg is not None else "N/A"),
        ("tuning_seed", str(recipe.tuning_seed)),
        ("tuningfork_version", recipe.tuningfork_version),
        ("blackjax_version", recipe.blackjax_version),
        ("jax_version", recipe.jax_version),
        ("timestamp_utc", recipe.timestamp_utc),
    ]

    return pd.DataFrame(rows, columns=["Property", "Value"])
