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

__all__ = ["load_recipe", "summarize_recipe", "list_recipes"]


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


def list_recipes(model_name: str) -> list[Path]:
    """List all recipe JSON files for a catalog model.

    Returns paths to {groundtruth.json, recipes/*.json} under
    tuningfork/catalog/<model_name>/.

    Parameters
    ----------
    model_name
        The model name as it appears in MODELS (e.g., "eight_schools_ncp").

    Returns
    -------
    list[Path]
        Sorted list of recipe paths. Empty list if no recipes are present.
        Raises FileNotFoundError if the model dir doesn't exist.
    """
    model_dir = _repo_root() / "tuningfork" / "catalog" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"catalog dir not found: {model_dir}")
    paths = []
    gt = model_dir / "groundtruth.json"
    if gt.exists():
        paths.append(gt)
    recipes_subdir = model_dir / "recipes"
    if recipes_subdir.exists():
        paths.extend(sorted(recipes_subdir.glob("*.json")))
    return paths


def _is_smc_recipe(p: Path) -> bool:
    """Return True if the JSON at ``p`` is an SMCRecipe (has 'smc_method_name' key)."""
    import json as _json

    try:
        d = _json.loads(p.read_text())
        return "smc_method_name" in d
    except Exception:  # noqa: BLE001
        return False


def _load_recipe_from_path(p: Path):  # type: ignore[return]
    """Dispatch to SMCRecipe.load or Recipe.load based on JSON content."""
    if _is_smc_recipe(p):
        from tuningfork.recipes._base_smc import SMCRecipe

        return SMCRecipe.load(p)
    from tuningfork.recipes._base import Recipe

    return Recipe.load(p)


def load_recipe(path: str | Path):  # type: ignore[return]
    """Load a Recipe or SMCRecipe from a path.

    Resolves relative paths against the tuningfork repo root so that a
    notebook running in any working directory can load a recipe with a
    path like::

        recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
        smc_recipe = load_recipe("tuningfork/catalog/gmm_25/recipes/smc__adaptive_tempered_smc__rwm.json")

    SMC recipes (files with a ``smc_method_name`` key) are automatically
    dispatched to ``SMCRecipe.load``; MCMC recipes go to ``Recipe.load``.

    Parameters
    ----------
    path
        Path to the recipe JSON. Absolute paths are used as-is. Relative
        paths are resolved against the tuningfork repo root (detected via
        the installed package location).

    Returns
    -------
    Recipe | SMCRecipe
        The loaded recipe dataclass (type depends on file content).

    Raises
    ------
    FileNotFoundError
        If the recipe file cannot be found after trying absolute and
        repo-root-relative resolution.
    """
    recipe_path = Path(path)
    if recipe_path.is_absolute():
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe file not found at {recipe_path}. "
                "Check the path and ensure the file exists."
            )
        return _load_recipe_from_path(recipe_path)

    # Relative path: try as-is first, then against repo root
    if recipe_path.exists():
        return _load_recipe_from_path(recipe_path)

    candidate = _repo_root() / recipe_path
    if candidate.exists():
        return _load_recipe_from_path(candidate)

    raise FileNotFoundError(
        f"Recipe file not found: tried {recipe_path!r} (relative to cwd) "
        f"and {candidate!r} (relative to repo root). "
        "Recipes live under "
        "tuningfork/catalog/<model>/recipes/<effort>__<sampler>__<warmup>.json "
        "or smc__<smc_method>__<inner>.json for SMC recipes."
    )


def _summarize_smc_recipe(recipe) -> pd.DataFrame:  # type: ignore[return]
    """Return a summary DataFrame for an SMCRecipe (SMC-specific fields)."""
    import pandas as pd

    auto_gate = recipe.gate_evidence.get("auto", {})
    override = recipe.gate_evidence.get("override", {})
    verdict = auto_gate.get("verdict", "NOT_RUN")
    override_decision = override.get("decision", "")
    b = recipe.calibration_budget
    sp = recipe.smc_params

    rows = [
        ("model", recipe.model_name),
        ("type", "SMC"),
        ("smc_method", recipe.smc_method_name),
        ("inner_kernel", recipe.inner_method_name),
        ("num_particles", str(recipe.num_particles)),
        ("target_ess", str(sp.get("target_ess", "N/A"))),
        ("num_mcmc_steps", str(sp.get("num_mcmc_steps", "N/A"))),
        ("num_integration_steps", str(sp.get("num_integration_steps", "N/A"))),
        ("parameter_update_strategy", recipe.parameter_update_strategy),
        ("n_smc_steps", str(b.get("n_smc_steps", "N/A"))),
        ("lambda_final", str(b.get("lambda_final", "N/A"))),
        ("stored gate verdict (auto)", verdict),
        ("override decision", override_decision if override_decision else "none"),
        (
            "particle_ess",
            (
                f"{auto_gate['particle_ess']:.1f}"
                if auto_gate.get("particle_ess") is not None
                else "N/A"
            ),
        ),
        (
            "max_abs_mean_z",
            (
                f"{auto_gate['max_abs_mean_z']:.3f}"
                if auto_gate.get("max_abs_mean_z") is not None
                else "N/A"
            ),
        ),
        (
            "mode_coverage_fraction",
            (
                f"{auto_gate['mode_coverage_fraction']:.3f}"
                if auto_gate.get("mode_coverage_fraction") is not None
                else "N/A"
            ),
        ),
        (
            "headline_metric",
            (
                f"{recipe.headline_metric:.5f}"
                if recipe.headline_metric is not None
                else "N/A"
            ),
        ),
        ("seed", str(recipe.seed)),
        ("tuningfork_version", recipe.tuningfork_version),
        ("blackjax_version", recipe.blackjax_version),
        ("jax_version", recipe.jax_version),
        ("timestamp_utc", recipe.timestamp_utc),
    ]
    return pd.DataFrame(rows, columns=["Property", "Value"])


def summarize_recipe(recipe) -> pd.DataFrame:  # type: ignore[return]
    """Return a summary DataFrame for a Recipe or SMCRecipe.

    Returns a 2-column ``(Property, Value)`` DataFrame suitable for
    inline Jupyter rendering. The ``inverse_mass_matrix`` field is
    excluded (too verbose). Rows cover the most decision-relevant fields
    a statistician needs when verifying a recipe.

    Accepts both MCMC ``Recipe`` and ``SMCRecipe`` objects — the returned
    DataFrame adapts its columns accordingly (SMC recipes show SMC-specific
    fields such as ``particle_ess``, ``n_smc_steps``, ``target_ess``
    instead of the MCMC warmup / chain budget fields).

    Parameters
    ----------
    recipe
        A loaded Recipe or SMCRecipe object.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``Property`` and ``Value``. Auto-renders as HTML
        in Jupyter notebooks.

    Notes
    -----
    For MCMC Recipes, fields included:

    - model, effort, sampler, warmup
    - num_chains, n_warmup, n_samples (sample-budget fields; see below)
    - stored gate verdict, R̂_max, min_bulk_ESS, n_divergences
    - tuning_seed
    - tuningfork / blackjax / jax versions
    - timestamp_utc

    ``num_chains`` is read from ``warmup_params`` first, then
    ``calibration_budget`` (legacy groundtruth recipes record it only in
    ``calibration_budget``).  ``n_warmup`` follows the same fallback pattern.
    ``n_samples`` is read from ``calibration_budget`` first, then
    ``warmup_params``.  All three show ``"N/A"`` when the field is absent
    (e.g., legacy groundtruth recipes that pre-date the protocol).
    """
    # Dispatch to SMC-specific summary if this is an SMCRecipe.
    if hasattr(recipe, "smc_method_name"):
        return _summarize_smc_recipe(recipe)

    import pandas as pd

    auto_gate = recipe.gate_evidence.get("auto", {})
    rhat = auto_gate.get("rhat_max")
    min_ess = auto_gate.get("min_bulk_ess")
    n_diverg = auto_gate.get("n_divergences")
    verdict = auto_gate.get("verdict", "NOT_RUN")

    _num_chains = recipe.warmup_params.get(
        "num_chains", recipe.calibration_budget.get("num_chains")
    )
    _n_warmup = recipe.warmup_params.get(
        "n_warmup", recipe.calibration_budget.get("n_warmup")
    )
    _n_samples = recipe.calibration_budget.get(
        "n_samples", recipe.warmup_params.get("n_samples")
    )

    # Schema extension: warmup_inner_kernel shown only when explicitly set (non-None).
    # Omitting it for legacy / default-None recipes keeps the summary compact;
    # it only adds noise for the standard case where the implicit substitute-
    # family logic picks the kernel.
    _warmup_inner_kernel = getattr(recipe, "warmup_inner_kernel", None)

    rows = [
        ("model", recipe.model_name),
        ("effort", recipe.effort.value),
        ("sampler", recipe.base_method_name),
        ("warmup", recipe.warmup_name),
    ]
    if _warmup_inner_kernel is not None:
        rows.append(("warmup_inner_kernel", _warmup_inner_kernel))
    rows.extend(
        [
            (
                "num_chains",
                str(int(_num_chains)) if _num_chains is not None else "N/A",
            ),
            ("n_warmup", str(int(_n_warmup)) if _n_warmup is not None else "N/A"),
            ("n_samples", str(int(_n_samples)) if _n_samples is not None else "N/A"),
            ("stored gate verdict", verdict),
            ("R_hat_max", f"{rhat:.4f}" if rhat is not None else "N/A"),
            ("min_bulk_ESS", f"{min_ess:.1f}" if min_ess is not None else "N/A"),
            (
                "n_divergences",
                str(int(n_diverg)) if n_diverg is not None else "N/A",
            ),
            ("tuning_seed", str(recipe.tuning_seed)),
            ("tuningfork_version", recipe.tuningfork_version),
            ("blackjax_version", recipe.blackjax_version),
            ("jax_version", recipe.jax_version),
            ("timestamp_utc", recipe.timestamp_utc),
        ]
    )

    return pd.DataFrame(rows, columns=["Property", "Value"])
