# flake8: noqa: F811
# mypy: disable-error-code="no-redef"
"""Tuningfork Catalog Explorer — interactive marimo notebook.

Launch via:
    uv run --group notebook marimo edit tuningfork/catalog/notebooks/catalog_explorer.py

Pick a model from the dropdown -> see available recipes -> pick a recipe ->
inspect summary + plots. Reactive cells re-execute on dropdown change.

Linter notes: every cell in a marimo notebook is `def _()` (single
underscore — marimo's canonical convention), so mypy/flake8 see `_`
redefined across cells. Disable both at file scope — this is a marimo
notebook idiom, not a code issue.

Naming gotcha: marimo treats names with a LEADING underscore (e.g.,
`_my_var`, `import inspect as _inspect`) as CELL-LOCAL — not visible to
downstream cells. Use plain names (e.g., `pyinspect`) for any module
alias or variable that other cells need to access.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # Note: do NOT prefix module aliases with `_` — marimo treats underscore-
    # prefixed names as cell-local, so they're not visible to downstream cells.
    import inspect as pyinspect

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt

    from tuningfork.catalog import (
        list_recipes,
        load_idata,
        load_recipe,
        plot_recipe_diagnostics,
        summarize_recipe,
    )
    from tuningfork.model import MODELS

    return (
        MODELS,
        az,
        list_recipes,
        load_idata,
        load_recipe,
        mo,
        plot_recipe_diagnostics,
        pyinspect,
        summarize_recipe,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Tuningfork Catalog Explorer

    Interactive notebook for browsing the 14-model BlackJAX/tuningfork catalog.

    1. Pick a **model** below — see its source code and available recipes.
    2. Pick a **recipe** — see its summary metadata + cert verdict.
    3. Diagnostic plots auto-render: trace + pair on headline parameters
       (hyperpriors), forest on the rest.

    See [catalog README](../README.md) for the broader API + the
    [headline_params decision doc](https://github.com/blackjax-devs/claude-config/blob/main/project/worklog/decisions/2026-05-18-headline-params-per-model.md)
    for what counts as "headline" per model.
    """
    )
    return


@app.cell
def _(MODELS, mo):
    model_name = mo.ui.dropdown(
        options=sorted(MODELS.keys()),
        value="eight_schools_ncp",
        label="Model",
    )
    model_name
    return (model_name,)


@app.cell(hide_code=True)
def _(MODELS, mo, model_name, pyinspect):
    posterior_entry = MODELS[model_name.value]
    model_module = pyinspect.getmodule(posterior_entry.__class__)
    if model_module is not None:
        source = pyinspect.getsource(model_module)
    else:
        source = "(source not available)"
    mo.md(f"### Model source: `{model_name.value}`\n\n```python\n{source}\n```")
    return (posterior_entry,)


@app.cell
def _(list_recipes, mo, model_name):
    recipe_paths = list_recipes(model_name.value)
    recipe_options = {p.name: str(p) for p in recipe_paths}
    # NB: marimo's dropdown `value=` expects one of the OPTION KEYS (the
    # label-side of the dict), not the value. dropdown.value returns the
    # mapped value (the str path) which is what we pass to load_recipe below.
    recipe_dropdown = mo.ui.dropdown(
        options=recipe_options,
        value=next(iter(recipe_options.keys())) if recipe_options else None,
        label="Recipe",
    )
    recipe_dropdown
    return (recipe_dropdown,)


@app.cell
def _(load_recipe, recipe_dropdown, summarize_recipe):
    if recipe_dropdown.value is None:
        recipe = None
        summary_df = None
    else:
        recipe = load_recipe(recipe_dropdown.value)
        summary_df = summarize_recipe(recipe)
    summary_df
    return (recipe,)


@app.cell
def _(load_idata, mo, recipe):
    if recipe is not None and recipe.effort == "groundtruth":
        try:
            idata = load_idata(recipe)
            mo.md(f"**Posterior sites**: `{list(idata['posterior'].data_vars)}`")
        except FileNotFoundError:
            idata = None
            mo.md(
                "*Cache miss — only GROUNDTRUTH recipes have populated draws caches today.*"
            )
    else:
        idata = None
        mo.md("*Pick a GROUNDTRUTH recipe to see diagnostic plots.*")
    return (idata,)


@app.cell
def _(idata, plot_recipe_diagnostics, posterior_entry):
    # Compute the 3 plots in one pass; render in separate cells below.
    if idata is not None:
        try:
            figs = plot_recipe_diagnostics(idata, posterior_entry, n_forest_top=20)
            plot_error = None
        except Exception as e:
            figs = {"trace": None, "pair": None, "forest": None}
            plot_error = f"plot_recipe_diagnostics failed: {type(e).__name__}: {e}"
    else:
        figs = {"trace": None, "pair": None, "forest": None}
        plot_error = None
    return figs, plot_error


@app.cell(hide_code=True)
def _(figs, mo, plot_error):
    mo.vstack(
        [
            mo.md("### Trace + KDE plot (headline params)"),
            (
                mo.md(f"*Plot error:* `{plot_error}`")
                if plot_error
                else (figs["trace"] if figs["trace"] is not None else mo.md(""))
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(figs, mo):
    mo.vstack(
        [
            mo.md("### Pair plot (headline params)"),
            figs["pair"] if figs["pair"] is not None else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(figs, mo):
    if figs["forest"] is not None:
        mo.vstack(
            [
                mo.md("### Forest plot (bulk params, capped at 20)"),
                figs["forest"],
            ]
        )
    else:
        mo.md("*No bulk params for this model — headline covers everything.*")
    return


@app.cell
def _(az, idata, mo, posterior_entry):
    if idata is not None:
        var_names = (
            list(posterior_entry.headline_params)
            if posterior_entry.headline_params is not None
            else None
        )
        summary_table = az.summary(idata, var_names=var_names)
        result_summary = mo.vstack(
            [mo.md("### ArviZ summary table (headline params)"), summary_table]
        )
    else:
        result_summary = mo.md("")
    result_summary
    return


if __name__ == "__main__":
    app.run()
