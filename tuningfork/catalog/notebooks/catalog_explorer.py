# flake8: noqa: F811
# mypy: disable-error-code="no-redef"
"""Tuningfork Catalog Explorer — interactive marimo notebook.

Launch via:
    uv run --group notebook marimo edit tuningfork/catalog/notebooks/catalog_explorer.py

Pick a model from the dropdown -> see available recipes -> pick a recipe ->
inspect summary + plots. Reactive cells re-execute on dropdown change.

Linter notes: every cell in a marimo notebook is `def __()`, so mypy sees
`def __` redefined and flake8 sees F811. Disable both at file scope —
this is a marimo notebook idiom, not a code issue.
"""

import marimo

__generated_with = "0.10"
app = marimo.App(width="medium")


@app.cell
def __():
    import inspect as _inspect

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
        _inspect,
        az,
        list_recipes,
        load_idata,
        load_recipe,
        mo,
        plot_recipe_diagnostics,
        plt,
        summarize_recipe,
    )


@app.cell
def __(mo):
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
def __(MODELS, mo):
    model_name = mo.ui.dropdown(
        options=sorted(MODELS.keys()),
        value="eight_schools_ncp",
        label="Model",
    )
    model_name
    return (model_name,)


@app.cell
def __(MODELS, _inspect, mo, model_name):
    posterior_entry = MODELS[model_name.value]
    model_module = _inspect.getmodule(posterior_entry.__class__)
    if model_module is not None:
        source = _inspect.getsource(model_module)
    else:
        source = "(source not available)"
    mo.md(f"### Model source: `{model_name.value}`\n\n```python\n{source}\n```")
    return model_module, posterior_entry, source


@app.cell
def __(list_recipes, mo, model_name):
    recipe_paths = list_recipes(model_name.value)
    recipe_options = {p.name: str(p) for p in recipe_paths}
    recipe_dropdown = mo.ui.dropdown(
        options=recipe_options,
        value=next(iter(recipe_options.values())) if recipe_options else None,
        label="Recipe",
    )
    recipe_dropdown
    return recipe_dropdown, recipe_options, recipe_paths


@app.cell
def __(load_recipe, recipe_dropdown, summarize_recipe):
    if recipe_dropdown.value is None:
        recipe = None
        summary_df = None
    else:
        recipe = load_recipe(recipe_dropdown.value)
        summary_df = summarize_recipe(recipe)
    summary_df
    return recipe, summary_df


@app.cell
def __(load_idata, mo, recipe):
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
def __(idata, mo, plot_recipe_diagnostics, posterior_entry):
    if idata is not None:
        figs = plot_recipe_diagnostics(idata, posterior_entry, n_forest_top=20)
        result = mo.vstack(
            [
                mo.md("### Trace plot (headline params)"),
                figs["trace"],
                mo.md("### Pair plot (headline params, divergences highlighted)"),
                figs["pair"],
                (
                    mo.md("### Forest plot (bulk params, capped at 20)")
                    if figs["forest"] is not None
                    else mo.md("")
                ),
                figs["forest"] if figs["forest"] is not None else mo.md(""),
            ]
        )
    else:
        result = mo.md("")
    result
    return (result,)


@app.cell
def __(az, idata, mo, posterior_entry):
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
    return (result_summary,)


if __name__ == "__main__":
    app.run()
