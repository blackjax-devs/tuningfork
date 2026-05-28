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
        cached_idata_for_recipe,
        format_timing_context,
        list_recipes,
        load_idata,
        load_recipe,
        plot_recipe_diagnostics,
        summarize_recipe,
    )
    from tuningfork.model import MODELS
    from tuningfork.recipes._base import Effort
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    return (
        Effort,
        MODELS,
        az,
        cached_idata_for_recipe,
        format_timing_context,
        list_recipes,
        load_idata,
        load_recipe,
        mo,
        plot_recipe_diagnostics,
        pyinspect,
        run_recipe_to_idata,
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


@app.cell(hide_code=True)
def _(format_timing_context, mo, recipe):
    """Display timing metadata + machine info from calibration_budget when available."""
    if recipe is None:
        timing_panel = mo.md("")
    else:
        budget = recipe.calibration_budget or {}
        warmup_wall = budget.get("warmup_wall_seconds")
        sampling_wall = budget.get("sampling_wall_seconds")
        spd = budget.get("sampling_seconds_per_draw")
        wall_est = budget.get("wall_seconds_estimate")
        minfo = budget.get("machine_info") or {}

        if warmup_wall is not None and sampling_wall is not None:
            # Show measured timing info with context column
            cpu = minfo.get("cpu_model", "unknown")
            jax_ver = minfo.get("jax_version", "?")
            x64 = minfo.get("jax_x64_enabled", False)
            spd_str = f"{spd * 1000:.3f} ms/draw" if spd is not None else "N/A"

            # Format the context column
            ctx = format_timing_context(recipe)

            timing_panel = mo.callout(
                mo.md(
                    f"**Measured timings** (emit run):\n\n"
                    f"| | | |\n|---|---|---|\n"
                    f"| Warmup wall | `{warmup_wall:.1f} s` | {ctx['warmup_wall']} |\n"
                    f"| Sampling wall | `{sampling_wall:.1f} s` | {ctx['sampling_wall']} |\n"
                    f"| Per-draw | `{spd_str}` | {ctx['per_draw']} |\n"
                    f"| Total wall est. | `{wall_est:.1f} s` | {ctx['total_wall']} |\n"
                    f"| Machine | `{cpu}` / JAX `{jax_ver}` / x64={x64} | {ctx['machine']} |"
                ),
                kind="info",
            )
        elif wall_est is not None:
            timing_panel = mo.callout(
                mo.md(
                    f"**Estimated wall**: `{wall_est:.1f} s` "
                    f"(model-derived; no measured timing stamp)"
                ),
                kind="neutral",
            )
        else:
            timing_panel = mo.md("")

    timing_panel
    return


@app.cell(hide_code=True)
def _(Effort, mo, recipe):
    """Sampling controls — 3 knobs for the run mode."""
    if recipe is None or recipe.effort in (Effort.GROUNDTRUTH, Effort.FAILED):
        # No controls needed for GROUNDTRUTH (load from cache) or FAILED.
        use_cached_switch = None
        skip_warmup_toggle = None
        n_samples_slider = None
        controls_panel = mo.md("")
    else:
        use_cached_switch = mo.ui.switch(
            value=True,
            label="Use cache (skip re-run)",
        )
        skip_warmup_toggle = mo.ui.switch(
            value=False,
            label="Skip warmup (instant sampling)",
        )
        n_samples_slider = mo.ui.slider(
            start=100,
            stop=2000,
            step=100,
            value=1000,
            label="n_samples",
        )
        controls_panel = mo.vstack(
            [
                mo.md("#### Sampling controls"),
                mo.hstack(
                    [use_cached_switch, skip_warmup_toggle, n_samples_slider],
                    gap=2,
                ),
                mo.md(
                    "_n_samples and skip-warmup take effect when cache is off "
                    "or skip-warmup is on. Cache hit serves the recipe's default "
                    "n_samples regardless of slider._"
                ),
            ]
        )

    controls_panel
    return n_samples_slider, skip_warmup_toggle, use_cached_switch


@app.cell(hide_code=True)
def _(
    Effort,
    mo,
    n_samples_slider,
    recipe,
    skip_warmup_toggle,
    use_cached_switch,
):
    """Dynamic wall estimate + Run button for non-cache paths.

    Naming convention: intermediate locals are underscore-prefixed (marimo
    cell-local — avoids `budget`/`spd` name collisions with the timing-display
    cell). Only ``run_button`` is exported (consumed by the sampling cell);
    it's predeclared ``None`` so the return statement is safe in all
    branches and downstream gates handle ``None`` explicitly.
    """
    run_button = None
    _estimate_and_button = mo.md("")
    if recipe is None or recipe.effort in (Effort.GROUNDTRUTH, Effort.FAILED):
        # No estimate box or button needed — either load from cache or FAILED.
        pass
    else:
        _use_cache = use_cached_switch.value if use_cached_switch is not None else True
        if _use_cache:
            # Cache path: no estimate box or button needed.
            pass
        else:
            # Non-cache path: show dynamic estimate + Run button.
            _n = n_samples_slider.value if n_samples_slider is not None else 1000
            _skip = (
                skip_warmup_toggle.value if skip_warmup_toggle is not None else False
            )

            _budget = recipe.calibration_budget or {}
            _spd = _budget.get("sampling_seconds_per_draw", 0.0)
            _ww = _budget.get("warmup_wall_seconds", 0.0)
            # Chain count: recipe.warmup_params["num_chains"] is the canonical
            # source (per _recipe_runner.py:1317 — defaults to RECIPE_NUM_CHAINS=4
            # when unset). Recipe has no top-level `num_chains` attribute.
            _c = int((recipe.warmup_params or {}).get("num_chains", 4))

            # Compute dynamic estimate. Add a flat 20 s overhead for JAX
            # compile + executor setup that the per-draw `_spd` doesn't
            # capture (observed: small runs estimate ~1 s but actually take
            # ~20 s on cold start). Additive (not a floor) is the right model
            # — the setup cost is incurred on every cold-start run regardless
            # of size, so it should add to compute-dominated runs too.
            _OVERHEAD_S = 20.0  # JAX compile + executor setup (empirical)
            _est_samp = _spd * _n * _c  # per-draw is per-draw-per-chain
            _est_warm = 0.0 if _skip else _ww
            _est_tot = _est_samp + _est_warm + _OVERHEAD_S
            _est_min = _est_tot / 60.0

            run_button = mo.ui.run_button(label="Run sampling")

            _estimate_panel = mo.callout(
                mo.md(
                    f"This will sample for at least **{_est_min:.1f} minutes**. "
                    f"Click **Run** when ready."
                ),
                kind="warn",
            )

            _estimate_and_button = mo.vstack([_estimate_panel, run_button])

    _estimate_and_button
    return (run_button,)


@app.cell
def _(
    Effort,
    cached_idata_for_recipe,
    load_idata,
    mo,
    n_samples_slider,
    recipe,
    run_button,
    run_recipe_to_idata,
    skip_warmup_toggle,
    use_cached_switch,
):
    idata = None
    if recipe is None:
        mo.md("*Pick a recipe to see diagnostic plots.*")
    elif recipe.effort == Effort.GROUNDTRUTH:
        # Groundtruth: load from cache (already persisted)
        try:
            idata = load_idata(recipe)
            mo.md(f"**Posterior sites**: `{list(idata['posterior'].data_vars)}`")
        except FileNotFoundError:
            mo.md("*Cache miss for GROUNDTRUTH recipe (git lfs pull may be needed).*")
    elif recipe.effort == Effort.FAILED:
        # FAILED: no valid config to run.
        mo.md(
            "*FAILED recipes have no gate-passing configuration to sample. "
            "See the recipe notes for attempted configurations and diagnostics.*"
        )
    else:
        # Non-GROUNDTRUTH, non-FAILED: sample with one of 3 modes based on controls.
        _skip = skip_warmup_toggle.value if skip_warmup_toggle is not None else False
        _use_cache = use_cached_switch.value if use_cached_switch is not None else True
        _n = n_samples_slider.value if n_samples_slider is not None else 1000

        # Build data-swap advisory when skip_warmup is active.
        _skip_warn = None
        if _skip:
            _skip_warn = mo.callout(
                mo.md(
                    "⚠ **Skip-warmup mode**: Chains start from GT reference means "
                    "(`reference/summary.json`).  If the model's dataset has been "
                    "modified since the reference was computed, these means may be "
                    "stale and sampling could be unreliable.  "
                    "Check `catalog/<model>/reference/metadata.json` for the cert "
                    "timestamp."
                ),
                kind="warn",
            )

        try:
            if _use_cache:
                # Cache path: cached_idata_for_recipe() now raises FileNotFoundError
                # on cache miss (per the 2026-05-28 API change — no more silent
                # re-sample). Catch + surface a UI message directing the user
                # to toggle off cache and click Run.
                try:
                    idata = cached_idata_for_recipe(recipe)
                except FileNotFoundError:
                    idata = None
                    mo.md(
                        "⚠ **Cache miss** for this recipe. Toggle off "
                        "**Use cache** and click **Run** to generate it."
                    )
            elif run_button is not None and run_button.value:
                # Non-cache path with Run button click: sample now.
                # (``run_button`` is None when the estimate cell short-circuited
                # — recipe None / GROUNDTRUTH / FAILED / cache mode — so the
                # ``is not None`` guard keeps this branch safe in all states.)
                if _skip:
                    # Skip-warmup: bypass adaptation, use stored step_size/IMM,
                    # stationary init from GT-means. Always re-runs (no cache path).
                    idata = run_recipe_to_idata(recipe, skip_warmup=True, n_samples=_n)
                else:
                    # Force resample: re-run warmup + sampling with recipe tuning_seed
                    # and the requested n_samples.
                    _seed = recipe.tuning_seed if recipe.tuning_seed != 0 else 20260517
                    idata = run_recipe_to_idata(
                        recipe,
                        force_resample_config={"seed": _seed, "n_samples": _n},
                    )

            if idata is not None:
                mo.vstack(
                    [
                        _skip_warn or mo.md(""),
                        mo.md(
                            f"**Posterior sites**: `{list(idata['posterior'].data_vars)}`"
                        ),
                    ]
                )
        except Exception as e:
            mo.md(f"*Failed to sample recipe: {type(e).__name__}: {e}*")
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
            (
                figs["pair"]
                if figs["pair"] is not None
                else mo.md(
                    "*Skipped — the headline-params set has > 6 scalar coords "
                    "(`headline_params=None` for high-dim single-block models per "
                    "the 2026-05-18 decision); pair grid would exceed "
                    '`rcParams["plot.max_subplots"]=40`. Use the forest plot '
                    "below for the full posterior.*"
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(figs, mo):
    # marimo renders the LAST expression of the cell. Assigning to `forest_output`
    # then exposing it as the final statement ensures both branches render.
    if figs["forest"] is not None:
        forest_output = mo.vstack(
            [
                mo.md("### Forest plot (bulk params, capped at 20)"),
                figs["forest"],
            ]
        )
    else:
        forest_output = mo.md(
            "*No bulk params for this model — headline covers everything.*"
        )
    forest_output
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
