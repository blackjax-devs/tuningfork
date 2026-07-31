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
        regenerate_idata,
        summarize_recipe,
    )
    from tuningfork.model import MODELS
    from tuningfork.recipes._base import Effort

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
        regenerate_idata,
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

    See [catalog README](../README.md) for the broader API and the
    `headline_params` / `headline_coords` fields on each model's `Posterior` dataclass
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
    _is_smc = recipe is not None and hasattr(recipe, "smc_method_name")
    if recipe is None or _is_smc:
        # SMC recipes display timing via summarize_recipe; no warmup/sampling wall here.
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
    """Sampling controls — 3 knobs for the run mode (+ regenerate slider for FAIL)."""
    _is_smc = recipe is not None and hasattr(recipe, "smc_method_name")
    if recipe is None or _is_smc or recipe.effort == Effort.GROUNDTRUTH:
        # No sampling controls for SMC recipes (SMC has no MCMC warmup/sampling loop).
        # No controls needed for GROUNDTRUTH (load from cache).
        use_cached_switch = None
        skip_warmup_toggle = None
        n_samples_slider = None
        controls_panel = mo.md("")
    elif recipe.effort == Effort.FAILED:
        # FAILED: only offer the n_samples slider for the on-demand re-run.
        use_cached_switch = None
        skip_warmup_toggle = None
        n_samples_slider = mo.ui.slider(
            start=100,
            stop=2000,
            step=100,
            value=400,
            label="n_samples (diagnostic preview)",
        )
        controls_panel = mo.vstack(
            [
                mo.md("#### Regenerate controls (FAIL recipe)"),
                n_samples_slider,
                mo.md(
                    "_Lower n_samples (200–400) for a quick preview of the "
                    "failure mode. Full warmup is always run._"
                ),
            ]
        )
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
    cell). ``run_button`` (non-cache "Run sampling") and ``populate_btn``
    (cache-miss "Run + populate cache") are exported and consumed by the
    sampling cell. Both are predeclared ``None`` so the return is safe in
    all branches, and the sampling cell uses ``is not None`` guards.

    Marimo rule (the reason both buttons are created here, not inline in
    the sampling cell): a UI element's ``.value`` cannot be accessed in
    the same cell that created it — the click event must propagate cross-cell.
    So buttons are created here; the sampling cell reads ``.value``.
    """
    run_button = None
    populate_btn = None
    fail_regenerate_btn = None
    _estimate_and_button = mo.md("")
    _is_smc = recipe is not None and hasattr(recipe, "smc_method_name")
    if recipe is None or _is_smc or recipe.effort == Effort.GROUNDTRUTH:
        # No estimate box or button needed for SMC recipes or GROUNDTRUTH.
        pass
    elif recipe.effort == Effort.FAILED:
        # FAILED: show a diagnostic callout + regenerate button.
        # FAIL recipes may have incomplete calibration_budget (stub entries
        # with only wall_seconds_estimate; no sampling_seconds_per_draw or
        # warmup_wall_seconds) — guard every timing field against None.
        _n = n_samples_slider.value if n_samples_slider is not None else 400
        _budget = recipe.calibration_budget or {}
        _ww = _budget.get("warmup_wall_seconds") or 0.0
        _spd = _budget.get("sampling_seconds_per_draw") or 0.0
        _c = int(
            _budget.get("num_chains")
            or (recipe.warmup_params or {}).get("num_chains")
            or 4
        )
        _OVERHEAD_S = 20.0
        _est_tot = _ww + _spd * _n * _c + _OVERHEAD_S
        _est_min = _est_tot / 60.0
        # If all timing fields were None, the estimate is dominated by _OVERHEAD_S
        # alone (~0.3 min) — show "estimate unavailable" instead of a misleading 0.
        _has_timing = _ww > 0.0 or _spd > 0.0
        _est_str = (
            f"Estimated wall: **{_est_min:.1f} min** for n_samples={_n}"
            if _has_timing
            else "Estimated wall: **unavailable** (no timing stamp on this FAIL recipe)"
        )

        fail_regenerate_btn = mo.ui.run_button(label="Re-run failed config")

        _fail_callout = mo.callout(
            mo.md(
                "⚠️ **FAIL recipe** — no gate-passing configuration. "
                "Click **Re-run failed config** to execute the pinned warmup + "
                "sampler settings and render diagnostic plots (trace, rank, "
                "divergences) so the failure mode is visually inspectable. "
                f"{_est_str} "
                "(full warmup; skip_warmup not used)."
            ),
            kind="danger",
        )
        _estimate_and_button = mo.vstack([_fail_callout, fail_regenerate_btn])
    else:
        _use_cache = use_cached_switch.value if use_cached_switch is not None else True
        if _use_cache:
            # Cache path: create the populate button so the sampling cell can
            # display it on a cache miss (no estimate box in this cell — the
            # cache should be instant when it hits; the populate button is
            # only surfaced on miss by the sampling cell).
            populate_btn = mo.ui.run_button(label="Run + populate cache")
        else:
            # Non-cache path: show dynamic estimate + Run button.
            _n = n_samples_slider.value if n_samples_slider is not None else 1000
            _skip = (
                skip_warmup_toggle.value if skip_warmup_toggle is not None else False
            )

            _budget = recipe.calibration_budget or {}
            # NB: `dict.get(key, default)` returns the stored value (even None)
            # when the key is present; only absent keys fall back to `default`.
            # Some recipes (e.g. irt_2pl medium) have explicit None values for
            # these timing fields (M2 backfill gap), so we need `or 0.0` to
            # coerce None → 0.0. Matches the no-skip branch at line 300.
            _spd = _budget.get("sampling_seconds_per_draw") or 0.0
            _ww = _budget.get("warmup_wall_seconds") or 0.0
            # Chain count: recipe.warmup_params["num_chains"] is the canonical
            # source (per _certification_runner.py — defaults to 4 chains
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
    return fail_regenerate_btn, populate_btn, run_button


@app.cell(hide_code=True)
def _(
    Effort,
    cached_idata_for_recipe,
    fail_regenerate_btn,
    load_idata,
    mo,
    n_samples_slider,
    populate_btn,
    recipe,
    regenerate_idata,
    run_button,
    skip_warmup_toggle,
    use_cached_switch,
):
    # Marimo cell-display rule: only the LAST bare expression renders. Assign
    # all UI messages to a single `_panel` variable + bare-display it at the
    # end (matches the timing_panel / _estimate_and_button pattern in this
    # same notebook). Without this, every `mo.md(...)` call in the branches
    # below gets evaluated-and-discarded — explaining "mo.md is not displaying
    # anything" debugging.
    idata = None
    _panel = mo.md("")
    _is_smc = recipe is not None and hasattr(recipe, "smc_method_name")
    if recipe is None:
        _panel = mo.md("*Pick a recipe to see diagnostic plots.*")
    elif _is_smc:
        _panel = mo.callout(
            mo.md(
                "**SMC recipe** — the MCMC sampling panel does not apply. "
                "See the summary table above for cert evidence "
                "(particle_ess, max_abs_mean_z, λ_final, n_smc_steps, override note)."
            ),
            kind="info",
        )
    elif recipe.effort == Effort.GROUNDTRUTH:
        # Groundtruth: load from cache (already persisted)
        try:
            idata = load_idata(recipe)
            _panel = mo.md(
                f"**Posterior sites**: `{list(idata['posterior'].data_vars)}`"
            )
        except FileNotFoundError:
            _panel = mo.md(
                "*Cache miss for GROUNDTRUTH recipe (git lfs pull may be needed).*"
            )
    elif recipe.effort == Effort.FAILED:
        # FAILED: offer on-demand re-run via the regenerate button created above.
        # The button was already displayed by the estimate cell; here we consume its
        # .value and trigger regenerate_idata when clicked.
        _n = n_samples_slider.value if n_samples_slider is not None else 400
        if fail_regenerate_btn is not None and fail_regenerate_btn.value:
            try:
                idata = regenerate_idata(recipe, n_samples=_n, replay_pinned=False)
                _panel = mo.callout(
                    mo.md(
                        f"✅ Re-run complete — **n_samples={_n}** per chain. "
                        "Diagnostic plots render below. "
                        "Divergence markers (▲ in trace plots) indicate divergent "
                        "transitions; stuck chains appear as flat traces."
                    ),
                    kind="success",
                )
            except Exception as exc:
                _panel = mo.callout(
                    mo.md(
                        f"❌ Re-run failed: `{type(exc).__name__}: {exc}`\n\n"
                        "This may indicate the sampler is non-terminating "
                        "(e.g. stiff ODE geometry with adjusted_mclmc_tuning) "
                        "or a genuine runtime error."
                    ),
                    kind="danger",
                )
        else:
            _panel = mo.md(
                "*Click **Re-run failed config** above to visually inspect "
                "the failure mode (trace plots, divergences, rank plots).*"
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
                # re-sample). On miss, surface the populate button (created in
                # the estimate cell; marimo rule prevents creating + accessing
                # .value in the same cell) so the user can opt-in to re-sample +
                # populate the cache (force_regenerate=True — the explicit-
                # consent path kept on the API).
                try:
                    idata = cached_idata_for_recipe(recipe)
                except FileNotFoundError:
                    if populate_btn is not None and populate_btn.value:
                        # Click: re-sample, save to cache, done.
                        idata = cached_idata_for_recipe(recipe, force_regenerate=True)
                    else:
                        idata = None
                        _panel = mo.vstack(
                            [
                                mo.callout(
                                    mo.md(
                                        "⚠ **Cache miss** for this recipe. "
                                        "Click below to sample + populate the "
                                        "cache (takes ~the wall-time you'd see "
                                        "by toggling off **Use cache**), or "
                                        "toggle off **Use cache** to preview "
                                        "the estimate before running."
                                    ),
                                    kind="warn",
                                ),
                                populate_btn if populate_btn is not None else mo.md(""),
                            ]
                        )
            elif run_button is not None and run_button.value:
                # Non-cache path with Run button click: sample now.
                # (``run_button`` is None when the estimate cell short-circuited
                # — recipe None / GROUNDTRUTH / FAILED / cache mode — so the
                # ``is not None`` guard keeps this branch safe in all states.)
                if _skip:
                    # Skip-warmup: bypass adaptation, use stored step_size/IMM,
                    # stationary init from GT-means. Always re-runs (no cache path).
                    idata = regenerate_idata(recipe, n_samples=_n, replay_pinned=True)
                else:
                    # Force resample: re-run warmup + sampling with recipe tuning_seed
                    # and the requested n_samples.
                    idata = regenerate_idata(
                        recipe,
                        n_samples=_n,
                        seed=recipe.tuning_seed,
                        replay_pinned=False,
                    )

            if idata is not None:
                _panel = mo.vstack(
                    [
                        _skip_warn or mo.md(""),
                        mo.md(
                            f"**Posterior sites**: `{list(idata['posterior'].data_vars)}`"
                        ),
                    ]
                )
        except Exception as e:
            _panel = mo.md(f"*Failed to sample recipe: {type(e).__name__}: {e}*")
    _panel
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
