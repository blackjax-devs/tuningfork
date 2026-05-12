---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Worked Example: Inspecting an eight_schools_ncp Groundtruth Recipe

This notebook shows the full ArviZ-direct inspection flow for the
`eight_schools_ncp` NUTS groundtruth recipe — the canonical Phase 0 reference.

It uses `tuningfork.notebooks` (4 functions) and then calls ArviZ directly.
No custom render functions; no custom wrapper code.

Reference: [sampling-book change-of-variable HMC](https://blackjax-devs.github.io/sampling-book/models/change-of-variable-hmc/#arviz-plots)

## Step 1: Load + summarize the recipe

```{code-cell} ipython3
import matplotlib
import matplotlib.pyplot as plt

import arviz as az
from tuningfork.notebooks import load_recipe, load_idata, summarize_recipe

matplotlib.use("Agg")
plt.rcParams["figure.figsize"] = (10, 6)

recipe = load_recipe(
    "tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json"
)
summarize_recipe(recipe)  # auto-renders as HTML table in Jupyter
```

## Step 2: Load InferenceData (posterior + sample_stats)

`load_idata` is the recommended one-call. It bundles `load_samples` +
`load_chain_stats` + `samples_to_idata`. The returned `InferenceData`
carries both the posterior group AND a `sample_stats` group with NUTS
diagnostics mapped to ArviZ's canonical schema names. For GROUNDTRUTH
recipes specifically the sample_stats are further enriched with
`step_size` (broadcast adapted scalar) and `reached_max_treedepth`.

```{code-cell} ipython3
idata = load_idata(recipe)
print("posterior sites:", list(idata["posterior"].data_vars))
print("sample_stats:", list(idata["sample_stats"].data_vars))
```

## Step 3: ArviZ summary

```{code-cell} ipython3
az.summary(idata)
```

## Step 4: Trace plot

```{code-cell} ipython3
az.plot_trace(idata)
plt.tight_layout()
plt.show()
```

## Step 5: Rank plot

```{code-cell} ipython3
az.plot_rank(idata)
plt.tight_layout()
plt.show()
```

## Step 6: Energy plot

```{code-cell} ipython3
az.plot_energy(idata)
plt.tight_layout()
plt.show()
```

## Step 7: Divergences (if any)

For GROUNDTRUTH recipes the relaxed gate allows up to 0.1% divergence rate.
Visualize divergent transitions in pairs of parameters:

```{code-cell} ipython3
n_divs = int(idata["sample_stats"]["diverging"].sum())
print(f"divergent transitions: {n_divs} of {idata['sample_stats']['diverging'].size}")
if n_divs > 0:
    az.plot_pair(idata, divergences=True)
    plt.show()
```

## Step 8: Tree-depth distribution (NUTS diagnostic)

```{code-cell} ipython3
import numpy as np
td = np.asarray(idata["sample_stats"]["tree_depth"])
print(f"tree_depth p50={int(np.median(td))}, p95={int(np.percentile(td, 95))}, max={int(td.max())}")
if "reached_max_treedepth" in idata["sample_stats"].data_vars:
    rmt = np.asarray(idata["sample_stats"]["reached_max_treedepth"])
    print(f"reached_max_treedepth: {int(rmt.sum())} / {rmt.size} ({100*rmt.mean():.2f}%)")
```

## Interpretation

- **`az.summary`**: check R̂ (< 1.01 is ideal) and ESS (> 400 per parameter).
- **`az.plot_trace`**: trace bars should be roughly uniform — poor mixing shows as
  structured (low or high) bars.
- **`az.plot_rank`**: complements the trace; clear non-uniformity signals chains
  that are not mixing.
- **`az.plot_energy`**: `E-BFMI > 0.3` indicates good HMC energy exploration.
  A bimodal or narrow marginal energy distribution (versus the transition
  distribution) is a sign of step-size miscalibration.
- **Divergences**: at production groundtruth scale, a handful (≤ 0.1% of
  n_samples) reflects geometry, not adaptation failure. A higher rate signals
  posterior-curvature issues — see `STATISTICIAN_DIAGNOSTICS_RECIPE.md`.
- **`reached_max_treedepth`**: should be 0%. If non-zero, NUTS hit the tree
  doubling cap (e.g., `max_num_doublings=10` → 1024 leapfrog steps) — bump
  `max_num_doublings` in the recipe's warmup_params (Phase 0 statistician
  precedent: horseshoe used 15).

If any of these look pathological for a groundtruth recipe, escalate to the TL
per `STATISTICIAN_BAYESIAN_WORKFLOW.md`.
