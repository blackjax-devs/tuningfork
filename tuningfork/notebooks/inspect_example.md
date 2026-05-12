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
from tuningfork.notebooks import load_recipe, summarize_recipe, load_samples, samples_to_idata

matplotlib.use("Agg")
plt.rcParams["figure.figsize"] = (10, 6)

recipe = load_recipe(
    "tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json"
)
summarize_recipe(recipe)  # auto-renders as HTML table in Jupyter
```

## Step 2: Load cached samples and convert to InferenceData

```{code-cell} ipython3
samples = load_samples(recipe)
print("Loaded sites:", {k: v.shape for k, v in samples.items()})
idata = samples_to_idata(samples)
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

## Interpretation

- **`az.summary`**: check R̂ (< 1.01 is ideal) and ESS (> 400 per parameter).
- **`az.plot_trace`**: rank bars should be roughly uniform — poor mixing shows as
  structured (low or high) bars.
- **`az.plot_rank`**: complements the trace; clear non-uniformity signals chains
  that are not mixing.
- **`az.plot_energy`**: `E-BFMI > 0.3` indicates good HMC energy exploration.
  A bimodal or narrow marginal energy distribution (versus the transition
  distribution) is a sign of step-size miscalibration.

If any of these look pathological for a groundtruth recipe, escalate to the TL
per `STATISTICIAN_BAYESIAN_WORKFLOW.md`.
