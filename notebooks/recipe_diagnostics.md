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

# Recipe Diagnostics — NUTS / HMC

Load a NUTS or HMC groundtruth recipe, inspect its metadata, load the cached
reference draws, and run ArviZ diagnostics directly.

**Target audience**: a statistician verifying that a recipe's cached reference
draws look healthy before using them as a benchmark ground truth.

For other sampler families (MCLMC, SMC, VI, specialised), see the deferred
design in `worklog/threads/notebook-arviz-redesign.md`.

```{code-cell} ipython3
:tags: [parameters]

# Papermill / jupytext parameter cell — edit these to inspect a different recipe
RECIPE_PATH: str = "tuningfork/catalog/eight_schools_ncp/groundtruth.json"
QUICK_MODE: bool = True
N_SAMPLES_QUICK: int = 1000
N_CHAINS: int = 4
```

```{code-cell} ipython3
import matplotlib
import matplotlib.pyplot as plt

import arviz as az
from tuningfork.catalog.inspect import load_recipe, summarize_recipe
from tuningfork.catalog.render import load_idata

matplotlib.use("Agg")
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (10, 6)

recipe = load_recipe(RECIPE_PATH)
summarize_recipe(recipe)
```

```{code-cell} ipython3
# Only NUTS / HMC groundtruth recipes are handled in this notebook.
# For other families, see worklog/threads/notebook-arviz-redesign.md § 3.
assert recipe.base_method_name in {"nuts", "hmc"}, (
    f"This notebook handles only NUTS/HMC recipes. "
    f"Got {recipe.base_method_name!r}. "
    "For other sampler families see worklog/threads/notebook-arviz-redesign.md"
)
```

```{code-cell} ipython3
# One-call: returns InferenceData with posterior + sample_stats
# (diverging, energy, acceptance_rate, n_steps, tree_depth — and for
# GROUNDTRUTH recipes, additionally step_size + reached_max_treedepth).
idata = load_idata(recipe)
print(f"posterior sites: {list(idata['posterior'].data_vars)}")
print(f"sample_stats fields: {list(idata['sample_stats'].data_vars)}")
```

```{code-cell} ipython3
az.summary(idata)
```

```{code-cell} ipython3
az.plot_trace(idata)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
az.plot_rank(idata)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
az.plot_energy(idata)
plt.tight_layout()
plt.show()
```
