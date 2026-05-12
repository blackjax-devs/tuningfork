# tuningfork.notebooks — Recipe inspection helpers

User-facing API for inspecting any recipe in `inference/recipes/starter/`.
Designed for Jupyter Lab use. Statistician-friendly: minimal wrapper code,
ArviZ-direct workflow.

## Quick start (5 lines in a Jupyter cell)

```python
from tuningfork.notebooks import load_recipe, summarize_recipe, load_samples, samples_to_idata
import arviz as az

recipe = load_recipe("tuningfork/inference/recipes/starter/eight_schools_ncp/groundtruth__nuts__stan_window.json")
summarize_recipe(recipe)  # renders inline as HTML table
samples = load_samples(recipe)
idata = samples_to_idata(samples)
az.plot_trace(idata)
az.summary(idata)
```

See `inspect_example.md` for a full worked example.
See `notebooks/recipe_diagnostics.md` (one directory up) for the parametrized template.

## API

| Function | Returns | Notes |
|---|---|---|
| `load_recipe(path)` | `Recipe` | Resolves relative paths against repo root |
| `summarize_recipe(recipe)` | `pd.DataFrame` | Auto-renders inline; IMM excluded |
| `load_samples(recipe)` | `dict[str, jax.Array]` | Raises `FileNotFoundError` on cache miss |
| `samples_to_idata(samples)` | `az.InferenceData` / `DataTree` | Re-export of diagnostics helper; `is_multichain=False` default |

### `load_recipe(path)`

Accepts an absolute path or a path relative to the tuningfork repo root (detected
via the installed package location). The repo-root resolution means notebooks can
use paths like `"tuningfork/inference/recipes/starter/..."` regardless of the
kernel's working directory.

Raises `FileNotFoundError` with a clear message if the file cannot be found.

### `summarize_recipe(recipe)`

Returns a 13-row `pd.DataFrame(columns=["Property", "Value"])` covering:
model, effort, sampler, warmup, stored gate verdict, R̂_max, min_bulk_ESS,
n_divergences, tuning_seed, tuningfork / blackjax / jax versions, timestamp.

The `inverse_mass_matrix` field is intentionally excluded (too verbose for a
summary table; inspect `recipe.base_method_params` directly for the IMM).

### `load_samples(recipe)`

Cache-only in v1 — no re-run path. Only GROUNDTRUTH recipes have a populated
reference cache today. For other effort tiers, raises `FileNotFoundError` with
a message pointing at the Phase 0 sweep documentation.

To populate the cache, run the Phase 0 ground-truth sweep:
`tuningfork reference <model_name>`.

### `samples_to_idata(samples, is_multichain=False)`

Re-export of `tuningfork.diagnostics.samples_to_idata`. The default
`is_multichain=False` matches the shape returned by `load_samples`
(single-chain reference draws, shape `(n_samples, *event_shape)`), which gets
promoted to `(1, n_samples, *event_shape)` for ArviZ.

For multi-chain outputs (e.g., from your own warmup+sampler run), pass
`is_multichain=True`.

## Sampling-book pattern reference

We mirror the canonical statistician-facing workflow from the
[sampling-book change-of-variable HMC example](https://blackjax-devs.github.io/sampling-book/models/change-of-variable-hmc/#arviz-plots).
ArviZ calls are direct (`az.plot_trace`, `az.summary`, `az.plot_rank`,
`az.plot_energy`); no custom wrapper functions in the user-facing notebook.

## File layout

```
tuningfork/notebooks/
├── __init__.py          # re-exports load_recipe, summarize_recipe, load_samples, samples_to_idata
├── inspect.py           # load_recipe, summarize_recipe
├── render.py            # load_samples, samples_to_idata
├── README.md            # this file
└── inspect_example.md   # worked example notebook (jupytext .md)
```

The parametrized template notebook lives at `notebooks/recipe_diagnostics.md`
(one level up, alongside the notebooks/ source directory).

## Version history

- 2026-05-12 (notebook-arviz-redesign): initial public API.
  NUTS/HMC only; family B/C/D/E deferred to Recipe Phases 2-6.
