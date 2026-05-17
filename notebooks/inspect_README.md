# tuningfork inspect / render — Recipe inspection helpers

User-facing API for inspecting any recipe in `inference/recipes/starter/`.
Designed for Jupyter Lab use. Statistician-friendly: minimal wrapper code,
ArviZ-direct workflow.

## Quick start (4 lines in a Jupyter cell)

```python
from tuningfork.inspect import load_recipe, summarize_recipe
from tuningfork.render import load_idata
import arviz as az

recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
summarize_recipe(recipe)               # HTML table inline
idata = load_idata(recipe)             # posterior + sample_stats
az.plot_trace(idata)
az.plot_energy(idata)                  # uses sample_stats.energy
az.plot_pair(idata, divergences=True)  # uses sample_stats.diverging
az.summary(idata)
```

`load_idata` is the **recommended one-call** that bundles `load_samples` +
`load_chain_stats` + `samples_to_idata`. The returned `InferenceData` carries
both the **posterior** group and a **sample_stats** group with NUTS
diagnostics mapped to ArviZ's canonical schema names (`diverging`,
`energy`, `acceptance_rate`, `n_steps`, `tree_depth`). For GROUNDTRUTH
recipes specifically the sample_stats are **further enriched** with
`step_size` (broadcast adapted scalar) and `reached_max_treedepth` (derived
from `num_trajectory_expansions` ≥ `max_num_doublings`).

See `inspect_example.md` for a full worked example.
See `notebooks/recipe_diagnostics.md` (one directory up) for the parametrized template.

## API

| Function | Returns | Notes |
|---|---|---|
| `load_recipe(path)` | `Recipe` | Resolves relative paths against repo root |
| `summarize_recipe(recipe)` | `pd.DataFrame` | Auto-renders inline; IMM excluded |
| `load_idata(recipe)` | `az.InferenceData` / `DataTree` | **Recommended.** Posterior + sample_stats; GROUNDTRUTH gets enrichment |
| `load_samples(recipe)` | `dict[str, jax.Array]` | Advanced: raw draws. Raises `FileNotFoundError` on cache miss |
| `load_chain_stats(recipe)` | `dict[str, np.ndarray] \| None` | Advanced: raw chain_stats. `None` on miss (non-fatal) |
| `samples_to_idata(samples, chain_stats=None)` | `az.InferenceData` / `DataTree` | Manual conversion |

### `load_recipe(path)`

Accepts an absolute path or a path relative to the tuningfork repo root (detected
via the installed package location). The repo-root resolution means notebooks can
use paths like `"tuningfork/catalog/<model>/..."` regardless of the
kernel's working directory.

Raises `FileNotFoundError` with a clear message if the file cannot be found.

### `summarize_recipe(recipe)`

Returns a 13-row `pd.DataFrame(columns=["Property", "Value"])` covering:
model, effort, sampler, warmup, stored gate verdict, R̂_max, min_bulk_ESS,
n_divergences, tuning_seed, tuningfork / blackjax / jax versions, timestamp.

The `inverse_mass_matrix` field is intentionally excluded (too verbose for a
summary table; inspect `recipe.base_method_params` directly for the IMM).

### `load_idata(recipe)`

The recommended entry point for inspection. Returns an `InferenceData` with:

- **posterior** — the cached samples, shape `(1, n_draws, *event)` (single
  long chain promoted to multi-chain layout for ArviZ).
- **sample_stats** — per-step NUTS diagnostics mapped to ArviZ canonical
  names: `diverging`, `energy`, `acceptance_rate`, `n_steps`, `tree_depth`,
  plus a prefixed `tuningfork_is_turning`.
- **(GROUNDTRUTH only)** further enriched sample_stats with `step_size`
  (broadcast adapted scalar) and `reached_max_treedepth` (derived from
  `num_trajectory_expansions ≥ max_num_doublings`).

Use `az.plot_energy`, `az.plot_pair(idata, divergences=True)`,
`az.plot_trace(idata, divergences="bottom")` etc. — the sample_stats group
makes these work out of the box.

### `load_samples(recipe)`

Cache-only in v1 — no re-run path. Only GROUNDTRUTH recipes have a populated
reference cache today. For other effort tiers, raises `FileNotFoundError` with
a message pointing at the Phase 0 sweep documentation.

To populate the cache, run the Phase 0 ground-truth sweep:
`tuningfork reference <model_name>`.

### `load_chain_stats(recipe)`

Returns the raw per-step NUTS info dict (or `None` on cache miss). Most users
should call `load_idata` instead — this exists for diagnostic deep-dives.

### `samples_to_idata(samples, is_multichain=False, chain_stats=None)`

Re-export of `tuningfork.diagnostics.samples_to_idata`. The default
`is_multichain=False` matches the shape returned by `load_samples`
(single-chain reference draws, shape `(n_samples, *event_shape)`), which gets
promoted to `(1, n_samples, *event_shape)` for ArviZ.

For multi-chain outputs (e.g., from your own warmup+sampler run), pass
`is_multichain=True`.

When `chain_stats` is provided, the function projects it into the
`sample_stats` group using `_CHAIN_STATS_TO_SAMPLE_STATS` (renames
`is_divergent → diverging`, `num_integration_steps → n_steps`, etc.).

## Sampling-book pattern reference

We mirror the canonical statistician-facing workflow from the
[sampling-book change-of-variable HMC example](https://blackjax-devs.github.io/sampling-book/models/change-of-variable-hmc/#arviz-plots).
ArviZ calls are direct (`az.plot_trace`, `az.summary`, `az.plot_rank`,
`az.plot_energy`); no custom wrapper functions in the user-facing notebook.

## File layout

```
tuningfork/
├── inspect.py           # load_recipe, summarize_recipe
└── render.py            # load_samples, load_chain_stats, load_idata, samples_to_idata

notebooks/
├── inspect_README.md    # this file
├── inspect_example.md   # worked example notebook (jupytext .md)
└── recipe_diagnostics.md  # parametrized template notebook
```

## Version history

- 2026-05-12 (notebook-arviz-redesign): initial public API.
  NUTS/HMC only; family B/C/D/E deferred to Recipe Phases 2-6.
- 2026-05-12 (`samples_to_idata` + `load_idata` extension): added
  `load_idata` one-call helper + `load_chain_stats`; `samples_to_idata`
  gains `chain_stats` kwarg projecting to ArviZ canonical `sample_stats`
  schema (6 mappings + 2 GROUNDTRUTH-derived fields).
