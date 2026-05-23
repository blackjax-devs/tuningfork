# tuningfork inspect / render — Recipe inspection helpers

User-facing API for inspecting recipes in `tuningfork/catalog/<model>/`.
The recommended entry point is `from tuningfork.catalog import load_recipe, summarize_recipe`.
Designed for Jupyter Lab use. Statistician-friendly: minimal wrapper code,
ArviZ-direct workflow.

## Quick start (4 lines in a Jupyter cell)

```python
from tuningfork.catalog.inspect import load_recipe, summarize_recipe
from tuningfork.catalog.render import load_idata
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
See `recipe_diagnostics.md` (in this directory) for the parametrized template.

## Interactive marimo notebook

For interactive exploration with dropdowns (no need to edit string paths):

```bash
uv sync --group notebook   # install marimo (~20 MB; opt-in)
uv run --group notebook marimo edit tuningfork/catalog/notebooks/catalog_explorer.py
```

The marimo notebook gives you a **model dropdown → recipe dropdown → auto-loaded
summary + plots**. Reactive cells re-execute on dropdown change; no
`widgets.observe` callbacks needed. Per-model "headline params" (typically
hyperpriors — e.g., `mu`/`tau` for eight_schools_ncp, `mu`/`phi`/`sigma` for
stoch_vol) render as `az.plot_trace` + `az.plot_pair` plots; bulk params
(NCP innovations etc.) render as a single `az.plot_forest` capped at 20
entries (avoid 500-row forest plots on high-dim models like stoch_vol).

Per-model headline_params + headline_coords are declared as fields on the
`Posterior` dataclass — see [`worklog/decisions/2026-05-18-headline-params-per-model.md`](https://github.com/blackjax-devs/claude-config/blob/main/project/worklog/decisions/2026-05-18-headline-params-per-model.md)
for the ratified per-model values + rationale.

For the jupytext / Jupyter flow (papermill-batch-compatible, no widgets),
keep using `recipe_diagnostics.md` (in this directory).

## API

| Function | Returns | Notes |
|---|---|---|
| `load_recipe(path)` | `Recipe` | Resolves relative paths against repo root |
| `summarize_recipe(recipe)` | `pd.DataFrame` | Auto-renders inline; IMM excluded; 16 rows including num_chains / n_warmup / n_samples |
| `load_idata(recipe)` | `az.InferenceData` / `DataTree` | **Recommended.** Posterior + sample_stats; GROUNDTRUTH gets enrichment |
| `cached_idata_for_recipe(recipe)` | `az.InferenceData` | LOW/MEDIUM on-demand resample with per-recipe caching. Under the hood of `load_idata` for non-groundtruth recipes. |
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

Returns a 16-row `pd.DataFrame(columns=["Property", "Value"])` covering:
model, effort, sampler, warmup, **num_chains**, **n_warmup**, **n_samples**,
stored gate verdict, R̂_max, min_bulk_ESS, n_divergences, tuning_seed,
tuningfork / blackjax / jax versions, timestamp.

The `inverse_mass_matrix` field is intentionally excluded (too verbose for a
summary table; inspect `recipe.base_method_params` directly for the IMM).

The three sample-budget fields (`num_chains`, `n_warmup`, `n_samples`) read
from `warmup_params` with fallback to `calibration_budget` (or vice versa for
`n_samples`); legacy groundtruth recipes that pre-date these fields show
`"N/A"`.

### `load_idata(recipe)`

The recommended entry point for inspection. Dispatches transparently based on
`recipe.effort`:

- **GROUNDTRUTH**: loads directly from the committed reference cache at
  `<model>/groundtruth_samples/blackjax/{draws,chain_stats}.npz`.
- **LOW / MEDIUM**: calls `cached_idata_for_recipe` which warmup+samples on first
  access and persists to `<model>/_cache/<recipe_stem>.{draws,chain_stats}.npz`
  (gitignored). Subsequent calls return the cache instantly (5 s → 0.0 s after
  first run).
- **FAILED**: raises `FileNotFoundError` — no gate-passing configuration exists.

Returns an `InferenceData` with:

- **posterior** — samples, shape `(num_chains, n_draws, *event)` for multi-chain
  LOW/MEDIUM recipes; `(1, n_draws, *event)` for single-chain GROUNDTRUTH.
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

GROUNDTRUTH recipes load directly from the reference cache at
`<model>/groundtruth_samples/blackjax/draws.npz`. LOW/MEDIUM recipes load
from the per-recipe on-demand cache populated by `cached_idata_for_recipe`
(PR #37); call `load_idata(recipe)` first to populate the cache if it doesn't
exist yet. FAILED recipes raise `FileNotFoundError` (no gate-passing config
exists to sample from).

### `load_chain_stats(recipe)`

Returns the raw per-step NUTS info dict (or `None` on cache miss). Most users
should call `load_idata` instead — this exists for diagnostic deep-dives.

### `samples_to_idata(samples, is_multichain=False, chain_stats=None)`

Re-export of `tuningfork.catalog.diagnostics.samples_to_idata`. The default
`is_multichain=False` matches the shape returned by `load_samples`
(single-chain reference draws, shape `(n_samples, *event_shape)`), which gets
promoted to `(1, n_samples, *event_shape)` for ArviZ.

For multi-chain outputs (e.g., from your own warmup+sampler run), pass
`is_multichain=True`.

When `chain_stats` is provided, the function projects it into the
`sample_stats` group using `_CHAIN_STATS_TO_SAMPLE_STATS` (renames
`is_divergent → diverging`, `num_integration_steps → n_steps`, etc.).

## Effort-aware loading

`load_idata(recipe)` dispatches on `recipe.effort` with three branches matching
what the `catalog_explorer.py` marimo notebook does:

| `recipe.effort` | Behaviour |
|---|---|
| `"groundtruth"` | Loads from committed reference cache (`groundtruth_samples/blackjax/`). Instant. |
| `"low"` / `"medium"` | Runs warmup + sampling on first call via `cached_idata_for_recipe`; persists to `<model>/_cache/<recipe_stem>.*` (gitignored). Fast on cache hit. |
| `"failed"` | Raises `FileNotFoundError`. No gate-passing config exists. Inspect the recipe's `attempted_configurations` field for the investigation trail. |
| `"high"` | Not yet shipped. Will follow the LOW/MEDIUM path when available. |

To force a fresh resample for a LOW/MEDIUM recipe, delete
`<model>/_cache/<recipe_stem>.draws.npz` and re-call `load_idata`.

## Reproducing a recipe

Given any recipe, `emit_script(recipe)` returns a Python script that
reproduces the recipe's inference. The **inference choreography** (warmup
+ sampler + inference loop) is hand-rolled inline — auditable in one file
with no tuningfork imports. The **model definition** is imported via
`from tuningfork.model import MODELS` — canonical NumPyro code lives
upstream, not duplicated as a template:

```python
from tuningfork.catalog import emit_script, load_recipe

recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
script = emit_script(recipe, num_samples=2000)

from pathlib import Path
Path("run_inference_for_select_recipe.py").write_text(script)
# Then in a fresh shell:
#   uv run --with tuningfork --with jax --with blackjax --with numpyro \
#       python run_inference_for_select_recipe.py
```

The emitted script's inference choreography shows the exact BlackJAX call
shape (`blackjax.window_adaptation(blackjax.nuts, ...)`, `blackjax.nuts(...).step`,
the `jax.lax.scan` inference loop) with the recipe's pinned hyperparameters
hard-coded. Users can inspect the inference shape without spelunking through
the tuningfork wiring layer.

**Design rationale** (R3.5-MVP follow-up clarification, 2026-05-17): we
considered fully-standalone scripts (model body inlined per-model template),
but that would have meant 14 model templates duplicating canonical NumPyro
code from `tuningfork/model/<model>.py` with permanent drift risk. Importing
the model upstream eliminates drift and keeps templates focused on the
*wiring* (which is what tuningfork ADDS over BlackJAX, per Principle A — a
heavy sampler or warmup template signals an upstream BlackJAX design smell
worth fixing there). The cost is one `pip install tuningfork` step.

As of R3.5-MVP (2026-05-17), templates exist for `window_adaptation_diag_imm` warmup and
`nuts` sampler only. R3.5b expands to the full 10 warmups × 24 samplers.

## Sampling-book pattern reference

We mirror the canonical statistician-facing workflow from the
[sampling-book change-of-variable HMC example](https://blackjax-devs.github.io/sampling-book/models/change-of-variable-hmc/#arviz-plots).
ArviZ calls are direct (`az.plot_trace`, `az.summary`, `az.plot_rank`,
`az.plot_energy`); no custom wrapper functions in the user-facing notebook.

## File layout

```
tuningfork/
└── catalog/
    ├── __init__.py          # re-exports: load_recipe, load_idata, ...
    ├── inspect.py           # load_recipe, summarize_recipe
    ├── render.py            # load_samples, load_chain_stats, load_idata, samples_to_idata
    ├── diagnostics.py       # ArviZ family-aware diagnostic renderers
    └── notebooks/
        ├── inspect_README.md    # this file
        ├── inspect_example.md   # worked example notebook (jupytext .md)
        └── recipe_diagnostics.md  # parametrized template notebook
```

## Version history

- 2026-05-12 (notebook-arviz-redesign): initial public API.
  NUTS/HMC only; family B/C/D/E deferred to future recipe sweeps.
- 2026-05-12 (`samples_to_idata` + `load_idata` extension): added
  `load_idata` one-call helper + `load_chain_stats`; `samples_to_idata`
  gains `chain_stats` kwarg projecting to ArviZ canonical `sample_stats`
  schema (6 mappings + 2 GROUNDTRUTH-derived fields).
