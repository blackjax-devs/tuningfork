# Starter Recipes

This directory contains canonical, committed recipes for the 3 starter benchmark
models (`mvn_10`, `neals_funnel`, `eight_schools_ncp`).

---

## What is a Recipe?

A **Recipe** is a pinned `(model, base_method, warmup)` configuration stored as
JSON. Each recipe includes:

- The sampler name and its pinned hyperparameters
- The warmup procedure used (or `"no_warmup"` for LOW-effort recipes)
- User-facing instructions for copy-pasting into analysis code
- Provenance: which versions of `bjx-bench`, `blackjax`, and `jax` produced it

Recipes serve three user personas:

| Effort | Persona | Calibration cost | When to use |
|--------|---------|-----------------|-------------|
| `low`  | One-off analyst | Zero (default HPs) | Exploratory work, prototyping |
| `medium` | Standard analysis | ~1 min (warmup only) | Routine analysis needing adapted step size |
| `high` | Production / repeated runs | ~30+ min (Tier-B BO) | Where ESS/grad matters and time is available |

---

## Filename Convention

```
<effort>__<base_method>__<warmup>.json
```

Examples:

```
low__nuts__no_warmup.json
medium__nuts__stan_window.json
high__hmc__stan_window.json
```

---

## Current Contents (Phase 2.5, Commit 3 of 4)

This directory currently holds **6 LOW-effort recipes** only:

- 3 starter models × 2 algorithms (`hmc`, `nuts`) × `low` = 6 JSON files

MEDIUM and HIGH recipes are deferred to a follow-up spawn. They require:

- MEDIUM: running `stan_window` warmup (~1 min per recipe × 6 = ~6 min)
- HIGH: running Tier-B BO with 50+ trials (~5–30 min per recipe × 6 = ~30–180 min)

---

## How to Regenerate

Run the generator script from the `bjx-bench/` project root:

```bash
cd bjx-bench
uv run python bjx_bench/inference/recipes/_generate_starter.py
```

This re-stamps all LOW-effort recipes against the current installed versions
of `bjx-bench`, `blackjax`, and `jax`. The script is idempotent — it
overwrites existing files with fresh provenance timestamps.

Regenerate whenever:
- `jax` or `blackjax` is upgraded (provenance versions change)
- A `BaseMethod.default_hp_space` changes (default HPs change)
- A new starter model is added (add it to the `STARTER_MODELS` list in the script)

---

## Schema Reference

See `bjx_bench/inference/recipes/_base.py` for the full `Recipe` dataclass
definition and `PLAN_bjx_bench_restructure.md` § "Recipe schema" for the
design rationale.
