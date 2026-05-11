# Starter Recipes

This directory contains canonical, committed recipes per `(model, warmup,
sampler)` cell. The cell space spans 14 models × the registered warmups
× the 24 base methods, scoped down by warmup–sampler compatibility and
the supersession map.

---

## What is a Recipe?

A **Recipe** is a pinned `(model, warmup, sampler)` configuration stored as
JSON. Each recipe includes:

- The sampler name and its pinned hyperparameters
- The warmup procedure used and its hyperparameters (LOW always runs warmup
  at recipe-build time — `no_warmup` is the warmup name only for cells where
  there is no canonical adaptation, e.g., gradient-free RWM/IRMH or specialised
  Laplace-marginal samplers)
- The Statistician auto-gate verdict (`gate_evidence.auto`) and any manual
  override (`gate_evidence.override`)
- User-facing instructions auto-templated from the pinned fields
- Provenance: which versions of `tuningfork`, `blackjax`, and `jax` produced it

Effort tiers measure **human + machine wall time to produce a gate-passing
recipe**, escalated by the Statistician → TL when the auto-gate fails.  See
`_base.py` `Effort` docstring for the full per-tier definition:

| Effort | What produced it | Wall time | When emitted |
|---|---|---|---|
| `low` | Conventional `(warmup, sampler)` pairing with library defaults; auto-gate passed at first emit | machine-only | Cell is in the *conventional combinations* set (e.g., `stan_window` + `nuts`, `mclmc_tuning` + `mclmc`) and defaults work |
| `medium` | Statistician investigation: either (a) LOW gate failure → seed/init/bug-fix workaround, or (b) explores a technically-possible-but-unconventional pairing (e.g., `stan_window` + `mala`, `stan_window` + `rmhmc`) | LOW + Statistician investigation | LOW failed, or the cell is non-canonical |
| `high` | Oracle (NUTS+window_adaptation) comparison + BO over warmup hyperparameters + model-specific param injection; the full journey is recorded in `workflow` | MEDIUM + extra Statistician work + BO compute | LOW and MEDIUM both failed |

**Per-cell recipe count is normally 1** (the lowest tier that passed).  Edge
case: a single cell can carry multiple recipes if LOW passes but is unstable
across seeds, or if extra effort yields better ESS/grad.

---

## Filename Convention

```
<effort>__<base_method>__<warmup>.json
```

Examples:

```
low__nuts__stan_window.json
low__mclmc__mclmc_tuning.json
medium__mala__stan_window.json    # unconventional pairing → MEDIUM
high__nuts__stan_window.json      # only emitted when LOW + MEDIUM failed
```

The warmup is determined by the cell, not by the effort.  `low__nuts__no_warmup.json`
would only appear if the LOW emit pipeline deliberately tested the no-warmup
variant of NUTS (and that's not a conventional combination — usually it would
fall under MEDIUM as an exploration of an unconventional pairing).

---

## How to Regenerate

Run the generator script from the `bjx-bench/` project root:

```bash
cd bjx-bench
uv run python tuningfork/inference/recipes/_generate_starter.py
```

This re-stamps all LOW-effort recipes against the current installed versions
of `tuningfork`, `blackjax`, and `jax`. The script is idempotent — it
overwrites existing files with fresh provenance timestamps.

Regenerate whenever:
- `jax` or `blackjax` is upgraded (provenance versions change)
- A `BaseMethod.default_hp_space` changes (default HPs change)
- A new starter model is added (add it to the `STARTER_MODELS` list in the script)

---

## Schema Reference

See `bjx_bench/inference/recipes/_base.py` for the full `Recipe` dataclass
definition and design rationale.
