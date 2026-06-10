# Phase (c) Track 2 Failure Analysis — stoch_vol

**Date:** 2026-06-09
**Branch:** feat/calibrated-emission @ a08abc4
**Author:** @statistician

## Verdict: 0/3 FAIL — mission fallback applies, track STOPS

| Seed  | Verdict | Error |
|-------|---------|-------|
| 77777 | ERROR   | Cannot concatenate arrays with different numbers of dimensions: got (1, 500), (1,), (1,), (1,). |
| 88888 | ERROR   | Cannot concatenate arrays with different numbers of dimensions: got (1, 500), (1,), (1,), (1,). |
| 99999 | ERROR   | Cannot concatenate arrays with different numbers of dimensions: got (1, 500), (1,), (1,), (1,). |

Gate: R-hat<1.01, minESS≥400, div_rate≤5%, ≥2/3 seeds PASS. All 3 attempts exhausted.

## Root Cause: Library Shape Bug in _run_cert_seed (NOT sampling failure)

**Predicted mechanism:** 0/3 PASS, expected reason = posterior geometry (AR(1) near-unit-root
phi~0.96, d=503, weak pilot mixing). The prediction was numerically correct (0/3) but
the actual mechanism is different — the run errored before any samples were collected.

**Actual mechanism:** `_run_cert_seed` chain-stacking code assumes UNIFORM RANK across all
parameter leaves in the pytree. stoch_vol has a mixed-rank pytree:
  - `h`: shape (T=500,) → with chain batch dim: (1, 500)  [rank 2]
  - `sigma`, `phi`, `mu`: shape () → with chain batch dim: (1,)  [rank 1]

When the code attempts `jnp.concatenate([...])` across chains, JAX rejects the mixed
(1, 500) vs (1,) shapes at compile time. The JAX error fires before any MCMC kernel
runs — no samples, no R-hat, no ESS.

**This is a production code bug**, not a stoch_vol geometry limitation. It would affect
ANY model with mixed-rank parameter pytrees. Needs fix in `_run_cert_seed` (or its
chain-stacking utility): reshape each leaf to 1D before concatenating, or use
`jax.tree_util.tree_map` with per-leaf reshaping. Reported to @swe.

## Consequence for Mission

Per mission fallback: stoch_vol track STOPS. No retry.
- Phase (d) regenerates ill_cond_50 + german_credit goldens only.
- stoch_vol flatinit fork stays dead.
- Honest-null REVIEW record for stoch_vol LRD remains in catalog.

**Note on prediction accuracy:** The 0/3 outcome matches the prediction, but the
blocking mechanism is infrastructure, not geometry. The algorithm's actual performance
on stoch_vol (would it mix? would LRD help?) remains genuinely unknown — the sampling
kernel never ran. If the shape bug is fixed in a future iteration, stoch_vol LRD
would still face the geometry challenge (phi near unit-root, d=503), but that question
is out of scope for this mission.

## Config (for record)

- model: stoch_vol (standard registered, d=503: h(500), sigma, phi, mu)
- init_strategy: None (model_default, prior samples)
- k_rank: 40, pilot_n_warmup=1000, pilot_n_samples=1000
- n_samples: 1000, num_chains: 4
- seeds: 77777 / 88888 / 99999
- path: `_run_cert_seed` (policy-binding, tuningfork/emit_mclmc_lrd.py)
