# BLOCKED: adjusted_mclmc / adjusted_mclmc_dynamic under jax.vmap

**Date**: 2026-05-30
**Branch**: mclmc-easy-model-sweep
**Blocking**: adjusted_mclmc and adjusted_mclmc_dynamic emit cells (10 of 15 planned)

## What was tried

Smoke → G1b (logistic_synthetic × adjusted_mclmc_tuning × adjusted_mclmc) crashes with:
```
ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape float32[]
The problem arose with the `float` function.
```
at `_recipe_runner.py:1022` (inside `jax.vmap(_step_one_chain)`).

## Root cause

`tuningfork/base_method/adjusted_mclmc.py:76`:
```python
n_steps = max(1, int(round(float(L) / float(step_size))))
```

This runs INSIDE `jax.vmap()` (the `_vmapped_step` closure in `_recipe_runner.py`). Under vmap, `step_size` and `L` are traced JAX arrays — `float()` on a traced value raises ConcretizationTypeError.

Same issue affects `adjusted_mclmc_dynamic` (same factory pattern).

## What would fix it

Runner-level fix (analogous to the `laplace_*` special path at runner:~952):
1. Detect `sampler_name in ("adjusted_mclmc", "adjusted_mclmc_dynamic")`
2. Build per-chain `step` functions OUTSIDE vmap using concrete per-chain `step_size` and `L` from `batched_params`
3. Vmap only the step function application (not the kernel build)

OR: Fix `adjusted_mclmc._factory` to avoid `float()` and use JAX-compatible `jnp.round(L / step_size)`, then verify `integration_steps_params=(jnp_n_steps,)` is accepted by blackjax.

## What unblocks this

@swe: fix `adjusted_mclmc.py:76` and/or the runner special path for adjusted_mclmc.

## Current state

Vanilla `mclmc` × `mclmc_tuning` works fine (no `float()` in its factory).
Proceeding with 5 vanilla mclmc cells. 10 adjusted_mclmc/adjusted_mclmc_dynamic cells blocked.
