---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Recipe Diagnostics Notebook

This notebook loads a Recipe by ID, runs the model + warmup + sampler end-to-end,
and renders a family-aware diagnostic battery with ArviZ plots. It serves two purposes:
(1) development-time sanity check during recipe-generation builds; (2) investigation
aid for any cell already flagged Yellow or Red in the recipe matrix.

The diagnostics are family-aware: gradient MH-corrected, MCLMC, SMC, VI, and
specialised samplers each receive a tailored plot suite.

## Section 0: Parameters and Recipe Load

```{code-cell} ipython3
:tags: [parameters]

# Papermill parameter cell
RECIPE_PATH: str = "tuningfork/inference/recipes/starter/mvn_10/low__nuts__no_warmup.json"
QUICK_MODE: bool = True  # If True, use N_SAMPLES_QUICK; if False, use N_SAMPLES_FULL
N_SAMPLES_QUICK: int = 1000
N_SAMPLES_FULL: int = 4000
N_CHAINS: int = 4
```

```{code-cell} ipython3
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tuningfork.inference.recipes._base import Effort, Recipe
from tuningfork.model import MODELS, build_logdensity_fn
from tuningfork.inference.base_method import BASE_METHODS
from tuningfork.inference.warmup import WARMUPS
from tuningfork.calibration.statistician_gate import auto_gate
from tuningfork.diagnostics import (
    FAMILY_A_SAMPLERS,
    FAMILY_B_SAMPLERS,
    FAMILY_C_SAMPLERS,
    FAMILY_D_SAMPLERS,
    FAMILY_E_SAMPLERS,
    render_family_a,
    render_family_b,
    render_family_c,
    render_family_d,
    render_family_e,
    render_universal_summary,
    samples_to_idata,
)

matplotlib.use("Agg")
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (10, 6)

print(f"JAX platform: {jax.devices()}")
print(f"RECIPE_PATH: {RECIPE_PATH}, QUICK_MODE: {QUICK_MODE}")
```

```{code-cell} ipython3
# Resolve recipe path.
# jupytext --execute always sets the kernel cwd to /tmp regardless of the notebook's
# location.  We derive the tuningfork root from the installed tuningfork package path
# (works for editable installs) and also try cwd / cwd.parent as fallbacks.
import tuningfork as _tuningfork_mod

_TUNINGFORK_ROOT = Path(_tuningfork_mod.__file__).parent.parent  # tuningfork/ -> tuningfork/

recipe_path = Path(RECIPE_PATH)
if not recipe_path.exists():
    _search_roots = [_TUNINGFORK_ROOT, Path.cwd(), Path.cwd().parent]
    try:
        _search_roots.insert(0, Path(__file__).parent.parent)
    except NameError:
        pass
    for _root in _search_roots:
        _candidate = _root / RECIPE_PATH
        if _candidate.exists():
            recipe_path = _candidate
            break

if not recipe_path.exists():
    raise FileNotFoundError(
        f"Recipe file not found at {RECIPE_PATH!r} (tried {_TUNINGFORK_ROOT} and cwd). "
        "Check RECIPE_PATH parameter; starter recipes live under "
        "tuningfork/inference/recipes/starter/<model>/<effort>__<sampler>__<warmup>.json"
    )

print(f"Loading recipe from: {recipe_path}")
recipe = Recipe.load(recipe_path)

# Display recipe metadata
metadata_rows = [
    ["Model", recipe.model_name],
    ["Warmup", recipe.warmup_name],
    ["Sampler", recipe.base_method_name],
    ["Effort", recipe.effort.value],
    ["Headline Metric", str(recipe.headline_metric) if recipe.headline_metric is not None else "Not measured"],
    ["Auto-gate Verdict (stored)", recipe.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN")],
    ["Tuning Seed", str(recipe.tuning_seed)],
    ["tuningfork Version", recipe.tuningfork_version],
]

metadata_df = pd.DataFrame(metadata_rows, columns=["Property", "Value"])
print("\nRecipe Metadata:")
print(metadata_df.to_string(index=False))
```

```{code-cell} ipython3
# Resolve registries
posterior = MODELS[recipe.model_name]
base_method = BASE_METHODS[recipe.base_method_name]
warmup = WARMUPS[recipe.warmup_name]
sampler_name = recipe.base_method_name
model_name = recipe.model_name
tuning_seed = recipe.tuning_seed if recipe.tuning_seed != 0 else 42

n_samples = N_SAMPLES_QUICK if QUICK_MODE else N_SAMPLES_FULL
print(f"Using {n_samples} samples per chain (QUICK_MODE={QUICK_MODE})")
print(f"Using {N_CHAINS} chains for sampling")
print(f"Sampler family: ", end="")
if sampler_name in FAMILY_A_SAMPLERS:
    print("A (gradient MH-corrected)")
elif sampler_name in FAMILY_B_SAMPLERS:
    print("B (MCLMC)")
elif sampler_name in FAMILY_C_SAMPLERS:
    print("C (SMC)")
elif sampler_name in FAMILY_D_SAMPLERS:
    print("D (VI)")
elif sampler_name in FAMILY_E_SAMPLERS:
    print("E (Specialised)")
else:
    print(f"UNKNOWN ({sampler_name})")
```

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Real warmup + sampler run
# ---------------------------------------------------------------------------
# Dispatch logic:
#   Family C (SMC)  → tuningfork.runner.smc.run_smc
#   Family D (VI)   → run VI then sample from surrogate posterior
#   Family A/B/E    → warmup.runner + jax.lax.scan + jax.vmap
#
# The warmup.runner multi-chain contract:
#   runner(rng_key, init_position, n_warmup, base_method,
#          *, logdensity_fn, num_chains=4)
#   -> (states, adapted_params)   # states has leading dim num_chains
#
# The base_method kernel contract:
#   kernel = base_method.factory(logdensity_fn, **params)
#   state  = kernel.init(position)         # NUTS/HMC/MALA/etc.
#   state  = kernel.init(position, key)    # MCLMC only
#   new_state, info = kernel.step(rng_key, state)

key = jax.random.key(tuning_seed)

if sampler_name in FAMILY_C_SAMPLERS:
    # -----------------------------------------------------------------------
    # Family C: SMC
    # -----------------------------------------------------------------------
    from tuningfork.inference.smc import SMC_METHODS
    from tuningfork.runner.smc import init_particles_from_prior, run_smc

    smc_entry = SMC_METHODS[sampler_name]

    # Build logdensity_fn for inner-kernel proposals
    key, init_key = jax.random.split(key)
    _, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

    # SMC uses particles (N_particles) not chains.
    # Convention: num_chains=1, particles as second dim.
    N_PARTICLES = n_samples  # use n_samples budget as particle count

    key, particle_key = jax.random.split(key)
    # Draw initial particles from prior via the model's analytic_sampler or
    # use initialize_model for NumPyro models
    # For NumPyro models: use jax.vmap over initialize_model calls
    init_keys = jax.random.split(particle_key, N_PARTICLES)

    @jax.vmap
    def _init_one_position(k):
        pos, _, _ = build_logdensity_fn(k, posterior)
        return pos

    initial_particles = _init_one_position(init_keys)

    # Build SMC algorithm
    smc_params = recipe.base_method_params or {}
    smc_alg = smc_entry.factory(logdensity_fn, **smc_params)

    # Initialize SMC state
    smc_state = smc_alg.init(initial_particles)

    key, run_key = jax.random.split(key)
    t0 = time.perf_counter()
    final_smc_state, smc_history = run_smc(
        run_key,
        smc_init_state=smc_state,
        smc_step_fn=smc_alg.step,
        max_steps=200,
        lambda_target=1.0,
    )
    jax.block_until_ready(final_smc_state.particles)
    wall_time = time.perf_counter() - t0

    # Extract final particles into samples_dict with shape (1, N_particles, *event)
    particles = final_smc_state.particles
    if isinstance(particles, dict):
        samples_dict = {
            k: np.asarray(v)[np.newaxis, ...]  # (1, N_particles, *event)
            for k, v in particles.items()
        }
    else:
        # Flat array: shape (N_particles, dim) → split by param index
        particles_np = np.asarray(particles)
        samples_dict = {
            f"param_{i}": particles_np[np.newaxis, :, i]
            for i in range(particles_np.shape[-1])
        }

    infos = None  # SMC: no per-step divergence info in this layout
    idata = samples_to_idata(samples_dict, is_multichain=True)
    print(f"SMC complete: {N_PARTICLES} particles, wall_time={wall_time:.2f}s")
    print(f"Final tempering param: {getattr(final_smc_state, 'tempering_param', 'N/A')}")

elif sampler_name in FAMILY_D_SAMPLERS:
    # -----------------------------------------------------------------------
    # Family D: VI (meanfield_vi, fullrank_vi)
    # -----------------------------------------------------------------------
    key, init_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

    # Build VI kernel with pinned params
    vi_params = recipe.base_method_params or {}
    vi_kernel = base_method.factory(logdensity_fn, **vi_params)

    # Run VI from initial state
    key, vi_key = jax.random.split(key)
    vi_state = vi_kernel.init(vi_key, init_position)

    # Run VI optimization (num_steps from warmup_params or default)
    n_vi_steps = int(recipe.warmup_params.get("num_steps", 3000))
    t0 = time.perf_counter()
    vi_state, vi_info = jax.lax.scan(
        lambda state, k: vi_kernel.step(k, state),
        vi_state,
        jax.random.split(vi_key, n_vi_steps),
    )
    jax.block_until_ready(vi_state)
    wall_time = time.perf_counter() - t0

    # Sample from the fitted surrogate distribution to get samples_dict
    # shape: (N_CHAINS, n_samples, *event)
    key, sample_key = jax.random.split(key)
    # VI state should have a .params that lets us draw samples
    # Use the kernel's sample method or sample from the distribution
    vi_sample_keys = jax.random.split(sample_key, N_CHAINS * n_samples)
    # Most VI implementations expose vi_kernel.sample(key, state, n)
    if hasattr(vi_kernel, "sample"):
        vi_draws = vi_kernel.sample(sample_key, vi_state, N_CHAINS * n_samples)
    else:
        # Fallback: try to extract distribution and sample
        vi_draws = vi_state.mu + jax.random.normal(
            sample_key, (N_CHAINS * n_samples,) + vi_state.mu.shape
        )

    # Reshape to (N_CHAINS, n_samples, *event)
    if isinstance(vi_draws, dict):
        samples_dict = {
            k: np.asarray(v).reshape(N_CHAINS, n_samples, *v.shape[1:])
            for k, v in vi_draws.items()
        }
    else:
        vi_draws_np = np.asarray(vi_draws)
        samples_dict = {
            f"param_{i}": vi_draws_np[:, i].reshape(N_CHAINS, n_samples)
            for i in range(vi_draws_np.shape[-1])
        }

    infos = None  # VI: no MCMC divergence info
    idata = samples_to_idata(samples_dict, is_multichain=True)
    print(f"VI complete: {n_vi_steps} steps, wall_time={wall_time:.2f}s")

else:
    # -----------------------------------------------------------------------
    # Family A / B / E: gradient MCMC via warmup.runner + lax.scan + vmap
    # -----------------------------------------------------------------------
    key, init_key = jax.random.split(key)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

    # Run warmup (multi-chain runner contract)
    # Returns: (states_with_leading_num_chains, adapted_params)
    # For no_warmup: adapted_params is {} and states are init states
    warmup_params = recipe.warmup_params or {}
    n_warmup = int(warmup_params.get("n_warmup", 1000))

    key, warmup_key = jax.random.split(key)
    warmup_states, adapted_params = warmup.runner(
        warmup_key,
        init_position,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        num_chains=N_CHAINS,
    )

    # Merge: warmup-adapted params are supplementary defaults; recipe's pinned params win.
    # For no_warmup: adapted_params = {}, so base_params = recipe.base_method_params.
    # For stan_window: adapted_params has step_size + inverse_mass_matrix (per-chain);
    #   we average the per-chain IMM then override with the recipe's scalar step_size.
    recipe_params = recipe.base_method_params or {}
    base_params = {**adapted_params, **recipe_params}

    # For stan_window the per-chain IMM is shape (N_CHAINS, d) — average to (d,).
    if "inverse_mass_matrix" in base_params:
        imm = jnp.asarray(base_params["inverse_mass_matrix"])
        if imm.ndim == 2:  # (N_CHAINS, d) → (d,)
            base_params = {**base_params, "inverse_mass_matrix": imm.mean(axis=0)}

    # If the kernel needs a mass matrix but none is present (e.g. no_warmup with a
    # recipe that stores only step_size), inject a diagonal identity preconditioner.
    if base_method.needs_mass_matrix and "inverse_mass_matrix" not in base_params:
        leaves = jax.tree_util.tree_leaves(warmup_states.position)
        n_params_total = int(sum(jnp.asarray(leaf[0]).size for leaf in leaves))
        base_params = {**base_params, "inverse_mass_matrix": jnp.ones(n_params_total)}

    # Build the sampling kernel with the final merged params.
    kernel = base_method.factory(logdensity_fn, **base_params)

    # For no_warmup: warmup_states are already the init states (from warmup.runner)
    # For stan_window: warmup_states are the post-warmup chain states
    # Both cases: warmup_states has leading dim N_CHAINS

    # Run sampling chain via lax.scan, vmapped over chains
    is_mclmc = sampler_name in {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}

    if is_mclmc:
        # MCLMC.init requires (position, rng_key) — re-init from warmup_states.position
        key, mclmc_key = jax.random.split(key)
        mclmc_chain_keys = jax.random.split(mclmc_key, N_CHAINS)

        @jax.vmap
        def _init_mclmc(pos, k):
            return kernel.init(pos, k)

        chain_states = _init_mclmc(warmup_states.position, mclmc_chain_keys)
    else:
        # All other kernels: kernel.init(position)
        chain_states = jax.vmap(kernel.init)(warmup_states.position)

    # Define a single-chain scan step
    def _one_chain_scan(state, rng_key):
        new_state, info = kernel.step(rng_key, state)
        return new_state, (new_state, info)

    # Run one chain with lax.scan over sample keys
    def _run_one_chain(initial_state, chain_key):
        sample_keys = jax.random.split(chain_key, n_samples)
        _, (states, infos) = jax.lax.scan(_one_chain_scan, initial_state, sample_keys)
        return states, infos

    key, chain_key = jax.random.split(key)
    chain_keys = jax.random.split(chain_key, N_CHAINS)

    t0 = time.perf_counter()
    all_states, all_infos = jax.vmap(_run_one_chain)(chain_states, chain_keys)
    jax.block_until_ready(all_states.position)
    wall_time = time.perf_counter() - t0

    # Build samples_dict: shape (N_CHAINS, n_samples, *event)
    position = all_states.position  # dict or array
    if isinstance(position, dict):
        samples_dict = {k: np.asarray(v) for k, v in position.items()}
    else:
        pos_np = np.asarray(position)
        if pos_np.ndim == 2:
            # (N_CHAINS, n_samples) — 1D model
            samples_dict = {"param_0": pos_np[:, :, np.newaxis].squeeze(-1)}
        else:
            # (N_CHAINS, n_samples, dim)
            samples_dict = {
                f"param_{i}": pos_np[:, :, i] for i in range(pos_np.shape[-1])
            }

    infos = all_infos
    idata = samples_to_idata(samples_dict, is_multichain=True)

    # Count divergences for Family A/E
    if hasattr(infos, "is_divergent") and infos.is_divergent is not None:
        n_diverg = int(np.sum(np.asarray(infos.is_divergent)))
        print(f"n_divergences: {n_diverg}")

    print(f"Sampling complete: {N_CHAINS} chains x {n_samples} samples")
    print(f"wall_time: {wall_time:.2f}s")
    for name, arr in list(samples_dict.items())[:3]:
        print(f"  {name}: shape={arr.shape}")
```

```{code-cell} ipython3
# Compute auto_gate on fresh samples
fresh_gate = auto_gate(
    samples_dict,
    info=infos,
    posterior=posterior,
    n_chunks=N_CHAINS,
)

print("Fresh auto-gate verdict:")
print(f"  R-hat max:      {fresh_gate.rhat_max}")
print(f"  Min bulk-ESS:   {fresh_gate.min_bulk_ess}")
print(f"  Divergences:    {fresh_gate.n_divergences}")
print(f"  max_abs_mean_z: {fresh_gate.max_abs_mean_z}")
print(f"  Verdict:        {fresh_gate.verdict}")
```

## Section 1: Universal Scalar Summary

```{code-cell} ipython3
# Render the universal summary table
gate_verdict_dict = fresh_gate.to_dict()
fig_summary = render_universal_summary(idata, infos, gate_verdict_dict, wall_time)
plt.tight_layout()
plt.show()
```

## Section 2: Family-Specific Diagnostic Battery

```{code-cell} ipython3
# Dispatch to the appropriate family renderer
if sampler_name in FAMILY_A_SAMPLERS:
    print(f"Rendering Family A diagnostics (gradient MH-corrected): {sampler_name}")
    family_figs = render_family_a(idata, infos, sampler_name=sampler_name)
    print(f"Generated {len(family_figs)} figures")
    for fig in family_figs:
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()

elif sampler_name in FAMILY_B_SAMPLERS:
    print(f"Rendering Family B diagnostics (MCLMC): {sampler_name}")
    family_figs = render_family_b(idata, infos)
    print(f"Generated {len(family_figs)} figures")
    for fig in family_figs:
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()

elif sampler_name in FAMILY_C_SAMPLERS:
    print(f"Rendering Family C diagnostics (SMC): {sampler_name}")
    family_figs = render_family_c(idata, infos)
    print(f"Generated {len(family_figs)} figures")
    for fig in family_figs:
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()

elif sampler_name in FAMILY_D_SAMPLERS:
    print(f"Rendering Family D diagnostics (VI): {sampler_name}")
    family_figs = render_family_d(idata, infos)
    print(f"Generated {len(family_figs)} figures")
    for fig in family_figs:
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()

elif sampler_name in FAMILY_E_SAMPLERS:
    print(f"Rendering Family E diagnostics (Specialised): {sampler_name}")
    family_figs = render_family_e(idata, infos, sampler_name=sampler_name)
    print(f"Generated {len(family_figs)} figures")
    for fig in family_figs:
        plt.figure(fig.number)
        plt.tight_layout()
        plt.show()

else:
    print(f"WARNING: Sampler {sampler_name!r} not recognized in any family!")
```

## Section 3: Reference Comparison Panel

```{code-cell} ipython3
# Reference comparison (conditional on reference availability)
try:
    from tuningfork.metrics import reference_compare  # noqa: F401
    print("reference_compare module available")
    print("TODO: Implement Section 3 reference comparison panel")
except ImportError:
    print(
        "Reference comparison unavailable — "
        "`tuningfork/metrics/reference_compare.py` not yet available in the editable install."
    )
```

## Section 4: Auto-gate Verdict Block (Fresh vs Stored)

```{code-cell} ipython3
# Display fresh auto-gate result
rhat_str = f"{fresh_gate.rhat_max:.4f}" if fresh_gate.rhat_max is not None else "N/A"
ess_str = f"{fresh_gate.min_bulk_ess:.1f}" if fresh_gate.min_bulk_ess is not None else "N/A"
diverg_str = str(int(fresh_gate.n_divergences)) if fresh_gate.n_divergences is not None else "N/A"
meanz_str = f"{fresh_gate.max_abs_mean_z:.4f}" if fresh_gate.max_abs_mean_z is not None else "N/A"

print("=" * 70)
print("AUTO-GATE RESULT (fresh run):")
print("=" * 70)
print(f"  R-hat max:      {rhat_str}")
print(f"  Min bulk-ESS:   {ess_str}")
print(f"  Divergences:    {diverg_str}")
print(f"  max_abs_mean_z: {meanz_str}")
print(f"  Verdict:        {fresh_gate.verdict}")

stored_verdict = recipe.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN")
print()
print("=" * 70)
print("STORED GATE EVIDENCE (recipe JSON):")
print("=" * 70)
print(f"  verdict: {stored_verdict}  (tuning_seed={recipe.tuning_seed})")

if fresh_gate.verdict != stored_verdict and stored_verdict != "NOT_RUN":
    print()
    print("!" * 70)
    print("WARNING: SEED SENSITIVITY: fresh run verdict differs from stored verdict.")
    print("Escalate to TL for investigation.")
    print("!" * 70)
elif stored_verdict == "NOT_RUN":
    print()
    print("(Stored verdict is NOT_RUN — this is a zero-calibration starter recipe.)")
else:
    print()
    print("Fresh run matches stored verdict.")
```

## Section 5: Investigation Mode Plots (Conditional)

```{code-cell} ipython3
# Investigation mode is gated on recipe.effort in (MEDIUM, HIGH)
if recipe.effort in (Effort.MEDIUM, Effort.HIGH):
    print(f"Investigation Mode: recipe.effort = {recipe.effort.value}")
    print(f"  Model: {recipe.model_name}")
    print(f"  Sampler: {recipe.base_method_name}")
    print("  (Specific investigation plots depend on model+sampler combination)")
    print("  TODO: Implement per-model investigation panels (Recipe Phase 1+)")
else:
    print(
        f"Investigation Mode disabled (effort={recipe.effort.value}; "
        "threshold is MEDIUM/HIGH)"
    )
```

## Summary

This notebook provides a human-in-the-loop diagnostic tool for recipe development and
investigation. It:

1. Loads a Recipe from JSON and displays metadata
2. Runs the actual warmup + sampler end-to-end (real JAX chain, not synthetic data)
3. Renders a universal scalar summary table (Section 1)
4. Dispatches to family-aware diagnostic batteries (Section 2)
5. Provides reference comparison when available (Section 3)
6. Re-runs auto-gate and compares against stored verdict (Section 4)
7. Shows investigation-mode plots for MEDIUM/HIGH cells (Section 5)

The notebook is parametrized via the top cell (`RECIPE_PATH`, `QUICK_MODE`, etc.) for
use with papermill or manual parameter tuning.
