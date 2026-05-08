"""MCLMC (Microcanonical Langevin Monte Carlo) algorithm wrapper.

Per the corrected PLAN_bjx_bench_API_phase2.md §"Grad costs": each MCLMC step
performs one integrator step. The default integrator (isokinetic_mclachlan)
costs 2 gradient evaluations per integrator step (palindromic [b1,a1,b2,a1,b1]
scheme → 2 position updates) → constant 2 grads per kernel step.
MCLMCInfo does NOT carry num_integration_steps; the trajectory length L (in
time units) controls momentum-resample cadence (L / step_size time units).

Init note: blackjax.mclmc.init requires an rng_key to generate the initial
unit-vector momentum. Call kernel.init(position, rng_key) rather than the
rng_key-free form used by HMC/MALA/Barker.

Requires pytree_size(position) >= 2 (enforced by blackjax upstream).

Adaptation: BlackJAX provides blackjax.mclmc_find_L_and_step_size as a
dedicated warmup routine. Tier-B (T2.6) will dispatch to it based on
AlgorithmEntry.name. T2.4 only declares the entry.
"""

from __future__ import annotations

import blackjax
import jax.numpy as jnp

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = AlgorithmEntry(
    name="mclmc",
    family="mcmc",
    factory=blackjax.mclmc,  # signature: (logdensity_fn, L, step_size, ...)
    grad_count_per_step=lambda info: jnp.asarray(2),
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # rejection-free; not applicable
    notes=(
        "Constant 2 grads/step (default isokinetic_mclachlan integrator). "
        "MCLMCInfo._fields = ('logdensity', 'kinetic_change', 'energy_change', 'nonans'); "
        "no num_integration_steps field. "
        "inverse_mass_matrix=1.0 default is scalar (global preconditioner). "
        "init requires rng_key: kernel.init(position, rng_key). "
        "Dedicated adaptation: blackjax.mclmc_find_L_and_step_size — not window_adaptation. "
        "pytree_size(position) >= 2 required by upstream."
    ),
)
