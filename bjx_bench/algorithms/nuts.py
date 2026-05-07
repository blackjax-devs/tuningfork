"""NUTS algorithm entry for the bjx-bench algorithm registry.

NUTS (No-U-Turn Sampler) is the default reference sampler in BlackJAX.
The only BO-tunable hyperparameter is ``step_size``; the
``inverse_mass_matrix`` comes from window adaptation warmup and is not
searched by Bayesian optimisation.

Grad cost per step: ``info.num_integration_steps`` (1 gradient per
leapfrog step, same accounting as HMC).
"""

from __future__ import annotations

import blackjax
import jax.numpy as jnp

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = AlgorithmEntry(
    name="nuts",
    family="mcmc",
    factory=blackjax.nuts,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.80,
    notes=(
        "Stan-default target acceptance 0.80; inverse_mass_matrix supplied "
        "by warmup adaptation, not BO. step_size is the only BO-tunable HP."
    ),
)
