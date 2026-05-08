"""HMC algorithm entry for the bjx-bench algorithm registry.

Fixed-L Hamiltonian Monte Carlo.  Both ``step_size`` and
``num_integration_steps`` are BO-tunable; ``inverse_mass_matrix`` comes from
window adaptation warmup.

Grad cost per step: ``info.num_integration_steps`` (1 gradient per leapfrog
step).  Optimal target acceptance rate ≈ 0.65 (Beskos et al. 2013).
"""

from __future__ import annotations

import blackjax
import jax.numpy as jnp

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="hmc",
    family="mcmc",
    factory=blackjax.hmc,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=128),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.65,
    notes=(
        "Beskos et al. optimal accept ≈ 0.65 for fixed-L HMC; both step_size "
        "and num_integration_steps are BO-tunable. inverse_mass_matrix comes "
        "from warmup adaptation, not BO."
    ),
)
