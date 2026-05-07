"""MALA algorithm entry for the bjx-bench algorithm registry.

Metropolis-Adjusted Langevin Algorithm.  Only ``step_size`` is BO-tunable;
no mass matrix is needed (MALA uses a fixed isotropic metric).

Grad cost per step: constant 1.  The MALAState caches ``logdensity_grad``
from the accepted proposal; a single ``value_and_grad`` call is made per
step to evaluate the candidate.  Optimal target acceptance rate ≈ 0.574
(Roberts & Rosenthal 1998).
"""

from __future__ import annotations

import blackjax
import jax.numpy as jnp

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = AlgorithmEntry(
    name="mala",
    family="mcmc",
    factory=blackjax.mala,  # called as factory(logdensity_fn, step_size=...)
    grad_count_per_step=lambda info: jnp.asarray(1),
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.574,
    notes=(
        "Roberts & Rosenthal '98 optimal accept ≈ 0.574 in high-D. "
        "Constant 1 grad/step (cached MALAState.logdensity_grad reused "
        "for MH ratio; single value_and_grad at proposal). "
        "No mass matrix required — isotropic Langevin dynamics."
    ),
)
