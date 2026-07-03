# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GHMC (Generalized HMC / persistent-momentum HMC) algorithm entry for the
tuningfork algorithm registry.

Generalized HMC (Horowitz 1991; Sohl-Dickstein et al. 2014) extends standard
HMC with a **momentum persistence** parameter ``alpha``.  At each step, the
existing momentum is partially retained (``alpha`` ∈ [0, 1]) rather than
refreshed from scratch.  ``alpha=0`` recovers standard HMC; ``alpha`` near 1
gives near-persistent dynamics with strong autocorrelation but reduced
exploration.  The slice-sampling parameter ``delta`` controls the acceptance
criterion for the generalized Metropolis step.  GHMC consistently outperforms
standard HMC on high-dimensional problems with strong posterior correlations,
where partial momentum refresh allows larger effective step sizes without
rejection spikes.

GHMC does exactly **1 leapfrog step per kernel call** (no trajectory length
hyperparameter), so ``grad_count_per_step`` is the constant ``1``.  The
inverse mass matrix (``momentum_inverse_scale``) is warmup-derived (MEADS);
the three BO-tunable hyperparameters are ``step_size``, ``alpha``, and
``delta``.

References
----------
- Horowitz, A. M. (1991). A generalized guided Monte Carlo algorithm.
  *Physics Letters B*, 268(2), 247–252.
- Sohl-Dickstein, J., Mudigonda, M., & DeWeese, M. (2014). Hamiltonian Monte
  Carlo without detailed balance. In *ICML 2014*.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

ENTRY = BaseMethod(
    name="ghmc",
    family="mcmc",
    factory=blackjax.ghmc,  # called as factory(logdensity_fn, **trial_params)
    grad_count_per_step=lambda info: jnp.asarray(1),  # 1 leapfrog per step (constant)
    grad_count_convention="1 (one leapfrog per step, constant)",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("alpha", "uniform", low=0.0, high=1.0),
        HyperparamSpace("delta", "uniform", low=0.0, high=1.0),
    ),
    needs_mass_matrix=True,  # momentum_inverse_scale comes from MEADS, not BO
    imm_kwarg_name="momentum_inverse_scale",  # blackjax.ghmc's own factory kwarg name;
    # no inverse_mass_matrix parameter at all, no **kwargs catch-all. See
    # BaseMethod.imm_kwarg_name docstring for the single-source-of-truth rationale.
    target_acceptance_rate=0.65,  # Beskos et al. optimal ≈ 0.65 (same as HMC)
    # T2.3 descriptors: step_size + imm per-chain from MEADS warmup.
    # NOTE: "inverse_mass_matrix" here is the semantic category marker the
    # emit-script generator's _needs_imm() checks for (_emit/_sampler.py:71),
    # NOT the literal batched_params key or factory kwarg name -- that
    # translation is imm_kwarg_name above. Do not "fix" this to
    # "momentum_inverse_scale"; it would break _needs_imm() for ghmc.
    per_chain_param_keys=("step_size", "inverse_mass_matrix"),
    reinit_state=False,  # GHMCState from MEADS is directly usable by the sampling kernel.
    # (Note: the audit suggested reinit_state=True for ghmc, but the current runner
    # does NOT reinit ghmc — keeping False preserves behavior-identical semantics.)
    extra_kwarg_builder=None,  # No extra kwargs beyond logdensity_fn + HP-space.
    notes=(
        "Generalized HMC with persistent momentum (Horowitz 1991; "
        "Sohl-Dickstein et al. 2014). alpha ∈ [0,1]: alpha=0 ≡ standard HMC "
        "(full momentum refresh), alpha→1 ≡ near-persistent dynamics. "
        "delta is the slice-sampling acceptance parameter. "
        "1 leapfrog step per call; grad_count_per_step=1 (constant). "
        "momentum_inverse_scale from MEADS warmup, not BO. "
        "Outperforms HMC on high-dim posteriors with strong correlations."
    ),
)
