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
"""Descriptor for RMHMC recipe emission.

Generated emission targets ``blackjax.rmhmc``
(``blackjax.mcmc.rmhmc.as_top_level_api``).

RMHMC is an alias of the HMC kernel with a different default integrator:
``implicit_midpoint`` instead of ``velocity_verlet``.  The implicit midpoint
integrator is more appropriate for non-separable Hamiltonians (e.g. when the
mass matrix is a function of position, making the kinetic energy position-
dependent).

Upstream signature: ``blackjax.rmhmc(logdensity_fn, step_size, mass_matrix,
num_integration_steps, *, divergence_threshold=1000, integrator=implicit_midpoint)``

CRITICAL — ``mass_matrix`` vs ``inverse_mass_matrix``:
  The rmhmc upstream takes ``mass_matrix`` (NOT ``inverse_mass_matrix``).
  Generated recipes and window adaptation provide an
  ``inverse_mass_matrix``.  The generated emitter converts it before calling
  ``blackjax.rmhmc``:

  - Diagonal case (1-D array): ``mass_matrix = 1.0 / inverse_mass_matrix``
  - Dense case (2-D array):   ``mass_matrix = jnp.linalg.inv(inverse_mass_matrix)``

  When ``mass_matrix`` is a constant array, rmhmc behaves as ordinary HMC but
  with the implicit_midpoint integrator (per upstream docstring line 40-43).
  The Riemannian metric-callable mode (``mass_matrix`` as a position-dependent
  function) is a future scope item via a future
  ``extra_required_kwargs=("mass_matrix_fn",)`` schema branch.

State / Info types:
  rmhmc reuses ``blackjax.mcmc.hmc.HMCState`` and ``blackjax.mcmc.hmc.HMCInfo``
  (it calls ``hmc.init`` and ``hmc.build_kernel`` directly).
  ``HMCState._fields = ('position', 'logdensity', 'logdensity_grad')``.
  ``HMCInfo._fields = ('momentum', 'acceptance_rate', 'is_accepted',
  'is_divergent', 'energy', 'proposal', 'num_integration_steps')``.

Integrator note:
  The default ``implicit_midpoint`` integrator requires solving an implicit
  equation per step (via fixed-point iteration), making each step slower than
  the explicit ``velocity_verlet`` used by vanilla HMC.  This cost is warranted
  when the Hamiltonian is non-separable (e.g. position-dependent mass matrix).
  In constant-mass-matrix mode, ``velocity_verlet`` would be more efficient;
  ``implicit_midpoint`` is kept for consistency with upstream defaults.

Hyperparameters:
  - ``step_size`` (loguniform [1e-3, 1.0]): integrator step size.
  - ``num_integration_steps`` (int [1, 20]): leapfrog steps per sample.
  - ``inverse_mass_matrix`` comes from warmup adaptation (``needs_mass_matrix=True``).

Target acceptance rate: 0.8 (higher than HMC's 0.65 per upstream recommendation
for implicit-midpoint integrators).

Grad cost per step: ``info.num_integration_steps`` (same as HMC — one gradient
evaluation per integrator step, but the implicit_midpoint integrator uses fixed-
point iteration, so actual wall-clock cost per grad is higher than velocity_verlet).

References
----------
- Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and
  Hamiltonian Monte Carlo methods. *JRSS-B*, 73(2), 123-214.
- BlackJAX upstream: ``blackjax/mcmc/rmhmc.py`` (reuses HMC kernel).
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="rmhmc",
    family="mcmc",
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
    grad_count_convention="info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    notes=(
        "Riemannian Manifold HMC (Girolami & Calderhead 2011). "
        "Upstream: blackjax.mcmc.rmhmc reuses hmc.init + hmc.build_kernel with "
        "implicit_midpoint integrator default (vs velocity_verlet for vanilla HMC). "
        "CRITICAL IMM→mass_matrix conversion in generated emission: window_adaptation "
        "adapts inverse_mass_matrix; rmhmc upstream takes mass_matrix (NOT IMM). "
        "Diagonal (1-D IMM): mass_matrix = 1.0 / inverse_mass_matrix. "
        "Dense (2-D IMM): mass_matrix = jnp.linalg.inv(inverse_mass_matrix). "
        "Constant-mass-matrix mode = HMC + implicit_midpoint integrator (per upstream "
        "docstring: 'simply an alias of the hmc kernel with a different choice of "
        "default integrator'). The Riemannian metric-callable mode (mass_matrix as a "
        "position-dependent function) is future work via a future "
        "extra_required_kwargs=('mass_matrix_fn',) schema branch. "
        "State/Info types: rmhmc reuses HMCState + HMCInfo (no distinct NamedTuples). "
        "HMCState._fields = ('position', 'logdensity', 'logdensity_grad'). "
        "HMCInfo._fields = ('momentum', 'acceptance_rate', 'is_accepted', "
        "'is_divergent', 'energy', 'proposal', 'num_integration_steps'). "
        "Implicit_midpoint integrator: slower per step than velocity_verlet (fixed-point "
        "iteration per step) but better for non-separable Hamiltonians. "
        "HP-space: step_size loguniform [1e-3, 1.0], num_integration_steps int [1, 20]. "
        "target_acceptance_rate=0.8 (higher than HMC's 0.65, appropriate for "
        "implicit_midpoint integrator). "
        "Generated emission applies the same IMM→mass_matrix conversion for both "
        "adapted and no-warmup paths."
    ),
)
