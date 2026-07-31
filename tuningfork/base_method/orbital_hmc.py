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
"""Descriptor for Orbital HMC recipe emission.

The upstream method is ``blackjax.orbital_hmc``
(``blackjax.mcmc.periodic_orbital.as_top_level_api``).

Each iteration of the Periodic Orbital MCMC outputs ``period`` weighted
samples from a single Hamiltonian orbit.  The orbit is built by applying
a symplectic integrator (default: velocity Verlet) ``period`` times; the
returned state holds ALL positions on the orbit (``PeriodicOrbitalState``).

Hyperparameters:
  - ``step_size`` (loguniform [1e-3, 1.0]): leapfrog step size.
  - ``period`` (int [2, 20]): number of orbit steps; also the number of
    weighted samples per state.  Smaller ``period`` = cheaper per call but
    fewer samples; larger ``period`` = more grad evals but richer geometry.
  - ``inverse_mass_matrix`` comes from warmup adaptation (``needs_mass_matrix=True``).

Grad cost per step: ``period`` gradient evaluations (one per orbit position).
Target acceptance rate: None (no MH step — orbital weights replace rejection).

Generated emission does not currently support this method. Enabling it requires
typed recipe inputs and a corresponding sampler emitter.

Note: ``PeriodicOrbitalState._fields = ('positions', 'weights', 'directions',
'logdensities', 'logdensities_grad')`` — the state carries the full orbit.

References
----------
- Neklyudov, K., & Welling, M. (2022). Orbital MCMC. *arXiv:2010.08047*.
"""

import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="orbital_hmc",
    family="mcmc",
    # period grad evals per step: the kernel builds a full orbit of `period`
    # positions, each requiring one gradient evaluation.
    # PeriodicOrbitalInfo does not carry a per-step grad count; we use `period`
    # as a constant.  The default period (from HP space) is used as a proxy
    # when the actual period is not accessible from info.
    # For a more accurate accounting, the emitter should read the period from
    # the state.directions.max() + 1 or from the recipe params.
    grad_count_per_step=lambda info: jnp.asarray(1),  # lower-bound: 1 grad/step
    grad_count_convention="1 (lower bound; OHO samples a full orbit, NIS ≥ 1)",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("period", "int", low=2, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=None,  # no MH step; orbital weights replace rejection
    notes=(
        "Periodic Orbital MCMC (Neklyudov & Welling 2022). Each iteration builds "
        "a Hamiltonian orbit of `period` positions and returns all as weighted samples. "
        "No MH rejection step — orbital weights (proportional to e^{logdensity - KE}) "
        "replace acceptance. PeriodicOrbitalState._fields = ('positions', 'weights', "
        "'directions', 'logdensities', 'logdensities_grad'): note plural fields since "
        "the state carries the full orbit of length `period`. "
        "Grad cost per step = `period` (one grad eval per orbit position). "
        "inverse_mass_matrix from warmup adaptation (needs_mass_matrix=True). "
        "HP-space: step_size loguniform [1e-3, 1.0], period int [2, 20]. "
        "The integrator defaults to velocity_verlet. Generated emission currently "
        "reports this method as unsupported; enabling it requires typed recipe "
        "inputs and a corresponding sampler emitter. "
    ),
)
