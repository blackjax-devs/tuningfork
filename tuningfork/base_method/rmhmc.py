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
"""RMHMC algorithm entry for the tuningfork algorithm registry.

Wraps ``blackjax.rmhmc`` (``blackjax.mcmc.rmhmc.as_top_level_api``).

RMHMC is an alias of the HMC kernel with a different default integrator:
``implicit_midpoint`` instead of ``velocity_verlet``.  The implicit midpoint
integrator is more appropriate for non-separable Hamiltonians (e.g. when the
mass matrix is a function of position, making the kinetic energy position-
dependent).

Upstream signature: ``blackjax.rmhmc(logdensity_fn, step_size, mass_matrix,
num_integration_steps, *, divergence_threshold=1000, integrator=implicit_midpoint)``

CRITICAL — ``mass_matrix`` vs ``inverse_mass_matrix``:
  The rmhmc upstream takes ``mass_matrix`` (NOT ``inverse_mass_matrix``).
  The tuningfork runner and ``window_adaptation`` both produce an
  ``inverse_mass_matrix``.  This wrapper converts at the factory boundary:

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

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]


def _factory(
    logdensity_fn,
    *,
    step_size: float,
    inverse_mass_matrix,
    num_integration_steps: int = 5,
    **kwargs,
):
    """Build a ``blackjax.rmhmc`` kernel, converting IMM to mass_matrix.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density (log-posterior) callable ``x -> float``.
    step_size
        Symplectic integrator step size.
    inverse_mass_matrix
        Inverse mass matrix from warmup adaptation.  Shape ``(d,)`` for
        diagonal, ``(d, d)`` for dense.  Converted to ``mass_matrix`` before
        passing to ``blackjax.rmhmc``.
    num_integration_steps
        Number of implicit_midpoint integrator steps per sample.  Default 5.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.

    Notes
    -----
    The IMM→mass_matrix conversion:
    - 1-D (diagonal) IMM: ``mass_matrix = 1.0 / inverse_mass_matrix``
    - 2-D (dense) IMM: ``mass_matrix = jnp.linalg.inv(inverse_mass_matrix)``
    This conversion is exact for diagonal and dense positive-definite matrices.
    """
    if inverse_mass_matrix.ndim == 1:
        mass_matrix = 1.0 / inverse_mass_matrix
    else:  # 2-D dense
        mass_matrix = jnp.linalg.inv(inverse_mass_matrix)

    return blackjax.rmhmc(
        logdensity_fn,
        step_size=step_size,
        mass_matrix=mass_matrix,
        num_integration_steps=int(num_integration_steps),
        **{k: v for k, v in kwargs.items() if k == "divergence_threshold"},
    )


ENTRY = BaseMethod(
    name="rmhmc",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(info.num_integration_steps),
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
        "CRITICAL IMM→mass_matrix conversion at factory boundary: window_adaptation "
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
        "window_adaptation compatibility: the wrapper drives window_adaptation correctly "
        "via the IMM→mass_matrix conversion. Direct use of blackjax.rmhmc with "
        "window_adaptation requires passing blackjax.rmhmc (the GenerateSamplingAPI object) "
        "as the algorithm — window_adaptation calls algorithm.build_kernel(integrator) "
        "directly and bypasses the factory wrapper."
    ),
)
