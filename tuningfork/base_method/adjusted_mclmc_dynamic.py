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
"""Dynamic Adjusted MCLMC — stochastic trajectory length variant.

This entry wraps ``blackjax.adjusted_mclmc_dynamic``, which draws the number
of integration steps from a distribution at each step rather than fixing it.
``integration_steps_fn`` and ``integration_steps_params`` together encode the
trajectory-length distribution:

- ``integration_steps_fn = make_random_trajectory_length_fn(True)``
  has signature ``(rng_arg, avg) -> int``, sampling uniformly around ``avg``.
- ``integration_steps_params = (avg_num_integration_steps,)``
  carries the adapted average from warmup.

The BO hyperparameter space ``(step_size, L)`` matches vanilla MCLMC for
consistency.  The adapter translates ``L`` to an average number of steps via
``avg = max(1.0, L / step_size)``.

Grad cost: same formula as static adjusted_mclmc — the default integrator
(isokinetic_mclachlan) evaluates 2 grads per integrator step.  For dynamic
trajectories, ``info.num_integration_steps`` is the realized random count per
kernel call.

Init: ``blackjax.adjusted_mclmc_dynamic.init(position, logdensity_fn, rng_key)``
requires an rng_key for the random_generator_arg.  The top-level
``SamplingAlgorithm`` wrapper has ``pass_rng_key_to_init=True``, so
``algo.init(position, rng_key=key)`` works at the user level.

Adaptation: ``blackjax.adjusted_mclmc_find_L_and_step_size`` with
``blackjax.mcmc.adjusted_mclmc.build_kernel()`` (not the dynamic variant).
The adapted (L, step_size, IMM) values are then wired into this factory for
actual sampling.
"""

import blackjax
import jax.numpy as jnp
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

from tuningfork.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY"]

# Module-level function so it is shared across factory calls (not recreated each time).
_steps_fn = make_random_trajectory_length_fn(True)  # (rng_arg, avg) -> int


def _factory(logdensity_fn, *, step_size, L, inverse_mass_matrix=1.0, **kwargs):
    """Build a BlackJAX adjusted_mclmc_dynamic SamplingAlgorithm.

    Translates the ``(step_size, L)`` BO hyperparameter space into
    ``integration_steps_params=(avg,)`` where ``avg = max(1.0, L / step_size)``.

    Parameters
    ----------
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size
        Leapfrog step size.
    L
        Target trajectory length in time units.  Converted to an average
        number of integration steps via ``avg = max(1.0, L / step_size)``.
    inverse_mass_matrix
        Diagonal preconditioning matrix (scalar or 1-D array).
        Default ``1.0`` (identity preconditioning).
    **kwargs
        Ignored; present for interface uniformity with the runner.

    Returns
    -------
    blackjax.SamplingAlgorithm
        Object with ``.init`` (requires ``rng_key``) and ``.step`` methods.
    """
    # BO tuning supplies concrete float trial values; trace-safe.
    avg = max(1.0, float(L) / float(step_size))
    return blackjax.adjusted_mclmc_dynamic(
        logdensity_fn,
        step_size=step_size,
        integration_steps_fn=_steps_fn,
        integration_steps_params=(avg,),
        inverse_mass_matrix=inverse_mass_matrix,
    )


ENTRY = BaseMethod(
    name="adjusted_mclmc_dynamic",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(2 * info.num_integration_steps),
    grad_count_convention="2 × info.num_integration_steps",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("L", "loguniform", low=0.1, high=100.0),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.9,
    notes=(
        "Dynamic Metropolis-adjusted MCLMC (adjusted_mclmc_dynamic). "
        "Factory translates (step_size, L) -> integration_steps_params=(avg,) "
        "where avg = max(1.0, L / step_size); integration_steps_fn samples "
        "uniformly around avg via make_random_trajectory_length_fn(True). "
        "grad_count_per_step = 2 * info.num_integration_steps "
        "(isokinetic_mclachlan default integrator: 2 grads/step; realized count). "
        "init: blackjax.adjusted_mclmc_dynamic.init(position, logdensity_fn, rng_key) "
        "— rng_key required for random_generator_arg. "
        "algo.init(position, rng_key=key) works via pass_rng_key_to_init=True. "
        "Adaptation: adjusted_mclmc_find_L_and_step_size (static kernel) with target=0.9; "
        "adapted params wired into this factory for sampling. "
        "needs_mass_matrix=True: IMM from adjusted_mclmc_find_L_and_step_size. "
        "target_acceptance_rate=0.9: canonical adjusted-MCLMC acceptance target."
    ),
)
