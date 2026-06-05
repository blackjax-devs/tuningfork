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
"""Additive Step Random Walk MH algorithm entry for the tuningfork registry.

Wraps ``blackjax.additive_step_random_walk`` (a ``GenerateSamplingAPI`` wrapping
``blackjax.mcmc.random_walk.additive_step_random_walk``).

The Additive Step RMH kernel adds a user-supplied ``proposal_generator`` step
to the current position: ``new_position = position + proposal_generator(key, position)``.
The proposal must be symmetric to maintain detailed balance.

This is a **specialised** entry (``extra_required_kwargs=("proposal_generator",)``):
the factory requires a ``proposal_generator`` callable beyond ``logdensity_fn`` and
the HP-space items.  The standard ``no_warmup`` path raises ``NotImplementedError``;
a specialised wiring path is required.

Hyperparameter-free: no BO-tunable scalar HPs (``default_hp_space=()``).  The
proposal distribution is a full callable that encodes its own scale parameters.

Gradient-free: ``grad_count_per_step=0``.
``target_acceptance_rate=None``: depends on proposal-vs-target overlap.

References
----------
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). Equation of state calculations by fast computing machines.
  *The Journal of Chemical Physics*, 21(6), 1087–1092.
- See also ``blackjax.additive_step_random_walk.normal_random_walk`` for the
  registered Gaussian special-case (line 122 of ``blackjax/__init__.py``).
"""

from collections.abc import Callable

import blackjax
import jax.numpy as jnp

from tuningfork.base_method._base import BaseMethod

__all__ = ["ENTRY", "_factory"]


def _factory(
    logdensity_fn,
    *,
    proposal_generator: Callable,
    **kwargs,
):
    """Build a ``blackjax.additive_step_random_walk`` kernel.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density (log-posterior) callable ``x -> float``.
    proposal_generator
        Callable ``(rng_key, position) -> step`` that produces an additive
        step.  The step is added to the current position to form the proposal:
        ``new_position = position + proposal_generator(key, position)``.
        Must be symmetric: ``P(step | position) = P(-step | position + step)``.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
        The returned state is ``RWState(position, logdensity)``.
        The info is ``RWInfo(acceptance_rate, is_accepted, proposal)``.
    """
    return blackjax.additive_step_random_walk(logdensity_fn, proposal_generator)


ENTRY = BaseMethod(
    name="additive_step_random_walk",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(0),  # gradient-free MH
    grad_count_convention="0 (gradient-free)",
    default_hp_space=(),  # HP-free; proposal_generator encodes its own scale
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # depends entirely on proposal-vs-target overlap
    extra_required_kwargs=("proposal_generator",),
    # T2.3 descriptors: gradient-free, no adapted step_size/imm from warmup.
    per_chain_param_keys=(),  # no_warmup returns empty batched_params.
    reinit_state=False,  # RWState from .init() is directly usable.
    extra_kwarg_builder=None,  # proposal_generator is injected by the caller
    # (model-specific, not a portable descriptor builder).
    notes=(
        "Additive Step Random Walk MH (Metropolis et al. 1953). The proposal adds "
        "a user-supplied step to the current position: new_pos = pos + proposal_generator(key, pos). "
        "The proposal_generator must be symmetric (P(step|pos) = P(-step|pos+step)). "
        "Specialised: extra_required_kwargs=('proposal_generator',); no_warmup raises "
        "NotImplementedError; a specialised wiring path is required. "
        "Gradient-free (grad_count_per_step=0). RWInfo carries acceptance_rate, is_accepted, proposal. "
        "Note: blackjax.additive_step_random_walk is a GenerateSamplingAPI instance with a registered "
        "normal_random_walk factory (line 122 of blackjax/__init__.py) for the Gaussian special case. "
        "The runner should supply proposal_generator from Posterior metadata or recipe parameters."
    ),
)
