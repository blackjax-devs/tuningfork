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
"""IRMH (Independent Random-Walk Metropolis-Hastings) algorithm entry.

Wraps ``blackjax.irmh`` (``blackjax.mcmc.random_walk.irmh_as_top_level_api``)
for the tuningfork algorithm registry.

The proposal is a full ``Callable`` supplied at factory time — typically
fitted from a VI / Pathfinder / Laplace approximation.  Unlike RWM, the
proposal ``q(y)`` does not depend on the current state ``x``.

``extra_required_kwargs=("proposal_distribution",)``: the factory requires
``proposal_distribution`` (and optionally ``proposal_logdensity_fn``) as
keyword arguments.  The standard ``no_warmup`` path raises
``NotImplementedError``; a specialised wiring path is required.

Hyperparameter-free: no BO-tunable scalar HPs (``default_hp_space=()``).
The proposal is a full callable; it cannot be reduced to a tunable scalar.
Gradient-free: ``grad_count_per_step=0``.
``target_acceptance_rate=None``: depends entirely on proposal-vs-target
overlap; no universal optimal rate.

References
----------
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). Equation of state calculations by fast computing
  machines. *The Journal of Chemical Physics*, 21(6), 1087–1092.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*
  (2nd ed.). Springer. §7.3 Independent Metropolis-Hastings.
"""

import blackjax
import jax.numpy as jnp

from tuningfork.inference.base_method._base import (  # noqa: F401
    BaseMethod,
    HyperparamSpace,
)

__all__ = ["ENTRY", "_factory"]


def _factory(
    logdensity_fn,
    *,
    proposal_distribution,
    proposal_logdensity_fn=None,
    **kwargs,
):
    """Build a ``blackjax.irmh`` kernel.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density (log-posterior) callable ``x -> float``.
    proposal_distribution
        Callable ``(rng_key) -> position`` that draws a proposal independent
        of the current state.  Must return a position pytree in the same
        domain as the target.
    proposal_logdensity_fn
        Optional callable ``(proposal) -> float`` giving the log-density of
        the proposal distribution at ``proposal``.  Required when the
        proposal is NOT symmetric (i.e. when the MH ratio must include a
        proposal-density correction).  If ``None``, the proposal is assumed
        symmetric and no correction is applied.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.
    """
    return blackjax.irmh(
        logdensity_fn,
        proposal_distribution=proposal_distribution,
        proposal_logdensity_fn=proposal_logdensity_fn,
    )


ENTRY = BaseMethod(
    name="irmh",
    family="mcmc",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(0),  # gradient-free MH
    default_hp_space=(),  # truly HP-free; proposal is a full callable
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # depends entirely on proposal-vs-target overlap
    extra_required_kwargs=("proposal_distribution",),
    notes=(
        "Independent Metropolis-Hastings (Wang et al. 2022 reference; standard textbook "
        "MH variant where the proposal q(y) is independent of current state x). The proposal "
        "is a Callable (rng_key -> position) supplied at factory time, typically fitted from "
        "a VI / Pathfinder / Laplace approximation. For non-symmetric proposals, also supply "
        "proposal_logdensity_fn. Gradient-free (grad_count_per_step=0). RWInfo carries "
        "acceptance_rate, is_accepted, proposal. extra_required_kwargs=('proposal_distribution',); "
        "no_warmup raises NotImplementedError; the proposal-construction "
        "path. Also used as an SMC inner kernel — the standalone wrapper here is the "
        "non-SMC entry point; the SMC integration lives in the smc module."
    ),
)
