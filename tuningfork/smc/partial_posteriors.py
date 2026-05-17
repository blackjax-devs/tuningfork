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
"""partial_posteriors_smc wrapper for the tuningfork SMC registry.

Wraps blackjax.smc.partial_posteriors_path.as_top_level_api which implements
data-tempering SMC: rather than annealing a temperature parameter, the SMC
path progressively includes more data points into the likelihood by stepping
through increasingly complete data masks.

Key divergence from the standard SMCMethod factory contract:
  partial_posteriors_smc does NOT take (logprior_fn, loglikelihood_fn) — it
  takes a ``partial_logposterior_factory``, a callable from binary data-mask
  array to logposterior function.  Both (logprior_fn, loglikelihood_fn) kwargs
  in the factory signature are IGNORED; they exist for registry-API uniformity
  only (the recipe-runner passes them; we discard them silently here).

  ``partial_logposterior_factory: Callable[[Array], Callable[[position], float]]``
  — must be JAX-traceable.  Signature: ``mask -> logposterior_fn``.

Step-function contract:
  Unlike adaptive_tempered_smc where step_fn(key, state) is standard, here the
  kernel returns a SamplingAlgorithm whose step_fn has an EXTRA positional arg:
    ``step_fn(rng_key, state, data_mask)``
  where ``data_mask`` is a 1-D boolean/integer array selecting which data points
  to include in the next partial posterior.  The recipe-runner must supply this
  extra argument at each SMC step; see ``step_kwargs_schema = ("data_mask",)``.

Init contract:
  ``init_fn(initial_particles, num_observations)`` — NOT the standard
  ``init_fn(initial_particles)``.  num_observations is the total number of
  data points in the dataset.

Inner kernel constraints (same as adaptive_tempered_smc):
  MUST be MH-based; MCLMC family excluded (microcanonical invariance broken).
  ``mcmc_parameters`` dict must contain ONLY JAX arrays.  Non-array params
  (e.g. random_step for RWM) must be bound via functools.partial before
  passing as mcmc_step_fn.

``as_top_level_api`` parameter order is positional for
  mcmc_step_fn, mcmc_init_fn, mcmc_parameters, resampling_fn, num_mcmc_steps,
  partial_logposterior_factory, update_strategy — NOT keyword-only.  The
  returned SamplingAlgorithm's init expects ``(position, num_observations)``,
  not ``(position,)`` alone.
"""

import blackjax.smc.partial_posteriors_path as _pp_path
from blackjax.smc import resampling as _resampling

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.smc._base import SMCMethod

__all__ = ["ENTRY"]


# Inner methods compatible with data-tempering (MH-based; excludes microcanonical).
_COMPATIBLE_INNER = (
    "rwm",
    "irmh",
    "mala",
    "barker",
    "hmc",
    "nuts",
    "ghmc",
    "dynamic_hmc",
)


def _factory(
    logprior_fn,  # ignored — partial_posteriors_smc uses partial_logposterior_factory
    loglikelihood_fn,  # ignored — ditto
    *,
    inner_kernel,
    mcmc_parameters: dict,
    partial_logposterior_factory,
    num_observations: int,
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    **kwargs,
):
    """Build partial_posteriors_smc with a fully-instantiated inner kernel.

    Parameters
    ----------
    logprior_fn
        IGNORED.  partial_posteriors_smc uses ``partial_logposterior_factory``
        instead of (logprior_fn, loglikelihood_fn).  Present only for
        registry-API uniformity with the standard SMCMethod factory contract.
    loglikelihood_fn
        IGNORED.  See ``logprior_fn`` above.
    inner_kernel
        A blackjax SamplingAlgorithm (with ``.step`` and ``.init``).
        The SMC layer extracts ``.step`` as ``mcmc_step_fn`` and ``.init``
        as ``mcmc_init_fn``.  Non-array parameters must already be bound to
        ``.step`` via ``functools.partial`` before passing here.
    mcmc_parameters
        Dict of MCMC step function parameters.  MUST contain only JAX arrays
        (no callables); callable params must be bound at ``inner_kernel``
        build time.
    partial_logposterior_factory
        Callable ``mask -> logposterior_fn`` where ``mask`` is a 1-D binary
        array of length ``num_observations``.  Must be JAX-traceable.
    num_observations
        Total number of data points in the dataset.  Required so that
        ``init_fn`` can create the initial all-zeros data mask.
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step.  Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``.  Defaults to
        ``systematic``.
    **kwargs
        Additional keyword arguments passed through to
        ``blackjax.smc.partial_posteriors_path.as_top_level_api``.

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with ``init_fn(particles, num_observations)``
        and ``step_fn(rng_key, state, data_mask)``.

    Notes
    -----
    The returned ``step_fn`` requires an extra ``data_mask`` argument — a 1-D
    binary array selecting which data points to include in the next partial
    posterior.  The recipe-runner must supply ``data_mask`` at each step;
    see ``step_kwargs_schema = ("data_mask",)`` in the registry ENTRY.
    """
    del logprior_fn, loglikelihood_fn  # explicitly discard unused registry args

    if resampling_fn is None:
        resampling_fn = _resampling.systematic

    return _pp_path.as_top_level_api(
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        num_mcmc_steps=num_mcmc_steps,
        partial_logposterior_factory=partial_logposterior_factory,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="partial_posteriors_smc",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),),
    step_kwargs_schema=("data_mask",),  # step_fn(key, state, data_mask)
    notes=(
        "Partial Posteriors SMC (data-tempering SMC). "
        "Rather than annealing a temperature, this SMC path progressively includes "
        "more data points by stepping through increasingly complete data masks. "
        "See Section 2.2 of https://arxiv.org/pdf/2007.11936. "
        "CRITICAL API divergence: factory takes 'partial_logposterior_factory' "
        "(a callable mapping binary data-mask → logposterior_fn) instead of the "
        "standard (logprior_fn, loglikelihood_fn) pair.  The (logprior_fn, "
        "loglikelihood_fn) kwargs are IGNORED when present (kept for registry "
        "uniformity only). "
        "CRITICAL step-fn divergence: step_fn(rng_key, state, data_mask) requires "
        "an extra 'data_mask' argument at each call; see step_kwargs_schema. "
        "CRITICAL init-fn divergence: init_fn(particles, num_observations) — "
        "num_observations is the total data count for the initial all-zeros mask. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm. "
        "num_particles=1000 default. SMC-level BO HPs: num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic. "
        "CRITICAL inner-kernel contract: mcmc_parameters dict must contain ONLY JAX "
        "arrays — callable params (e.g. random_step for RWM) must be bound via "
        "functools.partial at build time."
    ),
)
