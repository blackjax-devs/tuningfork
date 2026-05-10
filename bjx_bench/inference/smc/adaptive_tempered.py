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
"""adaptive_tempered_smc wrapper for the bjx-bench SMC registry.

Wraps blackjax.adaptive_tempered_smc which adapts the temperature
schedule based on a target effective sample size (target_ess) at each
SMC step. The temperature progression is computed via dichotomy root
solving on the ESS function.

Inner kernel constraints (statistician verdict):
- MUST be MH-based (HMC/NUTS/Barker/MALA/RWM/IRMH/GHMC/dynamic_hmc).
- MCLMC and adjusted_mclmc[_dynamic] are EXCLUDED — microcanonical
  invariance is violated by tempering.

Inner kernel contract (CRITICAL — differs from BaseMethod pattern):
The blackjax SMC layer requires raw ``mcmc_step_fn`` and ``mcmc_init_fn``
callables, not a pre-built SamplingAlgorithm. ``mcmc_step_fn`` must have
signature ``(rng_key, state, logdensity_fn, **array_params)`` where
``**array_params`` contains ONLY JAX arrays. Non-array parameters (e.g.
``random_step`` callable for RWM) must be bound via ``functools.partial``
BEFORE passing as ``mcmc_step_fn``. The ``inner_kernel`` arg in the
factory is a SamplingAlgorithm from which we extract ``.step`` and
``.init``; however, for RWM these are the raw blackjax kernel functions
accessed directly.

blackjax SMC's ``from_mcmc.unshared_parameters_and_step_fn``
calls ``.shape`` on every value in ``mcmc_parameters``, so it must contain
ONLY JAX arrays — passing a callable (like ``random_step``) in
``mcmc_parameters`` raises ``AttributeError: 'function' object has no
attribute 'shape'``. Solution: bind callable params via ``functools.partial``
on the step function at factory time. Verified end-to-end with RWM in tests.

Returns blackjax.SamplingAlgorithm; init_fn(initial_particles) takes
particles array (leading dim = num_particles), step_fn(key, state)
returns (TemperedSMCState, SMCInfo).
"""

import blackjax
from blackjax.smc import resampling as _resampling

from bjx_bench.inference.base_method._base import HyperparamSpace
from bjx_bench.inference.smc._base import SMCMethod

__all__ = ["ENTRY"]


# Inner methods compatible with tempering (MH-based; excludes microcanonical).
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
    logprior_fn,
    loglikelihood_fn,
    *,
    inner_kernel,
    mcmc_parameters: dict,
    target_ess: float = 0.5,
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    **kwargs,
):
    """Build adaptive_tempered_smc with a fully-instantiated inner kernel.

    Parameters
    ----------
    logprior_fn
        Log prior density function.
    loglikelihood_fn
        Log likelihood function (NOT log posterior).
    inner_kernel
        A blackjax SamplingAlgorithm (with ``.step`` and ``.init``).
        The SMC layer extracts ``.step`` as ``mcmc_step_fn`` and ``.init``
        as ``mcmc_init_fn``.  Non-array parameters must already be bound
        to ``.step`` via ``functools.partial`` before passing here.
    mcmc_parameters
        Dict of MCMC step function parameters.  MUST contain only JAX
        arrays (no callables); callable params must be bound at
        ``inner_kernel`` build time.
    target_ess
        Target effective sample size fraction.  Default 0.5.
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step.  Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``.  Defaults to
        ``systematic``.
    **kwargs
        Additional keyword arguments passed through to
        ``blackjax.adaptive_tempered_smc``.

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with init_fn(particles) and
        step_fn(rng_key, state) -> (TemperedSMCState, SMCInfo).
    """
    if resampling_fn is None:
        resampling_fn = _resampling.systematic
    return blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="adaptive_tempered_smc",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",  # statistician verdict: RWM/IRMH first
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("target_ess", "uniform", low=0.3, high=0.95),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=(),  # standard step(key, state) signature
    notes=(
        "Adaptive Tempered SMC (Del Moral et al. 2006, adapted via target ESS). "
        "Temperature schedule chosen at each step by dichotomy root-solving on "
        "ESS(loglikelihood_fn(particles)) hitting target_ess * num_particles. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. SMC-level BO HPs: "
        "target_ess (uniform [0.3, 0.95]) and num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic. step_fn standard (key, state); no extras. "
        "CRITICAL inner-kernel contract: blackjax SMC requires raw mcmc_step_fn "
        "with signature (rng_key, state, logdensity_fn, **array_params). The "
        "mcmc_parameters dict must contain ONLY JAX arrays — callable params (e.g. "
        "random_step for RWM) must be bound via functools.partial at build time. "
        "The factory receives inner_kernel.step/.init but these must already have "
        "non-array params bound."
    ),
)
