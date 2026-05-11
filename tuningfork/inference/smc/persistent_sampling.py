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
"""persistent_sampling_smc wrapper for the tuningfork SMC registry.

Wraps blackjax.smc.persistent_sampling.as_top_level_api which implements
the Persistent Sampling algorithm from Karamanis et al. 2025. This method
keeps track of all particles from all previous iterations, building a
growing ensemble that yields more stable posterior and marginal-likelihood
estimates at the cost of higher memory usage.

Step-signature contract (IMPORTANT — differs from adaptive_tempered_smc):
  The returned SamplingAlgorithm's step_fn has signature:
    ``step_fn(rng_key, state, lmbda)``  — 3-arg, caller must provide lmbda
  This is a NON-STANDARD step signature for tuningfork SMC methods; the
  recipe-runner must supply lmbda at each SMC step.  Tracked in
  ``step_kwargs_schema = ("lmbda",)``.

Memory preallocation constraint:
  Persistent Sampling requires ``n_schedule: int`` at construction time so
  arrays can be pre-allocated to shape ``(n_schedule + 1, num_particles)``.
  The caller must ensure that the tempering schedule used in sampling
  matches exactly ``n_schedule`` steps. A schedule with many steps leads to
  proportionally higher memory usage.

Tempering start constraint:
  The algorithm enforces that the tempering schedule STARTS at 0.0.
  If the supplied schedule also starts at 0.0, the first step is effectively
  applied twice (upstream behaviour).

Inner kernel constraints (same as adaptive_tempered_smc):
  MUST be MH-based; MCLMC family excluded (microcanonical invariance broken
  by tempering). ``mcmc_parameters`` dict must contain ONLY JAX arrays.
  Non-array params (e.g. random_step for RWM) must be bound via
  functools.partial before passing as mcmc_step_fn.

The upstream build_kernel's update_strategy parameter
  accepts a callable with signature:
    (mcmc_init_fn, logposterior_fn, mcmc_step_fn, num_mcmc_steps, n_particles)
    -> (mcmc_kernel, n_particles)
  If a custom update_strategy is passed, it must be bound before passing
  as mcmc_parameters (same JAX-arrays-only constraint applies to the
  mcmc_parameters dict that is passed through to update_strategy internally).
  The default update_strategy=update_and_take_last satisfies this.

Returns blackjax.SamplingAlgorithm; init_fn(initial_particles) takes
particles array (leading dim = num_particles), step_fn(key, state, lmbda)
returns (PersistentSMCState, PersistentStateInfo).

References
----------
Karamanis et al. 2025 — Persistent Sampling. (Cited in upstream docstrings.)
"""

import blackjax.smc.persistent_sampling as _ps
from blackjax.base import SamplingAlgorithm
from blackjax.smc import resampling as _resampling

from tuningfork.inference.base_method._base import HyperparamSpace
from tuningfork.inference.smc._base import SMCMethod

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
    n_schedule: int,
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    **kwargs,
) -> SamplingAlgorithm:
    """Build persistent_sampling_smc with a fully-instantiated inner kernel.

    Parameters
    ----------
    logprior_fn
        Log prior density function. Must be normalized (Z_0 = 1) for the
        weighting scheme to function correctly.
    loglikelihood_fn
        Log likelihood function (NOT log posterior).
    inner_kernel
        A blackjax SamplingAlgorithm (with ``.step`` and ``.init``).
        The SMC layer extracts ``.step`` as ``mcmc_step_fn`` and ``.init``
        as ``mcmc_init_fn``. Non-array parameters must already be bound
        to ``.step`` via ``functools.partial`` before passing here.
    mcmc_parameters
        Dict of MCMC step function parameters. MUST contain only JAX
        arrays (no callables); callable params must be bound at
        ``inner_kernel`` build time.
    n_schedule
        Number of steps in the tempering schedule. Required for memory
        pre-allocation: arrays are sized ``(n_schedule + 1, num_particles)``.
        The caller must ensure the actual tempering schedule matches.
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step. Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``. Defaults to
        ``systematic``.
    **kwargs
        Additional keyword arguments passed through to
        ``blackjax.smc.persistent_sampling.as_top_level_api`` (e.g.
        ``update_strategy``).

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with ``init_fn(particles)`` and
        ``step_fn(rng_key, state, lmbda)`` -> ``(PersistentSMCState,
        PersistentStateInfo)``.

    Notes
    -----
    The returned ``step_fn`` requires an extra ``lmbda`` argument (the
    tempering parameter for the current step). The recipe-runner must supply
    ``lmbda`` at each call; see ``step_kwargs_schema = ("lmbda",)``.
    """
    if resampling_fn is None:
        resampling_fn = _resampling.systematic

    return _ps.as_top_level_api(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        n_schedule=n_schedule,
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        num_mcmc_steps=num_mcmc_steps,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="persistent_sampling_smc",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("n_schedule", "int", low=5, high=50),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=("lmbda",),  # step_fn(key, state, lmbda)
    notes=(
        "Persistent Sampling SMC (Karamanis et al. 2025). "
        "Keeps track of all particles from all previous iterations, building a "
        "growing ensemble for more stable posterior and marginal-likelihood "
        "estimation at the cost of higher memory usage. "
        "CRITICAL step-fn divergence: step_fn(rng_key, state, lmbda) requires "
        "an extra 'lmbda' argument (the tempering parameter) at each call; "
        "see step_kwargs_schema. Caller is responsible for the tempering schedule. "
        "CRITICAL memory constraint: 'n_schedule' must be supplied at construction "
        "to pre-allocate arrays of shape (n_schedule + 1, num_particles). "
        "CRITICAL tempering constraint: schedule must START at 0.0 (enforced "
        "upstream); if the supplied schedule also starts at 0.0, the first step "
        "is applied twice. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. "
        "SMC-level BO HPs: n_schedule (int [5, 50]) and num_mcmc_steps "
        "(int [1, 50]). Resampling default: systematic. "
        "CRITICAL inner-kernel contract: blackjax SMC requires raw mcmc_step_fn "
        "with signature (rng_key, state, logdensity_fn, **array_params). The "
        "mcmc_parameters dict must contain ONLY JAX arrays — callable params (e.g. "
        "random_step for RWM) must be bound via functools.partial at build time. "
        ":"
    ),
)
