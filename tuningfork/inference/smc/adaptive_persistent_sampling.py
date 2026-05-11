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
"""adaptive_persistent_sampling_smc wrapper for the bjx-bench SMC registry.

Wraps blackjax.smc.adaptive_persistent_sampling.as_top_level_api which
implements the Adaptive Persistent Sampling algorithm from Karamanis et al.
2025. This method is identical to persistent_sampling_smc, but instead of
requiring the caller to supply the tempering parameter lmbda at each step,
the kernel computes lmbda automatically using a root-solver to hit a target
effective sample size (target_ess).

Step-signature contract (standard 2-arg — same as adaptive_tempered_smc):
  The returned SamplingAlgorithm's step_fn has signature:
    ``step_fn(rng_key, state)``  — 2-arg, kernel computes lmbda internally
  This IS the standard bjx-bench step signature.

Note (docstring vs actual arity drift):
  The upstream ``adaptive_persistent_sampling.as_top_level_api`` docstring
  (at time of pinning) incorrectly states the step signature as
  ``(rng_key, state, lmbda)`` — 3-arg — mirroring persistent_sampling.
  The actual ``step_fn`` defined in the source is 2-arg ``(rng_key, state)``.
  This is a known docstring/implementation mismatch (candidate #8).
  The tripwire in tests/test_api_pins_smc.py pins BOTH the docstring text AND
  the actual arity so future docstring fixes don't accidentally widen the
  call site in bjx-bench.

Memory preallocation constraint:
  Adaptive Persistent Sampling requires ``max_iterations: int`` at
  construction time so arrays can be pre-allocated to shape
  ``(max_iterations + 1, num_particles)``. Since the adaptive algorithm does
  not know in advance how many steps it will take, the user must set
  max_iterations high enough that the algorithm converges before exceeding
  the limit. No internal check is performed upstream.

target_ess note:
  Unlike standard SMC where ESS ∈ (0, 1] relative to N particles, in
  Persistent Sampling the ESS is computed over all particles from ALL
  previous iterations and can be > 1. The default target_ess = 3 (upstream
  default) reflects this extended ESS range. This is NOT a bug.

Inner kernel constraints (same as adaptive_tempered_smc):
  MUST be MH-based; MCLMC family excluded (microcanonical invariance broken
  by tempering). ``mcmc_parameters`` dict must contain ONLY JAX arrays.
  Non-array params (e.g. random_step for RWM) must be bound via
  functools.partial before passing as mcmc_step_fn.

Returns blackjax.SamplingAlgorithm; init_fn(initial_particles) takes
particles array (leading dim = num_particles), step_fn(key, state)
returns (PersistentSMCState, PersistentStateInfo).

References
----------
Karamanis et al. 2025 — Persistent Sampling. (Cited in upstream docstrings.)
"""

import blackjax.smc.adaptive_persistent_sampling as _aps
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
    max_iterations: int,
    target_ess: float = 3.0,
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    **kwargs,
) -> SamplingAlgorithm:
    """Build adaptive_persistent_sampling_smc with a fully-instantiated inner kernel.

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
        as ``mcmc_init_fn``. Non-array parameters must already be bound to
        ``.step`` via ``functools.partial`` before passing here.
    mcmc_parameters
        Dict of MCMC step function parameters. MUST contain only JAX arrays
        (no callables); callable params must be bound at ``inner_kernel``
        build time.
    max_iterations
        Maximum number of tempering steps to perform. Required for memory
        pre-allocation: arrays are sized ``(max_iterations + 1, num_particles)``.
        The inference loop should break if this limit is exceeded.
    target_ess
        Target effective sample size used by the root-solver to determine the
        next tempering parameter. Default 3.0 (upstream default). NOTE: In
        Persistent Sampling, ESS is computed over all particles from all
        previous iterations and can be > 1; a target > 1 is normal and not
        a bug.
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step. Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``. Defaults to
        ``systematic``.
    **kwargs
        Additional keyword arguments passed through to
        ``blackjax.smc.adaptive_persistent_sampling.as_top_level_api``
        (e.g. ``root_solver``, ``update_strategy``).

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with ``init_fn(particles)`` and
        ``step_fn(rng_key, state)`` -> ``(PersistentSMCState,
        PersistentStateInfo)``.

    Notes
    -----
    The step_fn has the STANDARD 2-arg signature ``(rng_key, state)`` — the
    kernel computes lmbda internally via root-solving. This differs from
    persistent_sampling_smc which requires a 3-arg step ``(rng_key, state,
    lmbda)``. See note in module docstring regarding the upstream
    docstring/implementation mismatch.
    """
    if resampling_fn is None:
        resampling_fn = _resampling.systematic

    return _aps.as_top_level_api(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        max_iterations=max_iterations,
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="adaptive_persistent_sampling_smc",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(
        HyperparamSpace("max_iterations", "int", low=5, high=50),
        HyperparamSpace("target_ess", "uniform", low=1.0, high=5.0),
        HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),
    ),
    step_kwargs_schema=(),  # standard step(key, state) signature — kernel computes lmbda
    notes=(
        "Adaptive Persistent Sampling SMC (Karamanis et al. 2025). "
        "Extends persistent_sampling_smc with automatic tempering-parameter "
        "selection: the kernel computes lmbda at each step by root-solving to "
        "hit target_ess. Step signature is the standard 2-arg (rng_key, state). "
        "NOTE: target_ess > 1 is normal — ESS is computed over all particles "
        "from ALL previous iterations and can be > 1 in Persistent Sampling. "
        "Default target_ess=3.0 (upstream default). "
        "CRITICAL memory constraint: 'max_iterations' must be supplied at "
        "construction to pre-allocate arrays of shape "
        "(max_iterations + 1, num_particles). The inference loop should break "
        "if this limit is exceeded; no internal check is performed. "
        "Note: upstream docstring incorrectly states step signature as "
        "(rng_key, state, lmbda) — the actual step is 2-arg (rng_key, state). "
        "Pinned in tests/test_api_pins_smc.py section 13. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. "
        "SMC-level BO HPs: max_iterations (int [5, 50]), target_ess "
        "(uniform [1.0, 5.0]), num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic. "
        "CRITICAL inner-kernel contract: mcmc_parameters dict must contain ONLY "
        "JAX arrays — callable params (e.g. random_step for RWM) must be bound "
        "via functools.partial at build time."
    ),
)
