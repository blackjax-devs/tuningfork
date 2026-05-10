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
"""tempered_smc wrapper for the bjx-bench SMC registry.

Wraps ``blackjax.smc.tempered.as_top_level_api`` which implements the
non-adaptive (fixed-schedule) Tempered SMC algorithm.  Unlike
``adaptive_tempered_smc``, the caller is responsible for supplying the
tempering parameter ``lmbda`` at each SMC step — the temperature schedule
is not computed automatically.

Upstream signature (blackjax/smc/tempered.py line 188-198):

.. code-block:: python

    def as_top_level_api(
        logprior_fn, loglikelihood_fn,
        mcmc_step_fn, mcmc_init_fn, mcmc_parameters,
        resampling_fn,
        num_mcmc_steps=10,
        update_strategy=update_and_take_last,
        update_particles_fn=None,
    )

Step-signature contract (IMPORTANT — differs from adaptive_tempered_smc):
  The returned ``SamplingAlgorithm.step`` has signature:
    ``step_fn(rng_key, state, tempering_param)``  — 3-arg
  Caller must supply ``tempering_param`` (a float in [0, 1]) at each SMC step.
  Tracked in ``step_kwargs_schema = ("tempering_param",)``.
  Note: the upstream field name is ``tempering_param`` (NOT ``lmbda`` as in
  ``persistent_sampling``).

State type: ``TemperedSMCState`` from ``blackjax.smc.tempered``.
  ``TemperedSMCState._fields = ('particles', 'weights', 'tempering_param')``.
  Note: the state field is also named ``tempering_param`` (not ``lmbda``).

Inner kernel constraints (same as adaptive_tempered_smc):
  MUST be MH-based (HMC/NUTS/Barker/MALA/RWM/IRMH/GHMC/dynamic_hmc).
  MCLMC and adjusted_mclmc family are EXCLUDED — microcanonical invariance
  is violated by tempering.
  ``mcmc_parameters`` dict must contain ONLY JAX arrays (no callables).
  Non-array params (e.g. ``random_step`` for RWM) must be bound via
  ``functools.partial`` BEFORE passing as ``mcmc_step_fn``.

Comparison with adaptive_tempered_smc:
  - ``adaptive_tempered_smc``: step_fn(key, state) — 2-arg; temperature chosen
    automatically via ESS-based dichotomy root-solving (target_ess HP).
  - ``tempered_smc``: step_fn(key, state, tempering_param) — 3-arg; caller
    manually drives the tempering schedule (e.g. linspace(0, 1, T)).
  Use ``adaptive_tempered_smc`` for most cases; ``tempered_smc`` when you want
  explicit control over the annealing schedule.

Returns blackjax.SamplingAlgorithm; ``init_fn(initial_particles)`` takes
particles array (leading dim = num_particles); ``step_fn(key, state,
tempering_param)`` returns ``(TemperedSMCState, SMCInfo)``.

References
----------
- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo
  samplers. *JRSS-B*, 68(3), 411-436.
- BlackJAX upstream: ``blackjax/smc/tempered.py``.
"""

import blackjax.smc.tempered as _tempered
from blackjax.base import SamplingAlgorithm
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
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    **kwargs,
) -> SamplingAlgorithm:
    """Build tempered_smc with a fully-instantiated inner kernel.

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
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step.  Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``.  Defaults to
        ``systematic``.
    **kwargs
        Additional keyword arguments passed through to
        ``blackjax.smc.tempered.as_top_level_api`` (e.g.
        ``update_strategy``, ``update_particles_fn``).

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with ``init_fn(particles)`` and
        ``step_fn(rng_key, state, tempering_param)`` ->
        ``(TemperedSMCState, SMCInfo)``.

    Notes
    -----
    The returned ``step_fn`` requires an extra ``tempering_param`` argument
    (the annealing parameter for the current step, in [0, 1]).  The caller
    must provide a tempering schedule and supply the correct value at each
    step; see ``step_kwargs_schema = ("tempering_param",)``.
    """
    if resampling_fn is None:
        resampling_fn = _resampling.systematic

    return _tempered.as_top_level_api(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        num_mcmc_steps=num_mcmc_steps,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="tempered_smc",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",  # statistician verdict: RWM/IRMH first
    num_particles_default=1000,
    default_hp_space=(HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),),
    step_kwargs_schema=("tempering_param",),  # step_fn(key, state, tempering_param)
    notes=(
        "Non-adaptive Tempered SMC (Del Moral et al. 2006). "
        "Caller is responsible for the tempering schedule: step_fn takes "
        "'tempering_param' as a 3rd argument (float in [0, 1]). "
        "Unlike adaptive_tempered_smc which auto-selects lmbda via ESS dichotomy, "
        "tempered_smc gives full control over the annealing schedule — useful when "
        "the schedule is known in advance or being tuned externally. "
        "CRITICAL step-fn divergence from adaptive variant: "
        "step_fn(rng_key, state, tempering_param) — 3-arg; "
        "step_kwargs_schema = ('tempering_param',). "
        "State type: TemperedSMCState from blackjax.smc.tempered. "
        "TemperedSMCState._fields = ('particles', 'weights', 'tempering_param'). "
        "Note: upstream field is 'tempering_param' (NOT 'lmbda'). "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm (statistician "
        "verdict: RWM/IRMH first). num_particles=1000 default. "
        "BO HP: num_mcmc_steps (int [1, 50]). Resampling default: systematic. "
        "CRITICAL inner-kernel contract: mcmc_parameters dict must contain ONLY "
        "JAX arrays — callable params (e.g. random_step for RWM) must be bound "
        "via functools.partial at build time. "
        "Comparison: use adaptive_tempered_smc for most benchmarks (auto-schedule); "
        "use tempered_smc when you want to control or study the annealing schedule."
    ),
)
