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
"""inner_kernel_tuning SMC wrapper for the bjx-bench SMC registry.

Wraps blackjax.smc.inner_kernel_tuning.as_top_level_api, which is a meta-SMC
algorithm that adapts the inner-kernel parameters (e.g. step_size) across SMC
iterations based on the particle cloud produced at each step.

Unlike standard SMC methods that hold the inner-kernel parameters fixed across
steps, inner_kernel_tuning applies a user-supplied ``mcmc_parameter_update_fn``
after each SMC step to compute new per-particle or global parameter overrides for
the NEXT step.  The resulting state is a ``StateWithParameterOverride`` (see
upstream ``blackjax.smc.inner_kernel_tuning``) that carries both the underlying
SMC state AND the current parameter dictionary.

Key divergences from the standard SMCMethod factory contract:

1. ``smc_algorithm`` (required keyword arg): the underlying SMC algorithm
   constructor, e.g. ``blackjax.adaptive_tempered_smc``.  The inner_kernel_tuning
   layer calls ``smc_algorithm(logprior_fn=..., loglikelihood_fn=..., ...)`` at
   each step to build a fresh step_fn with the current parameter override.

2. ``mcmc_parameter_update_fn`` (required keyword arg): a callable
   ``(rng_key, smc_state, smc_info) -> dict[str, Array]``
   that computes the new inner-kernel parameters from the most recent particle
   cloud.  MUST NOT be placed in ``mcmc_parameters`` (which must be JAX-arrays-only
   per the P5.10c constraint — ``from_mcmc.unshared_parameters_and_step_fn``
   calls ``.shape`` on every value in that dict).

3. ``initial_parameter_value`` (required keyword arg): ``dict[str, Array]``
   giving the initial inner-kernel parameters used before the first step.

4. ``mcmc_parameters`` (from standard contract): the initial value of
   ``initial_parameter_value``; they are the SAME dict here.  The
   inner_kernel_tuning layer takes ``initial_parameter_value`` directly and
   uses it as the initial ``parameter_override`` in ``StateWithParameterOverride``.

State shape: ``StateWithParameterOverride._fields == ('sampler_state', 'parameter_override')``.
Particles live at ``state.sampler_state.particles``.

Inner kernel constraints (same as adaptive_tempered_smc):
  MUST be MH-based; MCLMC family excluded (microcanonical invariance broken).
  ``mcmc_parameters`` / ``initial_parameter_value`` must contain ONLY JAX arrays.

Finding (P5.10d): ``smc_algorithm`` in ``build_kernel`` is called as a factory at
  EVERY step (not just at init time) — the inner_kernel_tuning layer re-instantiates
  the underlying SMC algorithm at each kernel invocation, passing the current
  parameter_override as ``mcmc_parameters``.  This is the mechanism by which
  parameters are updated.  The factory therefore does NOT accept the standard
  ``inner_kernel`` (SamplingAlgorithm) arg — instead it accepts the raw
  ``smc_algorithm`` constructor and the ``mcmc_step_fn``/``mcmc_init_fn``
  callables directly.
"""

import blackjax.smc.inner_kernel_tuning as _ikt
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
    smc_algorithm,
    mcmc_parameter_update_fn,
    initial_parameter_value: dict | None = None,
    num_mcmc_steps: int = 10,
    resampling_fn=None,
    smc_returns_state_with_parameter_override: bool = False,
    **kwargs,
):
    """Build inner_kernel_tuning SMC with an underlying SMC algorithm.

    Parameters
    ----------
    logprior_fn
        Log prior density function, passed through to ``smc_algorithm``.
    loglikelihood_fn
        Log likelihood function (NOT log posterior), passed through.
    inner_kernel
        A blackjax SamplingAlgorithm (with ``.step`` and ``.init``).
        Used to extract ``mcmc_step_fn = inner_kernel.step`` and
        ``mcmc_init_fn = inner_kernel.init``.  Non-array params must already
        be bound via ``functools.partial`` before passing.
    mcmc_parameters
        Initial MCMC parameter dict.  Must contain ONLY JAX arrays — no
        callables (P5.10c constraint: ``from_mcmc.unshared_parameters_and_step_fn``
        calls ``.shape`` on every value).  If ``initial_parameter_value`` is not
        provided, this dict is also used as ``initial_parameter_value``.
    smc_algorithm
        The underlying SMC algorithm constructor (callable), e.g.
        ``blackjax.adaptive_tempered_smc``.  The inner_kernel_tuning layer
        calls this at every step to build a fresh step_fn with the current
        parameter override as ``mcmc_parameters``.
    mcmc_parameter_update_fn
        A callable ``(rng_key, smc_state, smc_info) -> dict[str, Array]``
        that computes new inner-kernel parameters from the most recent
        particle cloud.  MUST NOT be placed in ``mcmc_parameters`` — it
        is a callable and would violate the JAX-arrays-only constraint.
    initial_parameter_value
        Initial parameter dictionary (JAX arrays only).  If ``None``,
        defaults to ``mcmc_parameters``.
    num_mcmc_steps
        Number of MCMC steps applied per particle per SMC step.  Default 10.
    resampling_fn
        Resampling function from ``blackjax.smc.resampling``.  Defaults to
        ``systematic``.
    smc_returns_state_with_parameter_override
        Set to ``True`` when composing multiple adaptation layers (e.g.
        pre-tuning with adaptive tuning).  Default ``False``.
    **kwargs
        Additional keyword arguments (e.g. ``target_ess`` for adaptive_tempered_smc)
        passed through to ``blackjax.smc.inner_kernel_tuning.as_top_level_api``
        as ``extra_parameters``.

    Returns
    -------
    SamplingAlgorithm
        A blackjax SamplingAlgorithm with ``init_fn(particles)`` and
        ``step_fn(rng_key, state) -> (StateWithParameterOverride, SMCInfo)``.
        Particles live at ``state.sampler_state.particles``.
    """
    if resampling_fn is None:
        resampling_fn = _resampling.systematic

    # initial_parameter_value defaults to mcmc_parameters when not supplied.
    _initial_params = (
        initial_parameter_value
        if initial_parameter_value is not None
        else mcmc_parameters
    )

    return _ikt.as_top_level_api(
        smc_algorithm=smc_algorithm,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=inner_kernel.step,
        mcmc_init_fn=inner_kernel.init,
        resampling_fn=resampling_fn,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=_initial_params,
        num_mcmc_steps=num_mcmc_steps,
        smc_returns_state_with_parameter_override=smc_returns_state_with_parameter_override,
        **kwargs,
    )


ENTRY = SMCMethod(
    name="inner_kernel_tuning",
    family="smc",
    factory=_factory,
    compatible_inner_methods=_COMPATIBLE_INNER,
    default_inner_method="rwm",
    num_particles_default=1000,
    default_hp_space=(HyperparamSpace("num_mcmc_steps", "int", low=1, high=50),),
    step_kwargs_schema=(),  # standard step(key, state) signature
    notes=(
        "Inner Kernel Tuning SMC (meta-SMC that adapts inner-kernel parameters). "
        "Wraps blackjax.smc.inner_kernel_tuning.as_top_level_api. At each SMC "
        "step, applies mcmc_parameter_update_fn(rng_key, smc_state, smc_info) "
        "to compute new per-step inner-kernel parameters for the NEXT mutation. "
        "State type: StateWithParameterOverride with _fields = "
        "('sampler_state', 'parameter_override'). Particles live at "
        "state.sampler_state.particles (not state.particles directly). "
        "Required extra factory kwargs: "
        "'smc_algorithm' (the underlying SMC algorithm constructor, e.g. "
        "blackjax.adaptive_tempered_smc) and "
        "'mcmc_parameter_update_fn' (callable: (rng_key, smc_state, smc_info) "
        "-> dict[str, Array]). "
        "CRITICAL: mcmc_parameter_update_fn MUST NOT go in mcmc_parameters — "
        "the JAX-arrays-only constraint from P5.10c (from_mcmc calls .shape on "
        "every value in that dict) means callables cannot be placed there. "
        "'initial_parameter_value' is the initial parameter dict; defaults to "
        "mcmc_parameters if not supplied. "
        "The smc_algorithm is re-instantiated at EVERY step with the current "
        "parameter_override as mcmc_parameters — this is the adaptation mechanism. "
        "Inner kernel must be MH-based — MCLMC family excluded (microcanonical "
        "invariance violated by tempering). Default inner: rwm. "
        "num_particles=1000 default. SMC-level BO HPs: num_mcmc_steps (int [1, 50]). "
        "Resampling default: systematic."
    ),
)
