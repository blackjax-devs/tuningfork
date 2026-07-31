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
"""adaptive_persistent_sampling descriptor for generated SMC execution plans."""

from __future__ import annotations

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.smc._base import SMCMethod

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


ENTRY = SMCMethod(
    name="adaptive_persistent_sampling_smc",
    family="smc",
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
