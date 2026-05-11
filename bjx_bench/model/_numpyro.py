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
"""NumPyro helper for building BlackJAX-ready log-density functions.

This is the single helper used by every Path-B (long-NUTS) reference run and by
every BO tuning / warmup-only run that needs ``logdensity_fn``.

Pinned upstream API (NumPyro 0.21.0):
    ``initialize_model`` returns
    ``ModelInfo(param_info, potential_fn, postprocess_fn, model_trace)``;
    ``param_info`` is ``ParamInfo(z, potential_energy, z_grad)``.
"""

from collections.abc import Callable

import jax
from numpyro.infer.util import initialize_model

from bjx_bench.model._base import Posterior

__all__ = ["build_logdensity_fn"]


def build_logdensity_fn(
    rng_key: jax.Array,
    entry: Posterior,
) -> tuple[
    dict[str, jax.Array],
    Callable[[dict], float],
    Callable[[dict], dict],
]:
    """Initialize a NumPyro model and produce BlackJAX-ready functions.

    Parameters
    ----------
    rng_key
        JAX random key used by NumPyro's initialization sampler.
    entry
        Registry entry describing the model.

    Returns
    -------
    init_position
        Unconstrained initial position as a dict keyed by site name.
    logdensity_fn
        Positive log-density in unconstrained space
        (i.e. ``logdensity_fn(position)`` returns a scalar ``float``).
    postprocess_fn
        Transforms unconstrained draws back to constrained space (useful for
        computing summary statistics in the original parameterisation).

    Notes
    -----
    The returned ``logdensity_fn`` is
    ``lambda position: -potential_fn(position)``.  NumPyro's ``potential_fn``
    is the negative log-joint (following Stan's convention), so we negate it
    to get the positive log-density that BlackJAX expects.
    """
    model_info = initialize_model(
        rng_key,
        entry.numpyro_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    init_position = model_info.param_info.z
    potential_fn = model_info.potential_fn

    def logdensity_fn(position: dict) -> float:
        return -potential_fn(position)

    return init_position, logdensity_fn, model_info.postprocess_fn
