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
"""10-dimensional isotropic standard Gaussian — sanity-baseline starter model.

This is the sanity baseline: every sampler must produce correct results here.
The analytic sampler returns i.i.d. draws from N(0, I_10), which are
shape-compatible with the unconstrained parameterisation NumPyro produces.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = ["ENTRY"]

DIM = 10


def _model() -> None:
    numpyro.sample("x", dist.Normal(jnp.zeros(DIM), jnp.ones(DIM)))


def _analytic_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    """Return n i.i.d. draws from N(0, I_10) in unconstrained space.

    Parameters
    ----------
    rng_key
        JAX random key.
    n
        Number of samples.

    Returns
    -------
    dict with key ``"x"`` and value of shape ``(n, 10)``.
    """
    return {"x": jax.random.normal(rng_key, (n, DIM))}


ENTRY = Posterior(
    name="mvn_10",
    dim=DIM,
    class_="gaussian",
    numpyro_model=_model,
    analytic_sampler=_analytic_sampler,
    description=(
        "Standard 10-D MVN, isotropic N(0, I_10). "
        "Sanity baseline — every sampler must pass."
    ),
)
