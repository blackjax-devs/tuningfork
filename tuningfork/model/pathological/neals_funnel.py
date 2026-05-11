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
"""Neal's funnel — pathological-geometry starter model.

A 10-dimensional funnel: v ~ N(0, 3²) and theta[i] ~ N(0, exp(v)) for i=1..9.
The geometry is strongly non-Euclidean: near v≈0 the conditional variance of
theta is exp(0)=1, but at v=-6 (two sigma below mean) it is exp(-6)≈0.002.

This model stresses any sampler that relies on a fixed metric, making it a
standard stress test for adaptation schemes.

Reference: Neal (2003), "Slice sampling". Annals of Statistics.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = ["ENTRY"]

DIM = 10
SCALE_LOG_V = 3.0  # std of v; variance of theta conditional on v is exp(v)


def _model() -> None:
    v = numpyro.sample("v", dist.Normal(0.0, SCALE_LOG_V))
    # 9 latents whose conditional std is exp(v / 2)
    numpyro.sample(
        "theta",
        dist.Normal(jnp.zeros(DIM - 1), jnp.exp(v / 2.0) * jnp.ones(DIM - 1)),
    )


def _analytic_sampler(rng_key: jax.Array, n: int) -> dict[str, jax.Array]:
    """Return n i.i.d. draws from the funnel in unconstrained space.

    The funnel factorises as p(v) p(theta|v), so we can sample exactly via
    ancestral sampling.

    Parameters
    ----------
    rng_key
        JAX random key.
    n
        Number of samples.

    Returns
    -------
    dict with keys ``"v"`` (shape ``(n,)``) and ``"theta"`` (shape ``(n, 9)``).
    """
    k_v, k_theta = jax.random.split(rng_key)
    v = jax.random.normal(k_v, (n,)) * SCALE_LOG_V
    theta = jax.random.normal(k_theta, (n, DIM - 1)) * jnp.exp(v[:, None] / 2.0)
    return {"v": v, "theta": theta}


ENTRY = Posterior(
    name="neals_funnel",
    dim=DIM,
    class_="funnel",
    numpyro_model=_model,
    analytic_sampler=_analytic_sampler,
    citations=("Neal 2003 'Slice sampling', Annals of Statistics",),
    description=(
        "10-D Neal's funnel: v~N(0,9), theta_i|v ~ N(0, exp(v)) for i=1..9. "
        "Pathological curvature — standard stress test for metric adaptation."
    ),
)
