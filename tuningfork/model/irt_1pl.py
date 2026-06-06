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
"""IRT 1PL (Rasch) model — 500-D NCP psychometric model (J=500 students, I=10 items)."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = [
    "ENTRY",
    "RESPONSE",
    "N_STUDENTS",
    "N_ITEMS",
    "DIM",
]

#: Number of students (J = 500)
N_STUDENTS: int = 500

#: Number of items (I = 10)
N_ITEMS: int = 10

#: Unconstrained dimensionality:
#: theta(N_STUDENTS) = 500
DIM: int = N_STUDENTS  # = 500

# ---------------------------------------------------------------------------
# Fixed item difficulties (from standard Rasch model benchmark)
# Linearly spaced between -2.0 and 2.0 across the 10 items.
# ---------------------------------------------------------------------------
_FIXED_DIFFICULTIES: jnp.ndarray = jnp.linspace(-2.0, 2.0, N_ITEMS)


# ---------------------------------------------------------------------------
# Synthetic Response Generation (deterministic at compile time, no CSV needed)
# ---------------------------------------------------------------------------
def _generate_synthetic_responses() -> jnp.ndarray:
    """Generates a binary response matrix of shape (J=500, I=10) with a fixed seed.

    Ensures zero runtime files, absolute reproducibility, and zero network calls.
    """
    key = jax.random.key(12345)
    k1, k2 = jax.random.split(key)

    # 1. Ground truth theta_j ~ N(0, 1.2^2)
    true_sigma_theta = 1.2
    true_theta = jax.random.normal(k1, (N_STUDENTS,)) * true_sigma_theta

    # 2. Probability of correct response under 1PL: P = logit^-1(theta_j - b_i)
    logits = true_theta[:, None] - _FIXED_DIFFICULTIES[None, :]
    probs = jax.nn.sigmoid(logits)

    # 3. Simulate binary observations
    y = jax.random.bernoulli(k2, probs).astype(jnp.float32)
    return y


RESPONSE: jnp.ndarray = _generate_synthetic_responses()


# ---------------------------------------------------------------------------
# NumPyro model (NCP IRT 1PL / Rasch Model)
# ---------------------------------------------------------------------------
def _model(response: jnp.ndarray) -> None:
    """NumPyro model: 500-D NCP 1-parameter logistic IRT (Rasch model).

    Students: J=500
    Items: I=10 (with fixed, centered difficulties)
    Parameters: J student abilities = 500 dimensions.
    """
    # 1. Student abilities (fixed scale = 1.0, unconstrained real line)
    theta = numpyro.sample(
        "theta", dist.Normal(jnp.zeros(N_STUDENTS), jnp.ones(N_STUDENTS))
    )

    # 2. Likelihood: logit(p) = theta_j - b_i
    # _FIXED_DIFFICULTIES sum to 0. No global intercept needed.
    logits = theta[:, None] - _FIXED_DIFFICULTIES[None, :]
    numpyro.sample("response", dist.Bernoulli(logits=logits), obs=response)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------
ENTRY = Posterior(
    name="irt_1pl",
    dim=DIM,
    class_="hierarchical",
    tags=("psychometrics", "high-dim", "hierarchical"),
    numpyro_model=_model,
    model_args=(RESPONSE,),
    analytic_sampler=None,
    description=(
        "500-D NCP 1-parameter logistic (Rasch) IRT model. "
        "Evaluates 500 estimated student abilities with 10 fixed item difficulties. "
        "A highly isotropic, log-concave, smooth custom psychometric geometry."
    ),
    headline_params=None,
    headline_coords=None,
)
