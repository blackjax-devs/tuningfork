"""Eight Schools NCP — Phase 1 starter model #3 (hierarchical).

Non-centred parameterisation of the classic 8-Schools model (Rubin 1981 /
Gelman et al. BDA3 §5.5). Latent dimensionality 10 = mu(1) + tau(1) +
theta_raw(8) in unconstrained space (tau is softplus-transformed internally
by NumPyro's HalfCauchy).

Reference draws are produced via the long-NUTS path (Path B) because the
posterior has no closed-form marginals.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from bjx_bench.registry._base import PosteriorEntry

__all__ = ["ENTRY"]

J = 8
SIGMA = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
Y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])


def _model(y: jnp.ndarray, sigma: jnp.ndarray) -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    theta_raw = numpyro.sample("theta_raw", dist.Normal(jnp.zeros(J), jnp.ones(J)))
    theta = numpyro.deterministic("theta", mu + tau * theta_raw)
    numpyro.sample("y", dist.Normal(theta, sigma), obs=y)


ENTRY = PosteriorEntry(
    name="eight_schools_ncp",
    # unconstrained: mu(1) + tau(1, softplus) + theta_raw(8) = 10
    dim=J + 2,
    class_="hierarchical",
    numpyro_model=_model,
    model_args=(Y, SIGMA),
    posteriordb_id="8_schools-eight_schools_noncentered",
    citations=(
        "Rubin 1981 'Estimation in parallel randomized experiments', "
        "J. Educational Statistics",
        "Gelman et al. BDA3 §5.5",
    ),
    description=(
        "8-Schools NCP: mu~N(0,5), tau~HalfCauchy(5), "
        "theta_raw[j]~N(0,1), theta=mu+tau*theta_raw, y~N(theta,sigma). "
        "Reference via long-NUTS (Path B)."
    ),
)
