"""10-dimensional isotropic standard Gaussian — Phase 1 starter model #1.

This is the sanity baseline: every sampler must produce correct results here.
The analytic sampler returns i.i.d. draws from N(0, I_10), which are
shape-compatible with the unconstrained parameterisation NumPyro produces.
"""

from __future__ import annotations

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
