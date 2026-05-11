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
"""Shared test fixtures and utilities for tuningfork tests.

This module provides:
- RNG key generation helpers (make_rng, rng_key parametrized fixture)
- Toy MVN logdensity and init position helpers for generic chain tests
- Pattern: mirrors blackjax/tests/fixtures.py for consistency
"""

import jax
import jax.numpy as jnp
import pytest


def make_rng(seed: int = 42) -> jax.Array:
    """Return a jax.random.key for the given seed.

    Parameters
    ----------
    seed
        Integer seed for reproducibility. Default 42 is the convention
        across tuningfork (see certify_reference.py, tune.py).

    Returns
    -------
    jax.Array
        A JAX random key ready for use with jax.random functions.

    Example
    -------
    >>> key = make_rng(0)
    >>> x = jax.random.normal(key, shape=(5,))
    """
    return jax.random.key(seed)


@pytest.fixture(params=[0, 1, 42], ids=["seed0", "seed1", "seed42"])
def rng_key(request) -> jax.Array:
    """Parametrized RNG key fixture for seed coverage.

    This fixture is parametrized over three seeds (0, 1, 42) to ensure
    tests that depend on random operations exercise different RNG states.
    Use this fixture when you need to verify seed robustness; otherwise,
    use make_rng(seed=42) directly.

    Yields
    ------
    jax.Array
        A JAX random key seeded with the parametrized seed.

    Example
    -------
    >>> def test_something(rng_key):  # Test runs 3x with seed=0, 1, 42
    ...     x = jax.random.normal(rng_key, shape=(5,))
    """
    return jax.random.key(request.param)


def mvn_5d_logdensity(position: jax.Array) -> jax.Array:
    """Standard 5-D isotropic Gaussian log-density.

    This is the canonical toy model used in tripwire and generic kernel tests.
    It simplifies to -0.5 * sum(position**2), making it cheap to evaluate
    while remaining realistic for integrator / adaptation tests.

    Parameters
    ----------
    position
        Shape (5,) array.

    Returns
    -------
    jax.Array
        Log-density (scalar).

    Example
    -------
    >>> import jax.numpy as jnp
    >>> x = jnp.zeros(5)
    >>> logp = mvn_5d_logdensity(x)
    >>> assert logp == 0.0  # log-density at origin
    """
    return -0.5 * jnp.sum(position**2)


def mvn_5d_init() -> jax.Array:
    """Initial position for 5-D Gaussian: zeros(5).

    Returns
    -------
    jax.Array
        Shape (5,) array of zeros.

    Example
    -------
    >>> x0 = mvn_5d_init()
    >>> assert x0.shape == (5,)
    >>> assert jnp.allclose(x0, 0.0)
    """
    return jnp.zeros(5)
