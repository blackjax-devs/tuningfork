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
"""Tests pinning build_smc_logfns / build_prior_sample_fn to a single,
consistent unconstrained-space convention.

SMC consumes ``logprior_fn``/``loglikelihood_fn`` and particle positions in
the same **unconstrained** parameterization as ``build_logdensity_fn``'s
``init_position`` and ``postprocess_fn`` (unconstrained -> constrained). A
model with only real-supported latents can't catch a space mix-up (the two
spaces coincide), so these tests use a tiny model with one positive-support
latent, where constrained and unconstrained genuinely differ, and check
against an independently-derived analytic reference built from NumPyro's own
bijector rather than from the functions under test.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import initialize_model

from tuningfork.model._base import Posterior
from tuningfork.model._numpyro import build_prior_sample_fn, build_smc_logfns

_Y_DATA = jnp.array([0.3, -0.2, 0.5, -0.1, 0.4])


def _toy_halfnormal_model(y: jnp.ndarray) -> None:
    """sigma ~ HalfNormal(2.0); y ~ Normal(0, sigma).

    One positive-support latent: constrained space (sigma > 0) and
    unconstrained space (log sigma, any real) genuinely differ, so a
    constrained/unconstrained mix-up is not invisible here.
    """
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    numpyro.sample("y", dist.Normal(0.0, sigma), obs=y)


_TOY_ENTRY = Posterior(
    name="test_toy_halfnormal",
    dim=1,
    class_="test",
    numpyro_model=_toy_halfnormal_model,
    model_args=(_Y_DATA,),
)

_SIGMA_TRANSFORM = biject_to(constraints.positive)

# Dated, not searched (per project convention: seeds are stamped, never tuned).
_SEED = 20261002
_Z_GRID = (-2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0)


def _analytic_logprior(z: float) -> float:
    """log p(sigma) + log|det J| at unconstrained z, via NumPyro's own bijector."""
    sigma = _SIGMA_TRANSFORM(jnp.asarray(z))
    log_abs_det_jac = _SIGMA_TRANSFORM.log_abs_det_jacobian(jnp.asarray(z), sigma)
    return float(dist.HalfNormal(2.0).log_prob(sigma) + log_abs_det_jac)


def _analytic_loglik(z: float) -> float:
    """log p(y | sigma = constrain(z)) at unconstrained z."""
    sigma = _SIGMA_TRANSFORM(jnp.asarray(z))
    return float(dist.Normal(0.0, sigma).log_prob(_Y_DATA).sum())


class TestBuildSmcLogfnsUnconstrainedSpace:
    """logprior_fn / loglikelihood_fn must both operate in unconstrained space."""

    @pytest.mark.fast
    def test_logprior_fn_matches_analytic_prior(self) -> None:
        key = jax.random.key(_SEED)
        _, logprior_fn, _, _ = build_smc_logfns(key, _TOY_ENTRY)
        for z in _Z_GRID:
            got = float(logprior_fn({"sigma": jnp.asarray(z)}))
            want = _analytic_logprior(z)
            assert got == pytest.approx(
                want, abs=1e-5
            ), f"logprior_fn({z}) = {got}, expected {want}"

    @pytest.mark.fast
    def test_loglikelihood_fn_matches_analytic_likelihood(self) -> None:
        key = jax.random.key(_SEED)
        _, _, loglik_fn, _ = build_smc_logfns(key, _TOY_ENTRY)
        for z in _Z_GRID:
            got = float(loglik_fn({"sigma": jnp.asarray(z)}))
            want = _analytic_loglik(z)
            assert got == pytest.approx(
                want, abs=1e-5
            ), f"loglikelihood_fn({z}) = {got}, expected {want}"

    @pytest.mark.fast
    def test_logprior_plus_loglikelihood_equals_negative_potential(self) -> None:
        """logprior_fn(z) + loglikelihood_fn(z) == -potential_fn(z) for every z.

        True by construction once each half is individually correct (the
        Jacobian terms must cancel exactly) -- kept as an explicit assertion
        since it's the invariant SMC's own tempered-target interpolation
        relies on.
        """
        key = jax.random.key(_SEED)
        _, logprior_fn, loglik_fn, _ = build_smc_logfns(key, _TOY_ENTRY)
        model_info = initialize_model(
            key,
            _TOY_ENTRY.numpyro_model,
            model_args=_TOY_ENTRY.model_args,
            model_kwargs=_TOY_ENTRY.model_kwargs,
            dynamic_args=False,
        )
        potential_fn = model_info.potential_fn
        for z in _Z_GRID:
            position = {"sigma": jnp.asarray(z)}
            total = float(logprior_fn(position)) + float(loglik_fn(position))
            expected = float(-potential_fn(position))
            assert total == pytest.approx(
                expected, abs=1e-5
            ), f"logprior+loglik={total}, -potential_fn={expected} at z={z}"

    @pytest.mark.fast
    def test_blocked_model_latent_sites_match_joint_model(self) -> None:
        """The prior-only (blocked) model's init_position has exactly the
        same latent-site key set as the joint model's -- blocking only ever
        hides is_observed sites, never adds or removes a latent one."""
        key = jax.random.key(_SEED)
        init_position, _, _, _ = build_smc_logfns(key, _TOY_ENTRY)
        assert set(init_position.keys()) == {"sigma"}


class TestBuildPriorSampleFnUnconstrainedSpace:
    """build_prior_sample_fn's Predictive fallback must also return
    unconstrained particles, matching postprocess_fn's own convention."""

    @pytest.mark.fast
    def test_predictive_fallback_particles_include_negative_values(self) -> None:
        """sigma's constrained support is strictly positive; any negative
        particle value proves these are not already-constrained draws."""
        prior_sample_fn = build_prior_sample_fn(_TOY_ENTRY)
        particles = prior_sample_fn(jax.random.key(_SEED + 1), 2000)
        assert jnp.any(particles["sigma"] < 0.0), (
            "particles look constrained (all non-negative); "
            "build_prior_sample_fn's fallback must return unconstrained draws"
        )

    @pytest.mark.fast
    def test_predictive_fallback_particles_recover_prior_mean_after_postprocess(
        self,
    ) -> None:
        """Pushing particles through postprocess_fn (unconstrained ->
        constrained) must recover the model's actual HalfNormal(2.0) prior
        on sigma. Already-constrained particles would be double-transformed
        (constrain an already-constrained value) and badly inflate the mean.
        """
        _, _, _, postprocess_fn = build_smc_logfns(jax.random.key(_SEED), _TOY_ENTRY)
        prior_sample_fn = build_prior_sample_fn(_TOY_ENTRY)
        particles = prior_sample_fn(jax.random.key(_SEED + 1), 20_000)

        constrained = postprocess_fn(particles)
        mean_sigma = float(jnp.mean(constrained["sigma"]))
        analytic_mean = 2.0 * (2.0 / jnp.pi) ** 0.5  # E[HalfNormal(scale=2)]
        assert mean_sigma == pytest.approx(analytic_mean, abs=0.05), (
            f"postprocess_fn(particles)['sigma'] mean = {mean_sigma}, "
            f"expected ~{analytic_mean} (HalfNormal(2.0)'s own mean)"
        )
