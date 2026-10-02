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
every generated recipe execution that needs ``logdensity_fn``.

Also provides SMC-specific helpers (W1/W2, Phase 8B.1):
  - ``build_smc_logfns`` — splits the joint log-density into (logprior_fn, loglik_fn)
    by blocking observed sites via the NumPyro ``block`` handler.
  - ``build_prior_sample_fn`` — returns a ``prior_sample_fn(key, n) -> dict``
    suitable for ``init_particles_from_prior``, using the model's analytic sampler
    when available (mvn_10, ill_cond_50, banana, neals_funnel, gmm_25) and
    falling back to NumPyro ``Predictive`` otherwise.

Pinned upstream API (NumPyro 0.21.0):
    ``initialize_model`` returns
    ``ModelInfo(param_info, potential_fn, postprocess_fn, model_trace)``;
    ``param_info`` is ``ParamInfo(z, potential_energy, z_grad)``.
"""

from collections.abc import Callable

import jax
from numpyro.handlers import block as _numpyro_block
from numpyro.handlers import seed as _numpyro_seed
from numpyro.handlers import trace as _numpyro_trace
from numpyro.infer.util import get_transforms as _numpyro_get_transforms
from numpyro.infer.util import initialize_model
from numpyro.infer.util import transform_fn as _numpyro_transform_fn

from tuningfork.model._base import Posterior

__all__ = ["build_logdensity_fn", "build_smc_logfns", "build_prior_sample_fn"]


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


def _get_observed_site_names(entry: Posterior) -> frozenset[str]:
    """Return the set of observed (data) site names in the NumPyro model.

    Runs the model once under a ``trace`` handler with a seeded RNG to
    identify which ``numpyro.sample`` sites carry ``is_observed=True``
    (i.e. have a fixed ``obs`` value).  The result is stable across
    different seeds.

    Parameters
    ----------
    entry
        Posterior registry entry.

    Returns
    -------
    frozenset[str]
        Names of the observed sites.
    """
    seeded_model = _numpyro_seed(entry.numpyro_model, rng_seed=0)
    with _numpyro_trace() as tr:
        seeded_model(*entry.model_args, **entry.model_kwargs)
    return frozenset(k for k, v in tr.items() if v.get("is_observed", False))


def build_smc_logfns(
    rng_key: jax.Array,
    entry: Posterior,
) -> tuple[
    dict[str, jax.Array],
    Callable[[dict], float],
    Callable[[dict], float],
    Callable[[dict], dict],
]:
    """Build logprior_fn and loglikelihood_fn from a joint NumPyro model.

    SMC algorithms require the prior and likelihood to be supplied separately
    (unlike MCMC, which uses the joint log-density).  This function splits
    the joint ``logdensity_fn`` into its prior and likelihood components by
    blocking the observed sites via NumPyro's ``block`` handler:

    - ``logprior_fn(position)`` = log p(constrain(position)) + log|det
      J(position)| — the *negated potential energy of the blocked (prior-only)
      model*, built the same way ``initialize_model`` builds the joint
      ``potential_fn``, so both share the identical unconstrained-space
      convention and Jacobian term.
    - ``loglikelihood_fn(position)`` = log p(data | constrain(position)) —
      computed as ``logposterior − logprior``; because both terms carry the
      *same* ``log|det J|`` contribution, it cancels out exactly, leaving a
      pure data-likelihood term with no Jacobian of its own (numerically
      exact, no separate model call needed).

    ``position`` is always the **unconstrained** parameterization used by
    ``initialize_model``/``find_valid_initial_params`` (the same space as
    ``build_logdensity_fn``'s ``init_position`` and this function's own
    ``init_position``) — the same space ``postprocess_fn`` maps *from*.
    Both ``logprior_fn`` and ``loglikelihood_fn`` are JAX-traceable and
    JIT-compatible.

    Parameters
    ----------
    rng_key
        JAX random key for ``initialize_model``, used verbatim for the joint
        model (so ``init_position`` is unchanged from any prior version of
        this function); the prior-only (blocked) model's own init search
        uses a key folded in from ``rng_key``.
    entry
        Posterior registry entry describing the NumPyro model.

    Returns
    -------
    init_position
        Unconstrained initial position dict (same as ``build_logdensity_fn``).
    logprior_fn
        ``(position: dict) -> float`` — log prior in unconstrained space.
    loglikelihood_fn
        ``(position: dict) -> float`` — log likelihood (joint minus prior),
        in the same unconstrained space.
    postprocess_fn
        Transforms unconstrained draws to constrained space.

    Notes
    -----
    The ``block`` handler is applied once at function-build time to identify
    observed sites; the resulting blocked (prior-only) model is then run
    through its own ``initialize_model`` call to obtain a genuine
    unconstrained-space potential function — matching the joint model's own
    ``inv_transforms`` for every latent site, since blocking only hides
    observed sites and never touches which sites are latent.
    """
    model_info = initialize_model(
        rng_key,
        entry.numpyro_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    init_position = model_info.param_info.z
    potential_fn = model_info.potential_fn  # joint: negative log-posterior

    # Identify observed sites once (stable across seeds).
    obs_sites = _get_observed_site_names(entry)

    # Blocked model: same as the original model but with observed sites
    # silenced -- its own initialize_model gives a prior-only potential_fn
    # in the identical unconstrained space/Jacobian convention as the joint
    # model's potential_fn above, so the two compose correctly below. Uses a
    # key folded in from rng_key (not a split of it) so init_position above
    # is bit-identical to what build_smc_logfns returned before this fix --
    # the blocked model's own key only affects its own init search, not the
    # resulting potential_fn.
    _blocked_model = _numpyro_block(entry.numpyro_model, hide=obs_sites)
    blocked_model_info = initialize_model(
        jax.random.fold_in(rng_key, 1),
        _blocked_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    blocked_potential_fn = blocked_model_info.potential_fn

    def logprior_fn(position: dict) -> float:
        """Log prior p(constrain(position)) + log|det J(position)|."""
        return -blocked_potential_fn(position)

    def loglikelihood_fn(position: dict) -> float:
        """Log likelihood log p(data | constrain(position)); Jacobian cancels."""
        return -potential_fn(position) - logprior_fn(position)

    return init_position, logprior_fn, loglikelihood_fn, model_info.postprocess_fn


def build_prior_sample_fn(
    entry: Posterior,
) -> Callable[[jax.Array, int], dict[str, jax.Array]]:
    """Return a prior sampler ``(key, n_particles) -> dict`` for SMC particle init.

    Dispatches based on whether the model has an analytic sampler:

    - **Fast path** (``entry.analytic_sampler is not None``): calls
      ``entry.analytic_sampler(key, n)`` directly.  Available for mvn_10,
      ill_cond_50, banana, neals_funnel, gmm_25.  Per ``Posterior``'s own
      contract, ``analytic_sampler`` already draws in unconstrained space.
    - **Fallback** (all other models): uses ``numpyro.infer.Predictive`` to
      draw samples from the prior predictive (in the model's *constrained*
      space), keeps only the *latent* (non-observed) sites, then applies
      each site's inverse constrain-to-unconstrain bijector so the returned
      particles match the analytic path's unconstrained convention.

    Parameters
    ----------
    entry
        Posterior registry entry.

    Returns
    -------
    prior_sample_fn
        Callable ``(rng_key: jax.Array, n_particles: int) -> dict[str, Array]``
        returning **unconstrained**-space particles, where each array has
        shape ``(n_particles, *site_shape)`` — the same convention
        ``build_smc_logfns``'s ``logprior_fn``/``loglikelihood_fn`` and
        ``postprocess_fn`` (unconstrained → constrained) expect.

    Notes
    -----
    ``Predictive`` itself returns samples in the model's constrained space
    (e.g. a HalfNormal-prior site comes back strictly positive); the inverse
    bijector (``biject_to(support).inv``, the same transform
    ``initialize_model`` registers for that site) is applied explicitly
    below — it is *not* automatic. The bijector set is derived once from a
    single seeded trace (any one valid draw fixes every site's declared
    support) and then applied batched over all ``n_particles`` at once, so
    no second per-particle model trace is needed.

    Generated SMC programs call this function's output as
    ``prior_sample_fn(key, n_particles)`` when initializing particles.
    """
    if entry.analytic_sampler is not None:
        # Fast path: analytically-defined prior sampler.
        _analytic = entry.analytic_sampler

        def _analytic_prior_sample_fn(
            rng_key: jax.Array, n_particles: int
        ) -> dict[str, jax.Array]:
            return _analytic(rng_key, n_particles)

        return _analytic_prior_sample_fn

    # Fallback: NumPyro prior-predictive.
    # Identify latent (non-observed) site names once.
    _obs_sites = _get_observed_site_names(entry)
    from numpyro.infer import Predictive as _Predictive  # noqa: PLC0415

    # Inverse (constrained -> unconstrained) bijectors, derived once from a
    # single seeded trace. biject_to(support) depends only on each site's
    # declared support, not on the realized value, so this is safe to reuse
    # unbatched against the later batch of n_particles draws.
    _seeded_model = _numpyro_seed(entry.numpyro_model, rng_seed=0)
    _transforms = _numpyro_get_transforms(
        _seeded_model, entry.model_args, entry.model_kwargs, {}
    )

    def _predictive_prior_sample_fn(
        rng_key: jax.Array, n_particles: int
    ) -> dict[str, jax.Array]:
        pred = _Predictive(entry.numpyro_model, num_samples=n_particles)
        samples = pred(rng_key, *entry.model_args, **entry.model_kwargs)
        # Keep only latent sites (exclude observed data arrays), then map
        # from the model's constrained space to the unconstrained space
        # build_smc_logfns/postprocess_fn expect.
        latent = {k: v for k, v in samples.items() if k not in _obs_sites}
        return _numpyro_transform_fn(_transforms, latent, invert=True)

    return _predictive_prior_sample_fn
