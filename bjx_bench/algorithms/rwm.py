"""RWM (Random-Walk Metropolis) algorithm entry for the bjx-bench algorithm registry.

Wraps ``blackjax.rmh`` with an isotropic Gaussian proposal parameterized by
``sigma`` (the proposal standard deviation).  Since ``blackjax.rmh`` expects a
``proposal_generator`` callable rather than a step-size scalar, this module
provides a thin factory wrapper that constructs the callable internally via
``jax.flatten_util.ravel_pytree`` so the position can be any JAX pytree.

Grad cost per step: 0.  RWM evaluates the log-density (no gradient) for the
MH accept/reject ratio.  Optimal target acceptance rate ≈ 0.234 (Gelman,
Roberts & Gilks 1996 / Roberts & Rosenthal 2001).
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace

__all__ = ["ENTRY", "_make_rwm"]


def _make_rwm(logdensity_fn: object, sigma: float) -> object:
    """Build a ``blackjax.rmh`` kernel with an isotropic Gaussian proposal.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density callable, same as for every other algorithm.
    sigma
        Proposal standard deviation.  The proposal is
        ``position + sigma * N(0, I)`` in the flattened parameter space,
        un-ravelled back to the original pytree structure.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX kernel object with ``.init`` and ``.step`` methods.

    Notes
    -----
    ``ravel_pytree`` is called once per ``proposal_generator`` invocation.
    The un-ravel function (``unravel``) is captured in the closure per call,
    which is safe because the pytree structure is fixed for a given model.
    """

    def proposal_generator(rng_key: jax.Array, position: object) -> object:
        flat, unravel = ravel_pytree(position)
        noise = jax.random.normal(rng_key, flat.shape) * sigma
        return unravel(flat + noise)

    return blackjax.rmh(logdensity_fn, proposal_generator)


ENTRY = AlgorithmEntry(
    name="rwm",
    family="mcmc",
    factory=_make_rwm,  # called as factory(logdensity_fn, sigma=...)
    grad_count_per_step=lambda info: jnp.asarray(0),
    default_hp_space=(HyperparamSpace("sigma", "loguniform", low=1e-3, high=10.0),),
    needs_mass_matrix=False,
    target_acceptance_rate=0.234,
    notes=(
        "Isotropic Gaussian proposal; sigma is the proposal scale. "
        "proposal_generator built internally via ravel_pytree so any JAX "
        "pytree position is supported (dict, flat array, NamedTuple, etc.). "
        "grad_count=0: RWM evaluates logdensity only, no gradient. "
        "Optimal accept ≈ 0.234 (Gelman, Roberts & Gilks 1996)."
    ),
)
