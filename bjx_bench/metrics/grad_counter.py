"""Gradient-evaluation aggregator for the headline metric.

This is the load-bearing primitive for ``bjx_bench.metrics.headline``
(T2.5).  It converts a per-step grad-count callable (carried on each
``BaseMethod``) into a scalar total across an entire chain.

BlackJAX ``run_inference_algorithm`` produces a ``(states, infos,
extra)`` tuple where ``infos`` is a NamedTuple-of-Arrays: each *field*
holds a rank-1 Array of shape ``(n_samples,)`` — one value per sampler
step.  ``jax.vmap`` over this NamedTuple broadcasts correctly because
vmap maps over the *leading axis* of every leaf; the resulting vmapped
call receives a NamedTuple whose every field is a scalar (or shape
matching the non-batch dims), exactly what a per-step callable expects.

Assumption documented here for T2.5 review:
    ``grad_count_per_step`` must:
    1. Accept a single-step info object (one scalar per counted field).
    2. Return a JAX scalar (``jnp.ndarray`` or Python ``int``); vmap
       lifts this to ``Array`` of shape ``(n_samples,)`` automatically.
    3. Be free of Python-level side effects (it will be vmapped and
       possibly JIT-compiled).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["total_grad_evals"]


def total_grad_evals(
    infos: Any,
    grad_count_per_step: Callable[[Any], Any],
) -> int:
    """Sum gradient evaluations across a chain.

    Parameters
    ----------
    infos
        A NamedTuple-of-Arrays as returned by BlackJAX
        ``run_inference_algorithm``.  Every field must have a leading
        axis of length ``n_samples``.
    grad_count_per_step
        A JAX-compatible callable mapping a *single-step* info object
        to the number of gradient evaluations for that step.  Must
        return a JAX scalar or Python ``int``; must be vmappable.

    Returns
    -------
    int
        Total gradient evaluations: ``sum over steps of
        grad_count_per_step(info_t)``.  Cast to Python ``int`` at the
        boundary so callers get a plain numeric type.

    Examples
    --------
    HMC / NUTS — cost varies per step:

    >>> total_grad_evals(infos, lambda i: i.num_integration_steps)

    MALA — constant 1 grad/step:

    >>> total_grad_evals(infos, lambda i: 1)

    RWM — no gradients:

    >>> total_grad_evals(infos, lambda i: 0)
    """
    counts = jax.vmap(grad_count_per_step)(infos)
    return int(jnp.sum(counts))
