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
"""Stan-style window-adaptation warmup with dense (full-rank) mass matrix.

This warmup wraps ``blackjax.window_adaptation`` with the
``is_mass_matrix_diagonal=False`` flag pinned, enabling full-rank mass matrix
estimation when posterior correlation is the dominant pathology.

Unlike ``window_adaptation_diag_imm``, the ``is_mass_matrix_diagonal`` keyword
is NOT exposed; it is fixed to ``False``. This enforces the design constraint
that dense adaptation is an intentional choice, not an incidental override.

Runs dual-averaging step-size adaptation together with full-rank mass-matrix
estimation, matching the Stan HMC/NUTS default with the diagonal constraint
removed.

Compatible with any BlackJAX kernel that accepts an ``inverse_mass_matrix``
keyword argument (HMC, NUTS, Barker, MALA — verified by tripwire tests
in ``tests/test_api_pins_mcmc.py``).

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, target_acceptance_rate=None,
            num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains ``"step_size"`` (shape ``(num_chains,)``) and
  ``"inverse_mass_matrix"`` (shape ``(num_chains, d, d)`` for dense).
  Per-chain values are returned (not averaged), so downstream callers can
  average if desired.

The ``adapted_params`` dict always contains at least ``"step_size"``
and ``"inverse_mass_matrix"`` on successful adaptation.  The per-chain IMM
has shape ``(d, d)`` (stacked to ``(num_chains, d, d)`` in the output).
HIGH-effort recipes that adapt a dense IMM should persist it via
``Recipe.save_imm_sidecar`` rather than inlining.

If the ``base_method`` declares a parameter that is NOT step_size or
inverse_mass_matrix (e.g. ``num_integration_steps`` for HMC), the
default value for that HP is injected into the ``window_adaptation``
call so the warmup kernel can construct itself; generated recipe resolution
supplies the recorded value at sampling time.
"""

from typing import Any

import blackjax
import jax

from tuningfork.warmup._base import Warmup
from tuningfork.warmup._window_adaptation_common import _window_adaptation_body

__all__ = ["ENTRY"]


def _build_dense_warmup(
    warmup_algorithm: Any,
    logdensity_fn: Any,
    *,
    target_acceptance_rate: float,
    **warmup_kwargs: Any,
) -> Any:
    """Build ``blackjax.window_adaptation`` pinned to dense mass matrix."""
    return blackjax.window_adaptation(
        warmup_algorithm,
        logdensity_fn,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=target_acceptance_rate,
        **warmup_kwargs,
    )


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    target_acceptance_rate: float | None = None,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any], Any]:
    """Run blackjax.window_adaptation with dense mass matrix over ``num_chains`` chains via vmap.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Split internally into
        ``num_chains`` independent per-chain keys.
    init_position
        Initial unconstrained parameter dict (from the model's prior sample).
        A SINGLE pytree (one chain's worth).  The runner replicates it across
        ``num_chains`` unless the caller pre-batches it (leading dim ==
        ``num_chains``).
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (carries ``factory``, ``default_hp_space``,
        ``target_acceptance_rate``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    target_acceptance_rate
        Override for the dual-averaging target.  Falls back to
        ``base_method.target_acceptance_rate``, then ``0.80``.
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``, matching Stan/NumPyro convention.  Pass ``num_chains=1``
        explicitly for isolated adaptation checks (chain count is orthogonal to
        parameter resolution).
    **kwargs
        Additional keyword arguments forwarded to ``window_adaptation``
        (e.g. ``num_integration_steps`` for HMC — the warmup kernel needs
        it to build its leapfrog integrator; generated recipes supply the
        recorded value later).

    Returns
    -------
    states
        Post-warmup BlackJAX kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with at least ``"step_size"`` and ``"inverse_mass_matrix"``.
        ``"step_size"`` has shape ``(num_chains,)``.
        ``"inverse_mass_matrix"`` has shape ``(num_chains, d, d)`` (dense).
    """
    return _window_adaptation_body(
        rng_key,
        init_position,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
        num_chains=num_chains,
        warmup_builder_fn=_build_dense_warmup,
        **kwargs,
    )


ENTRY = Warmup(
    name="window_adaptation_dense_imm",
    runner=_runner,  # type: ignore[arg-type]
    compatible_methods=(
        "hmc",
        "nuts",
        "mhmc",
        "dynamic_hmc",
        "dmhmc",
        "barker",
        "mala",
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
    ),
    notes=(
        "Stan-style window adaptation with dense (full-rank) inverse mass matrix. "
        "Compatible with hmc, nuts, mhmc, barker, mala, and laplace_* variants "
        "(all kernels that accept inverse_mass_matrix; mhmc verified by "
        "RECIPE_GENERATION.md Table 1A note + needs_mass_matrix=True in registry).  "
        "NOT compatible with dynamic_hmc/dmhmc: their DynamicHMCState requires "
        "random_generator_arg at init time; blackjax.window_adaptation calls "
        "algorithm.init(position, logdensity_fn) without that arg -- needs an "
        "adapter (similar to _laplace_adapter) to be composed properly.  "
        "Use when posterior correlation is the dominant pathology.  multi-chain "
        "by default (num_chains=4 via jax.vmap); per-chain adapted_params "
        "returned (step_size shape (num_chains,), dense IMM shape (num_chains, d, d))."
    ),
)
