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
"""Stan-style window-adaptation warmup, wrapping ``blackjax.window_adaptation``.

This warmup runs dual-averaging step-size adaptation together with mass-matrix
estimation, matching the Stan HMC/NUTS default.  Mass-matrix shape is
controlled by the ``is_mass_matrix_diagonal`` keyword (default ``True``,
matching upstream BlackJAX); set to ``False`` for **dense** (full-rank)
adaptation when posterior correlation is the dominant pathology.

Compatible with any BlackJAX kernel that accepts an ``inverse_mass_matrix``
keyword argument (HMC, NUTS, Barker, MALA — verified by tripwire tests
in ``tests/test_api_pins_mcmc.py``).

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, target_acceptance_rate=None,
            is_mass_matrix_diagonal=True, num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains ``"step_size"`` (shape ``(num_chains,)``) and
  ``"inverse_mass_matrix"`` (shape ``(num_chains, d)`` for diagonal or
  ``(num_chains, d, d)`` for dense).  Per-chain values are returned (not
  averaged), so downstream callers can average if desired.

The ``adapted_params`` dict always contains at least ``"step_size"``
and ``"inverse_mass_matrix"`` on successful adaptation.  When
``is_mass_matrix_diagonal=True`` the per-chain IMM has shape ``(d,)``
(stacked to ``(num_chains, d)`` in the output).  When ``False`` the
per-chain IMM has shape ``(d, d)`` (stacked to ``(num_chains, d, d)``).
HIGH-effort recipes that adapt a dense or large-diagonal IMM should
persist it via ``Recipe.save_imm_sidecar`` rather than inlining.

If the ``base_method`` has a BO-tunable HP that is NOT step_size or
inverse_mass_matrix (e.g. ``num_integration_steps`` for HMC), the
default value for that HP is injected into the ``window_adaptation``
call so the warmup kernel can construct itself; BO trials later override
those HPs via trial_params.
"""

from typing import Any

import blackjax
import jax

from tuningfork.warmup._base import Warmup, _maybe_replicate
from tuningfork.warmup._laplace_adapter import resolve_warmup_algorithm

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    target_acceptance_rate: float | None = None,
    is_mass_matrix_diagonal: bool = True,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run blackjax.window_adaptation over ``num_chains`` chains via vmap.

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
    is_mass_matrix_diagonal
        ``True`` (default) — Stan-style diagonal mass matrix; per-chain adapted
        IMM has shape ``(d,)``; output IMM has shape ``(num_chains, d)``.
        ``False`` — dense full-rank mass matrix; per-chain IMM has shape
        ``(d, d)``; output IMM has shape ``(num_chains, d, d)``.  Use
        ``False`` only when posterior correlation is the dominant pathology AND
        ``d`` is small enough that a ``d × d`` matrix is tractable (rule of
        thumb: d ≲ 200; consult ``Recipe.save_imm_sidecar`` for storage of
        large dense matrices).
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``, matching Stan/NumPyro convention.  Pass ``num_chains=1``
        explicitly for BO trials (intentionally single-chain — chain count is
        orthogonal to HP tuning).
    **kwargs
        Additional keyword arguments forwarded to ``window_adaptation``
        (e.g. ``num_integration_steps`` for HMC — the warmup kernel needs
        it to build its leapfrog integrator, even though BO will tune it
        later).

    Returns
    -------
    states
        Post-warmup BlackJAX kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with at least ``"step_size"`` and ``"inverse_mass_matrix"``.
        ``"step_size"`` has shape ``(num_chains,)``.
        ``"inverse_mass_matrix"`` has shape ``(num_chains, d)`` for diagonal
        or ``(num_chains, d, d)`` for dense.
    """
    from tuningfork.calibration.tune import default_value_for_space

    target = target_acceptance_rate or base_method.target_acceptance_rate or 0.80

    # Build extra kwargs for the warmup call: inject default values for any
    # HP that the kernel needs during warmup but is NOT step_size or
    # inverse_mass_matrix (those come from the adaptation itself).
    extra_kwargs: dict[str, Any] = dict(kwargs)  # caller-supplied overrides first
    for space in base_method.default_hp_space:
        if space.name not in ("step_size", "inverse_mass_matrix"):
            if space.name not in extra_kwargs:
                extra_kwargs[space.name] = default_value_for_space(space)

    # For laplace_* base methods, substitute blackjax.hmc as the warmup
    # algorithm so that window_adaptation receives a proper algorithm object
    # with .build_kernel and .init(position, logdensity_fn).  The caller
    # is responsible for passing the laplace marginal logdensity
    # (phi → float) as logdensity_fn — this adapter does not build it.
    warmup_algorithm, warmup_kwargs = resolve_warmup_algorithm(
        base_method, extra_kwargs
    )

    warmup = blackjax.window_adaptation(
        warmup_algorithm,
        logdensity_fn,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        target_acceptance_rate=target,
        **warmup_kwargs,
    )

    # Split the key for num_chains independent runs.
    chain_keys = jax.random.split(rng_key, num_chains)

    # Replicate init_position across chains.  Pass-through if pre-batched.
    init_positions = _maybe_replicate(init_position, num_chains)

    # vmap the warmup.run over (key, init_position).
    @jax.vmap
    def run_one(k: jax.Array, x0: Any) -> tuple[Any, Any]:
        (state, params), _info = warmup.run(k, x0, n_warmup)
        return state, params

    states, adapted_params = run_one(chain_keys, init_positions)
    return states, dict(adapted_params)


ENTRY = Warmup(
    name="window_adaptation_diag_imm",
    runner=_runner,
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
        "Standard Stan window adaptation: dual-averaging step_size + diagonal "
        "mass matrix.  Compatible with hmc, nuts, mhmc, dynamic_hmc, dmhmc, "
        "barker, mala, and laplace_* variants (all kernels that accept "
        "inverse_mass_matrix; mhmc/dynamic_hmc/dmhmc verified by RECIPE_GENERATION.md "
        "Table 1A note + needs_mass_matrix=True in their registry entries).  "
        "multi-chain by default (num_chains=4 via jax.vmap); "
        "per-chain adapted_params returned (step_size shape (num_chains,), IMM "
        "shape (num_chains, d) or (num_chains, d, d))."
    ),
)
