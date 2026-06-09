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
"""MCLMC Low-Rank + Diagonal (LRD) warmup.

Pipeline
--------
1. Run a NUTS pilot chain to collect geometry samples
   (``pilot_n_warmup`` + ``pilot_n_samples`` steps, single chain).
2. Extract rank-k LRD components via ``extract_lrd_from_samples``
   (SVD of standardised pilot samples).
3. Build a ``LowRankInverseMassMatrix`` and bind it via ``make_lrd_kernel``.
4. Run ``blackjax.mclmc_find_L_and_step_size`` per chain
   (``jax.vmap`` over ``num_chains`` chains) with ``diagonal_preconditioning=False``
   so the LRD geometry is preserved throughout adaptation.

Multi-chain contract (mirrors ``mclmc_tuning``)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains=4, k_rank=10,
            pilot_n_warmup=1000, pilot_n_samples=1000, **kwargs)
    -> (states, adapted_params)

``adapted_params`` keys:

- ``"L"``                     : (num_chains,) — adapted trajectory lengths
- ``"step_size"``             : (num_chains,) — adapted step sizes
- ``"inverse_mass_matrix"``   : ``LowRankInverseMassMatrix`` with leading
                                ``num_chains`` axis (vmappable per-chain)
- ``"_total_tuning_steps"``   : int — grad evals in adaptation
                                (summed across chains)

The ``inverse_mass_matrix`` field is a ``LowRankInverseMassMatrix(sigma, U, lam)``
where ``sigma.shape=(num_chains, d)``, ``U.shape=(num_chains, d, k)``,
``lam.shape=(num_chains, k)``.  Broadcasting the single shared LRD IMM to a
leading ``num_chains`` axis lets ``jax.vmap`` slice it per chain — exactly the
same contract as the diagonal ``(num_chains, d)`` IMM from ``mclmc_tuning``.

The upstream blackjax ``isokinetic_mclachlan`` integrator dispatches natively on
``LowRankInverseMassMatrix`` (blackjax PR #936), so no logdensity_fn wrapping or
coordinate-change is required.
"""

from typing import Any

import blackjax
import blackjax.mcmc.mclmc
import jax
import jax.numpy as jnp
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork.warmup._base import Warmup, _maybe_replicate
from tuningfork.warmup._mclmc_common import _unpack_mclmc_adaptation

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    k_rank: int = 10,
    pilot_n_warmup: int = 1000,
    pilot_n_samples: int = 1000,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run the LRD-preconditioned MCLMC warmup pipeline.

    Parameters
    ----------
    rng_key
        JAX random key; split internally for pilot / init / tuning phases.
    init_position
        Initial unconstrained parameter dict (single chain; replicated internally).
    n_warmup
        LRD adaptation steps passed as ``num_steps`` to
        ``mclmc_find_L_and_step_size``.
    base_method
        ``BaseMethod`` entry for MCLMC (carried for interface uniformity; the
        LRD kernel overrides its factory via ``make_lrd_kernel``).
    logdensity_fn
        BlackJAX-compatible log-density function.  **Not wrapped or translated**
        — preconditioning is purely via the LRD mass matrix.
    num_chains
        Number of parallel chains (default 4).
    k_rank
        Rank of the LRD approximation (default 10).  Should be
        ``k <= min(d, pilot_n_samples)``; typical range 8–40 depending on model
        dimensionality and pilot sample quality.
    pilot_n_warmup
        Warmup steps for the single-chain NUTS pilot (default 1000).
    pilot_n_samples
        Post-warmup samples collected from the NUTS pilot (default 1000).
    **kwargs
        Ignored; present for interface uniformity.

    Returns
    -------
    states
        Post-adaptation ``MCLMCState``, batched over ``num_chains``.
    adapted_params
        Dict with keys ``"L"``, ``"step_size"``, ``"inverse_mass_matrix"``
        (batched ``LowRankInverseMassMatrix``), ``"_total_tuning_steps"``.
    """
    from tuningfork.base_method.mclmc import (
        extract_lrd_from_samples,
        make_lrd_kernel,
        run_pilot_nuts,
    )

    # Split rng_key into 2 phases: pilot / chain-init+tuning.
    # (chain-init and mclmc-tuning keys are derived from init_key via a further
    # 2*num_chains split below — no separate warmup_key needed.)
    pilot_key, init_key = jax.random.split(rng_key, 2)

    # ── Phase 1: NUTS pilot ───────────────────────────────────────────────────
    pilot_positions = run_pilot_nuts(
        logdensity_fn,
        init_position,
        pilot_key,
        n_warmup=pilot_n_warmup,
        n_samples=pilot_n_samples,
    )

    # ── Phase 2: Extract LRD components ──────────────────────────────────────
    _k = min(int(k_rank), int(pilot_n_samples))
    _, sigma, U, lam = extract_lrd_from_samples(pilot_positions, k=_k)
    lrd_imm = LowRankInverseMassMatrix(sigma=sigma, U=U, lam=lam)

    # ── Phase 3: MCLMC adaptation (vmapped over num_chains) ──────────────────
    # Replicate init_position to (num_chains, ...).
    init_positions = _maybe_replicate(init_position, num_chains)

    # Split keys: num_chains init keys + num_chains warmup keys.
    all_keys = jax.random.split(init_key, 2 * num_chains)
    chain_init_keys = all_keys[:num_chains]
    chain_warmup_keys = all_keys[num_chains:]

    # Bind LRD IMM into the kernel closure; mclmc_find_L_and_step_size receives
    # the LRD geometry via make_lrd_kernel regardless of what diagonal placeholder
    # the tuner passes (diagonal_preconditioning=False → identity placeholder).
    lrd_kernel = make_lrd_kernel(lrd_imm)

    @jax.vmap
    def _init_one(k: jax.Array, x0: Any) -> Any:
        return blackjax.mcmc.mclmc.init(x0, logdensity_fn, k)

    init_states = _init_one(chain_init_keys, init_positions)

    @jax.vmap
    def _tune_one(k: jax.Array, state: Any) -> tuple[Any, Any, Any]:
        s, adaptation_state, total_steps = blackjax.mclmc_find_L_and_step_size(
            lrd_kernel,
            num_steps=n_warmup,
            state=state,
            rng_key=k,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
        )
        return s, adaptation_state, total_steps

    states, adaptation_states, total_tuning_steps_per_chain = _tune_one(
        chain_warmup_keys, init_states
    )

    # ── Phase 4: Broadcast LRD IMM to (num_chains, ...) ──────────────────────
    # jax.vmap slices over the leading axis of each field in the NamedTuple, so
    # broadcasting the single shared LRD IMM gives each chain its own slice.
    sigma_b = jnp.broadcast_to(sigma[None], (num_chains,) + sigma.shape)
    U_b = jnp.broadcast_to(U[None], (num_chains,) + U.shape)
    lam_b = jnp.broadcast_to(lam[None], (num_chains,) + lam.shape)
    lrd_imm_batched = LowRankInverseMassMatrix(sigma=sigma_b, U=U_b, lam=lam_b)

    # ── Unpack and return ─────────────────────────────────────────────────────
    states_out, adapted = _unpack_mclmc_adaptation(
        states, adaptation_states, total_tuning_steps_per_chain
    )
    # Replace diagonal IMM from _unpack_mclmc_adaptation with the batched LRD IMM.
    adapted["inverse_mass_matrix"] = lrd_imm_batched
    return states_out, adapted


ENTRY = Warmup(
    name="mclmc_lrd_tuning",
    runner=_runner,
    compatible_methods=("mclmc",),
    notes=(
        "LRD-preconditioned MCLMC warmup.  Pipeline: "
        "(1) single-chain NUTS pilot (pilot_n_warmup + pilot_n_samples steps); "
        "(2) rank-k_rank SVD extraction via extract_lrd_from_samples; "
        "(3) mclmc_find_L_and_step_size with make_lrd_kernel (LowRankInverseMassMatrix). "
        "Dispatches natively on the upstream isokinetic_mclachlan integrator "
        "(blackjax PR #936) — no logdensity_fn wrapping. "
        "inverse_mass_matrix is a batched LowRankInverseMassMatrix with leading "
        "num_chains axis so jax.vmap slices it per chain. "
        "Recommended for ill-conditioned targets where diagonal mclmc_tuning fails."
    ),
)
