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
"""MCLMC Low-Rank + Diagonal (LRD) warmup — thin delegate to blackjax.mclmc_lrd_warmup.

Delegates all geometry estimation, rank-guard logic, and inner-kernel dispatch
to the upstream Scheme A implementation (``blackjax.mclmc_lrd_warmup``, landed
in blackjax PR #937, SHA 359205da8b4c0f718662a64d6b9a2280fd8833b0).  This
module enforces the tuningfork runner contract (per-chain broadcast LRD IMM,
squeeze semantics, ``_total_tuning_steps``) and provides backward-compatible
parameter-name mapping.

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
- ``"_total_tuning_steps"``   : int — LRD adaptation steps per chain
                                (= ``n_warmup``; pilot steps not counted, for
                                historical continuity with calibration budget
                                accounting in ``_recipe_runner.py``)

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

__all__ = ["ENTRY"]

# frac_tune1=0.5 is hardcoded in the upstream certified recipe for the adjusted
# path (blackjax PR #937).  Accept this kwarg explicitly so callers that pass it
# do not have it silently dropped; reject any value other than 0.5.
_UPSTREAM_FRAC_TUNE1: float = 0.5


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
    # Adjusted-path kwargs — forwarded explicitly to upstream.
    inner_kernel: str = "mclmc",
    l_init_floor_factor: float = 1.15,
    adjusted_num_steps: int = 3000,
    # frac_tune1: upstream hardcodes 0.5 (certified recipe); accept here so
    # callers that pass it explicitly are never silently ignored.
    frac_tune1: float = _UPSTREAM_FRAC_TUNE1,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run the LRD-preconditioned MCLMC warmup pipeline via blackjax.mclmc_lrd_warmup.

    Delegates to the upstream Scheme A implementation.  The tuningfork contract
    (batched LRD IMM, per-chain L/step arrays, ``_total_tuning_steps``) is
    reconstructed from the upstream result.

    Parameters
    ----------
    rng_key
        JAX random key; split internally (upstream pipeline vs. state stubs).
    init_position
        Initial unconstrained parameter dict (single chain; replicated internally).
    n_warmup
        LRD adaptation steps passed as ``lrd_num_steps`` to the upstream.
    base_method
        ``BaseMethod`` entry (carried for interface uniformity; not used by
        this implementation — the upstream builds its own kernel internally).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of parallel chains (default 4).
    k_rank
        Requested LRD rank (``k`` in the upstream).  Hard-clamped to
        ``floor(n_eff / 2)`` if the pilot is under-mixed; see the upstream
        rank-guard logic.
    pilot_n_warmup
        Warmup steps for the diagonal MCLMC pilot (``pilot_num_warmup``
        in the upstream).  Default 1000.
    pilot_n_samples
        Post-warmup samples collected for the SVD geometry estimate
        (``pilot_num_samples`` in the upstream).  Default 1000.
    inner_kernel
        Inner kernel for the final tuning phase: ``"mclmc"`` (default, stable)
        or ``"adjusted_mclmc"`` (experimental).
    l_init_floor_factor
        L-init floor factor for the adjusted path (``floor_factor`` in the
        upstream).  Default 1.15; certified 3/3 on german_credit.  For stiff
        geometry where the oracle step exceeds the oracle L (e.g. ill_cond_50
        κ=1000), raise to ~1.5 and set ``adjusted_num_steps≥5000``.  The DA-
        ceiling ``UserWarning`` from the upstream is the runtime signal.
    adjusted_num_steps
        DA tuning steps for the adjusted phase-4 path (``adjusted_num_steps``
        in the upstream).  Default 3000 (certified config: 3000 ×
        ``frac_tune1=0.5`` = 1500 effective DA steps).
    frac_tune1
        Must equal 0.5 (the upstream certified value, hardcoded in the upstream
        adjusted path).  Accept here so callers that pass it explicitly are not
        silently ignored; raises ``ValueError`` for any other value.
    **kwargs
        No unexpected kwargs are accepted; raises ``TypeError`` if any are
        present (prevents silent drops of mis-spelled or unsupported kwargs).

    Returns
    -------
    states
        Freshly initialised ``MCLMCState``, batched over ``num_chains``.
        Most consumers discard these (``emit_mclmc_lrd._run_cert_seed`` assigns
        to ``_state``); use ``adapted_params`` for post-warmup positions.
    adapted_params
        Dict with keys ``"L"``, ``"step_size"``, ``"inverse_mass_matrix"``
        (batched ``LowRankInverseMassMatrix``), ``"_total_tuning_steps"``.
    """
    # ── Kwarg guards ───────────────────────────────────────────────────────────
    # Reject frac_tune1 values other than the upstream certified constant.
    if frac_tune1 != _UPSTREAM_FRAC_TUNE1:
        raise ValueError(
            f"frac_tune1={frac_tune1!r} is not supported; the upstream "
            f"mclmc_lrd_warmup hardcodes the certified value frac_tune1=0.5 "
            "(blackjax PR #937). Remove this kwarg or pass frac_tune1=0.5."
        )

    # Reject any remaining unexpected kwargs — never silently swallow.
    if kwargs:
        raise TypeError(
            "mclmc_lrd_tuning._runner received unexpected keyword arguments: "
            f"{sorted(kwargs)!r}"
        )

    # Split key: upstream pipeline gets a fresh sub-key; state stubs get another.
    upstream_key, states_key = jax.random.split(rng_key)

    # ── Delegate to upstream Scheme A ──────────────────────────────────────────
    result = blackjax.mclmc_lrd_warmup(
        logdensity_fn,
        init_position,
        upstream_key,
        k=k_rank,
        pilot_num_warmup=pilot_n_warmup,
        pilot_num_samples=pilot_n_samples,
        lrd_num_steps=n_warmup,
        num_chains=num_chains,
        inner_kernel=inner_kernel,
        floor_factor=l_init_floor_factor,
        adjusted_num_steps=adjusted_num_steps,
    )
    # result: MCLMCLRDAdaptationState(L, step_size, inverse_mass_matrix, diagnostics)
    # L and step_size are scalars (mean over chains, computed inside upstream).
    # inverse_mass_matrix is unbatched: sigma (d,), U (d,k), lam (k,).

    # ── Reconstruct tuningfork runner contract ─────────────────────────────────
    lrd_imm = result.inverse_mass_matrix

    # Broadcast unbatched LRD IMM → (num_chains, ...) for vmap sliceability.
    sigma_b = jnp.broadcast_to(lrd_imm.sigma[None], (num_chains,) + lrd_imm.sigma.shape)
    U_b = jnp.broadcast_to(lrd_imm.U[None], (num_chains,) + lrd_imm.U.shape)
    lam_b = jnp.broadcast_to(lrd_imm.lam[None], (num_chains,) + lrd_imm.lam.shape)
    lrd_imm_batched = LowRankInverseMassMatrix(sigma=sigma_b, U=U_b, lam=lam_b)

    # Broadcast scalar L / step_size → (num_chains,).
    L_arr = jnp.broadcast_to(result.L, (num_chains,))
    step_size_arr = jnp.broadcast_to(result.step_size, (num_chains,))

    adapted_params: dict[str, Any] = {
        "L": L_arr,
        "step_size": step_size_arr,
        "inverse_mass_matrix": lrd_imm_batched,
        # Per-chain LRD adaptation steps (= n_warmup).  Historical convention:
        # _recipe_runner.py multiplies by 2 * num_chains to derive calibration
        # budget; pilot steps are not included here (same as the pre-delegate
        # implementation).
        "_total_tuning_steps": int(n_warmup),
    }

    # ── Build state stubs ──────────────────────────────────────────────────────
    # The upstream does not expose final chain states; fresh init states satisfy
    # the runner contract.  Most consumers discard states entirely — only
    # adapted_params carries the meaningful output.
    init_keys = jax.random.split(states_key, num_chains)
    init_positions = _maybe_replicate(init_position, num_chains)

    @jax.vmap
    def _init_one(k: jax.Array, x0: Any) -> Any:
        return blackjax.mcmc.mclmc.init(x0, logdensity_fn, k)

    states = _init_one(init_keys, init_positions)

    return states, adapted_params


ENTRY = Warmup(
    name="mclmc_lrd_tuning",
    runner=_runner,
    compatible_methods=("mclmc",),
    notes=(
        "Scheme A pilot-free LRD-preconditioned MCLMC warmup — thin delegate to "
        "blackjax.mclmc_lrd_warmup (upstream PR #937, SHA 359205da). "
        "Pipeline: (1) single-chain diagonal MCLMC pilot; (2) rank-guard + SVD "
        "extraction of LowRankInverseMassMatrix; (3) multi-chain unadjusted LRD "
        "tuning (vmapped, L/step averaged); (4a) mclmc or (4b) adjusted_mclmc "
        "inner-kernel dispatch. All geometry estimation and rank-guard logic live "
        "in the upstream implementation; this wrapper enforces the tuningfork "
        "runner contract (per-chain broadcast LRD IMM, squeeze semantics, "
        "_total_tuning_steps) and provides backward-compatible parameter-name "
        "mapping. Recommended for ill-conditioned targets where diagonal "
        "mclmc_tuning fails."
    ),
)
