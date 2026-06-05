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
"""Emit-time Python function for the inference loop section.

Replaces ``_templates/inference_loop.py.tmpl`` (113 LOC) and
``_templates/inference_loop_singlechain.py.tmpl`` (78 LOC), which were
already deleted in PR #152 when ``_build_inference_loop`` was first created.
This module moves that function into the ``_emit/`` package and adds
descriptor-driven dispatch via ``BaseMethod`` fields.

The function reads ``base_method.reinit_state`` and ``base_method.per_chain_param_keys``
directly from the registry entry instead of accepting them as flags — the same
descriptors the runner reads for live dispatch.

D8 compliant: emitted string contains no ``import tuningfork``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tuningfork.base_method._base import BaseMethod


def emit_inference_loop(
    base_method: BaseMethod,
    *,
    num_samples: int,
    sampler_seed: int,
    tuning_seed: int,
    num_chains: int,
    sampling_pb: bool,
    warmup_is_perchain: bool,
    warmup_init_is_single_chain: bool,
) -> str:
    """Build straight-line inference loop code with all branches resolved at emit time.

    T1.1: replaces inference_loop.py.tmpl + inference_loop_singlechain.py.tmpl.
    No try/except NameError probes — every flag is known at generation time.

    Descriptor-driven: reads ``base_method.reinit_state`` directly from the
    registry entry (instead of accepting it as an explicit flag) so the same
    descriptor the runner uses for live dispatch also gates the emit path.

    Parameters
    ----------
    base_method : BaseMethod
        Registry entry for the sampler. Used for:
        - ``base_method.reinit_state`` — whether to re-init state post-warmup.
        - ``base_method.per_chain_param_keys`` — adapted parameter keys; empty
          tuple for gradient-free methods (rwm, elliptical_slice, irmh, etc.).
    num_samples : int
        Number of post-warmup samples to draw.
    sampler_seed : int
        RNG seed for post-warmup sampling.
    tuning_seed : int
        RNG seed from the recipe (used for re-init keys).
    num_chains : int
        Number of parallel chains.
    sampling_pb : bool
        If True → single-chain loop (progress_bar=True, io_callback safe).
        If False → multi-chain loop (scan + vmap, no progress bar).
    warmup_is_perchain : bool
        Warmup ran per-chain (jax.vmap). Adapted params are (num_chains, ...).
    warmup_init_is_single_chain : bool
        no_warmup path. State initialised by sampler template; needs broadcast.
    """
    # Descriptor-driven dispatch: read reinit_state from the registry entry.
    needs_state_reinit = base_method.reinit_state

    lines: list[str] = []
    a = lines.append

    a(f"_NUM_SAMPLES = {num_samples}")

    if sampling_pb:
        # ── Single-chain path ────────────────────────────────────────────────
        a(
            "_SAMPLING_PROGRESS_BAR = True"
            "  # single-chain (progress bar safe); set False for multi-chain"
        )
        a("# Single-chain sampling (progress_bar=True).")
        a("# progress_bar uses io_callback inside the scan body.  io_callback is not")
        a(
            "# supported inside jax.vmap, so multi-chain sampling cannot use a progress bar."
        )
        a(
            "# We sample ONE chain then re-add a leading axis of 1 so downstream consumers"
        )
        a("# see shape (1, num_samples, ...) regardless of num_chains.")
        a("")

        # Step-size / IMM resolution
        if warmup_init_is_single_chain:
            # no_warmup: broadcast state, set defaults
            a(
                "_state_post_warmup = jax.tree.map("
                "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
                " _state_post_warmup)"
            )
            a('_shared_step_size = float(_adapted_params.get("step_size", 1.0))')
            a("from jax.flatten_util import ravel_pytree as _il_ravel")
            a("_il_flat, _ = _il_ravel(init_position)")
            a("_n_dims = int(_il_flat.shape[0])")
            a("_shared_imm = jnp.ones(_n_dims)")
        elif warmup_is_perchain:
            # Per-chain warmup — use per-chain params, pick chain-0 for single-chain run
            a('_shared_step_size = _adapted_params["step_size"][0]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"][0]')
        else:
            # Single-chain warmup → scalar params
            a('_shared_step_size = _adapted_params["step_size"]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"]')

        a("")
        a(
            "from blackjax.util import run_inference_algorithm as _run_inference_algorithm"
        )
        a("from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm")
        a("")
        a("# Extract single-chain state (chain 0).")
        a("_single_chain_state = jax.tree.map(lambda x: x[0], _state_post_warmup)")

        if needs_state_reinit:
            a("")
            a("# Re-init per sampler state type (dynamic_hmc / dmhmc / ghmc).")
            a(f"_single_reinit_key = jax.random.key({tuning_seed + 998})")
            a(
                "_single_chain_state = _state_reinit("
                "_shared_step_size, _shared_imm,"
                " _single_chain_state.position, _single_reinit_key)"
            )

        a("")
        a("_single_chain_step = kernel_builder(_shared_step_size, _shared_imm)")
        a(
            "_sc_alg = _SamplingAlgorithm(lambda *args, **kwargs: None,"
            " _single_chain_step)"
        )
        a("")
        a(
            "_sc_final_state, (_sc_states_hist, _sc_infos_hist) ="
            " _run_inference_algorithm("
        )
        a(f"    jax.random.key({sampler_seed}),")
        a("    _sc_alg,")
        a("    num_steps=_NUM_SAMPLES,")
        a("    initial_state=_single_chain_state,")
        a("    progress_bar=True,")
        a(")")
        a("")
        a("# Re-add the leading chain axis (size 1) for downstream shape consistency.")
        a("_samples = jax.tree.map(lambda x: x[None], _sc_states_hist)")
        a("_infos = jax.tree.map(lambda x: x[None], _sc_infos_hist)")

    else:
        # ── Multi-chain path ─────────────────────────────────────────────────
        a(
            "_SAMPLING_PROGRESS_BAR = False"
            "  # multi-chain scan+vmap; set True for single-chain with progress bar"
        )
        a("")

        # Step-size / IMM resolution
        if warmup_init_is_single_chain:
            # no_warmup: broadcast state, set defaults
            a(
                "# no_warmup: broadcast init_position-derived state to (num_chains, ...)."
            )
            a(
                "_state_post_warmup = jax.tree.map("
                "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
                " _state_post_warmup)"
            )
            a('_shared_step_size = float(_adapted_params.get("step_size", 1.0))')
            a("from jax.flatten_util import ravel_pytree as _il_ravel")
            a("_il_flat, _ = _il_ravel(init_position)")
            a("_n_dims = int(_il_flat.shape[0])")
            a("_shared_imm = jnp.ones(_n_dims)")
        elif not warmup_is_perchain:
            # Single-chain warmup → scalar shared params
            a("# Single-chain warmup: adapted params are scalar / un-batched.")
            a('_shared_step_size = _adapted_params["step_size"]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"]')

        if needs_state_reinit:
            a("")
            a(
                "# Re-init per-chain state (dynamic_hmc / dmhmc / ghmc: different state"
                " type than warmup)."
            )
            a(
                f"_reinit_keys = jax.random.split(jax.random.key({tuning_seed + 999}),"
                f" num_chains)"
            )
            if warmup_is_perchain:
                a('_batched_step_size = _adapted_params["step_size"]')
                a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
                a(
                    "_state_post_warmup = jax.vmap("
                    "lambda s, k, ss, imm: _state_reinit(ss, imm, s.position, k)"
                    ")(_state_post_warmup, _reinit_keys, _batched_step_size,"
                    " _batched_imm)"
                )
            else:
                a(
                    "_state_post_warmup = jax.vmap("
                    "lambda s, k: _state_reinit(_shared_step_size, _shared_imm,"
                    " s.position, k)"
                    ")(_state_post_warmup, _reinit_keys)"
                )

        a("")
        a("# Build the vmapped step function.")
        a(
            "from blackjax.util import run_inference_algorithm as _run_inference_algorithm"
        )
        a("from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm")
        a("")

        if warmup_is_perchain:
            a("# Per-chain warmup: each chain gets its own (step_size, imm).")
            a('_batched_step_size = _adapted_params["step_size"]')
            a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
            a("")
            a("def _step_one_chain(state, key, step_size, imm):")
            a("    return kernel_builder(step_size, imm)(key, state)")
            a("")
            a("def _vmapped_step(rng_key, states):")
            a(f"    keys = jax.random.split(rng_key, {num_chains})")
            a(
                "    return jax.vmap(_step_one_chain)(states, keys, _batched_step_size,"
                " _batched_imm)"
            )
        else:
            a("# Single-chain warmup: shared step_size + IMM across all chains.")
            a("_kernel_step = kernel_builder(_shared_step_size, _shared_imm)")
            a("")
            a("def _vmapped_step(rng_key, states):")
            a(f"    keys = jax.random.split(rng_key, {num_chains})")
            a("    return jax.vmap(_kernel_step)(keys, states)")

        a("")
        a("# run_inference_algorithm: (num_steps, num_chains, ...) output.")
        a("_alg = _SamplingAlgorithm(lambda *args, **kwargs: None, _vmapped_step)")
        a("_final_state, (_states_hist, _infos_hist) = _run_inference_algorithm(")
        a(f"    jax.random.key({sampler_seed}),")
        a("    _alg,")
        a("    num_steps=_NUM_SAMPLES,")
        a("    initial_state=_state_post_warmup,")
        a("    progress_bar=False,")
        a(")")
        a("")
        a("# Swap axes: (num_steps, num_chains, ...) -> (num_chains, num_steps, ...).")
        a("_samples = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _states_hist)")
        a("_infos = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _infos_hist)")

    return "\n".join(lines)


def _build_inference_loop(
    *,
    num_samples: int,
    sampler_seed: int,
    tuning_seed: int,
    num_chains: int,
    sampling_pb: bool,
    warmup_is_perchain: bool,
    warmup_init_is_single_chain: bool,
    needs_state_reinit: bool,
) -> str:
    """Backward-compatible shim: wraps emit_inference_loop with explicit flags.

    Used internally by _emit_script.py while the caller-side transition is
    in progress. New callers should use emit_inference_loop() directly.
    """
    # Build a minimal synthetic BaseMethod with the correct reinit_state flag.
    # We only need reinit_state for the shim; other fields are taken from NUTS default.
    _bm: Any = type(
        "_SyntheticBaseMethod",
        (),
        {"reinit_state": needs_state_reinit, "per_chain_param_keys": ()},
    )()

    return emit_inference_loop(
        _bm,
        num_samples=num_samples,
        sampler_seed=sampler_seed,
        tuning_seed=tuning_seed,
        num_chains=num_chains,
        sampling_pb=sampling_pb,
        warmup_is_perchain=warmup_is_perchain,
        warmup_init_is_single_chain=warmup_init_is_single_chain,
    )
