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
"""Standard multi-chain NUTS and explicit-positions GT generation.

Covers two generation paths:

- **standard_multichain_nuts** — 9 models (eight_schools_ncp, german_credit,
  horseshoe, irt_1pl, irt_2pl, lgcp, logistic_synthetic, radon, stoch_vol):
  per-chain ``blackjax.window_adaptation(blackjax.nuts)`` with
  ``init_to_uniform_radius2`` starts.  stoch_vol uses
  ``target_acceptance=0.95`` (read from its committed ``sampler_config``).

- **explicit_positions** — 1 model (lotka_volterra): same NUTS path but with
  per-chain starting positions loaded directly from the committed
  ``provenance.init_positions.positions`` block, making regeneration fully
  self-contained without any reference to external summary statistics.

Both paths use vmap for parallel chain execution by default (``--sequential``
fallback for models with heterogeneous tree-depth costs).  Progress checkpoints
are emitted via jaxtap at 25/50/75/100% of the draw scan.

Environment flags
-----------------
``GT_X64=1``
    Enable 64-bit floating point.  Required for ``lotka_volterra``
    (``requires_x64=True`` in the model registry).
``PYTHONUNBUFFERED=1``
    Recommended for long-running models (stoch_vol, lotka_volterra) to ensure
    per-step progress lines reach the log without buffering.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tuningfork.groundtruth._emit import _provenance_lineage, write_gt_artifacts

__all__ = ["generate_nuts_multichain"]

_SMOKE_N_CHAINS = 2
_SMOKE_N_DRAWS = 100
_SMOKE_N_WARMUP = 100


# --------------------------------------------------------------------------- #
# init strategies
# --------------------------------------------------------------------------- #


def _build_perchain_inits_uniform(entry, key, num_chains: int) -> tuple[dict, Any]:
    """Per-chain init_to_uniform_radius2 (Stan/posteriordb convention).

    Returns ``(stacked, ld_fn)`` where ``stacked`` is
    ``{site: (num_chains, *shape)}`` and ``ld_fn`` is the model log-density.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import initialize_model

    keys = jax.random.split(key, num_chains)
    positions = []
    ld_fn = None
    for k in keys:
        mi = initialize_model(
            k,
            entry.numpyro_model,
            model_args=entry.model_args,
            model_kwargs=entry.model_kwargs,
            dynamic_args=False,
        )
        positions.append(mi.param_info.z)
        if ld_fn is None:
            pf = mi.potential_fn
            ld_fn = lambda p, _pf=pf: -_pf(p)  # noqa: E731
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *positions)
    return stacked, ld_fn


def _build_perchain_inits_from_positions(
    entry, key, positions_dict: dict
) -> tuple[dict, Any, int]:
    """Init from ``provenance.init_positions.positions`` (explicit_positions path).

    Used by the ``explicit_positions`` path (lotka_volterra) to make
    regeneration self-contained.  Returns ``(stacked, ld_fn, n_chains)``
    where ``n_chains`` is read from ``positions_dict`` and overrides any
    caller-supplied value.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import initialize_model

    k_struct = jax.random.split(key, 1)[0]
    mi = initialize_model(
        k_struct,
        entry.numpyro_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    pf = mi.potential_fn
    ld_fn = lambda p, _pf=pf: -_pf(p)  # noqa: E731

    sites = list(positions_dict.keys())
    n_chains = len(positions_dict[sites[0]])
    position_list = []
    for i in range(n_chains):
        z = {site: jnp.asarray(positions_dict[site][i]) for site in sites}
        position_list.append(z)

    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *position_list)
    return stacked, ld_fn, n_chains


def _load_explicit_positions(committed_summary: dict) -> dict:
    """Return ``provenance.init_positions.positions`` from ``committed_summary``.

    Raises ``KeyError`` if the block is missing.
    """
    provenance = committed_summary.get("provenance", {})
    init_positions = provenance.get("init_positions", {})
    positions = init_positions.get("positions")
    if positions is None:
        raise KeyError(
            "committed_summary['provenance']['init_positions']['positions'] "
            "is missing — cannot use explicit_positions path."
        )
    return positions


# --------------------------------------------------------------------------- #
# progress checkpoints (jaxtap)
# --------------------------------------------------------------------------- #


def _make_progress_cb(t0: float, label: str, total_steps: int):
    """Return a jaxtap on_step callback emitting [progress] lines to stderr.

    Used for vmap-multichain sampling: fires at every ``sample_every`` steps
    of the outer inference scan (``jax.lax.scan`` of length ``ns`` inside
    ``blackjax.util.run_inference_algorithm``).  Step 0 fires when JIT
    compilation is done and real execution begins — the most important
    checkpoint for models with long compile times (lotka_volterra ~28 min).

    The total-steps filter suppresses the noisy inner NUTS tree-expansion
    scans (those have ``event.total != total_steps`` and would otherwise flood
    the log with ~60k events per run).
    """

    def cb(event):
        if event.total is not None and int(event.total) != total_steps:
            return
        step = event.step
        total = event.total if event.total is not None else total_steps
        pct = int(100 * (step + 1) / total) if total > 0 else 0
        elapsed = time.perf_counter() - t0
        print(
            f"[progress] {label} {step + 1}/{total} ({pct}%) elapsed={elapsed:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    return cb


# --------------------------------------------------------------------------- #
# NUTS multi-chain runner
# --------------------------------------------------------------------------- #


def _run_nuts_multichain(
    key,
    inits: dict,
    ld_fn: Any,
    nc: int,
    nw: int,
    ns: int,
    target_acceptance: float,
    max_doublings: int,
    sequential: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """Run per-chain window_adaptation + NUTS sampling.

    Parameters
    ----------
    key
        Master RNG key.
    inits
        Stacked positions ``{site: (nc, *shape)}``.
    ld_fn
        Log-density function.
    nc, nw, ns
        Chains, warmup steps, draw steps.
    target_acceptance
        NUTS target acceptance rate.
    max_doublings
        Maximum NUTS tree doublings.
    sequential
        If True, run chains one at a time instead of via vmap.  Useful for
        heterogeneous-cost models (e.g. lotka_volterra) where vmap's
        max-depth unrolling may cause OOM.

    Returns
    -------
    positions, diag, timing
        ``{site: (nc, ns, *event)}``, per-chain diagnostics dict,
        ``{"warmup": float, "sampling": float}`` seconds.
    """
    import blackjax
    import jax
    import jaxtap as tap

    k_warm, k_sample = jax.random.split(key, 2)
    warm_keys = jax.random.split(k_warm, nc)
    samp_keys = jax.random.split(k_sample, nc)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        ld_fn,
        target_acceptance_rate=target_acceptance,
        max_num_doublings=max_doublings,
    )

    def one_warmup(k, pos):
        (state, params), _ = warmup.run(k, pos, nw)
        return state, params

    def one_sample(k, state, step_size, imm):
        kernel = blackjax.nuts(
            ld_fn,
            step_size=step_size,
            inverse_mass_matrix=imm,
            max_num_doublings=max_doublings,
        )
        _last, (chain_states, infos) = blackjax.util.run_inference_algorithm(
            rng_key=k,
            initial_state=state,
            inference_algorithm=kernel,
            num_steps=ns,
        )
        return (
            chain_states.position,
            infos.is_divergent,
            infos.energy,
            infos.acceptance_rate,
        )

    if sequential:
        wj = jax.jit(one_warmup)
        sj = jax.jit(one_sample)
        pos_l, div_l, en_l, acc_l, ss_l = [], [], [], [], []
        warmup_wall = sampling_wall = 0.0
        t_all = time.perf_counter()
        for i in range(nc):
            init_i = jax.tree.map(lambda x, _i=i: x[_i], inits)
            t0 = time.perf_counter()
            st, pr = wj(warm_keys[i], init_i)
            jax.block_until_ready((st, pr))
            warmup_wall += time.perf_counter() - t0
            print(
                f"[progress] warmup chain {i + 1}/{nc} done "
                f"elapsed={time.perf_counter() - t_all:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            t0 = time.perf_counter()
            p, d, en, ac = sj(
                samp_keys[i], st, pr["step_size"], pr["inverse_mass_matrix"]
            )
            jax.block_until_ready((p, d, en, ac))
            sampling_wall += time.perf_counter() - t0
            pos_l.append(p)
            div_l.append(d)
            en_l.append(en)
            acc_l.append(ac)
            ss_l.append(float(pr["step_size"]))
            print(
                f"[progress] draws chain {i + 1}/{nc} ({int(100 * (i + 1) / nc)}%) "
                f"elapsed={time.perf_counter() - t_all:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        positions = {
            s: np.stack([np.asarray(pl[s]) for pl in pos_l], 0) for s in pos_l[0]
        }
        is_div = np.stack([np.asarray(x) for x in div_l], 0)
        energy = np.stack([np.asarray(x) for x in en_l], 0)
        accept = np.stack([np.asarray(x) for x in acc_l], 0)
        step_size = np.asarray(ss_l)
    else:
        t0 = time.perf_counter()
        states, params = jax.vmap(one_warmup)(warm_keys, inits)
        jax.block_until_ready((states, params))
        warmup_wall = time.perf_counter() - t0
        print(
            f"[progress] warmup {nw}/{nw} done elapsed={warmup_wall:.0f}s",
            file=sys.stderr,
            flush=True,
        )

        checkpoint_every = max(1, ns // 4)
        t0 = time.perf_counter()
        progress_cb = _make_progress_cb(t0, "draws", ns)
        with tap.record(
            select=lambda _: (),
            sample_every=checkpoint_every,
            ops=("scan",),
            on_step=progress_cb,
        ):
            positions_raw, is_div_raw, energy_raw, accept_raw = jax.vmap(one_sample)(
                samp_keys,
                states,
                params["step_size"],
                params["inverse_mass_matrix"],
            )
            jax.block_until_ready((positions_raw, is_div_raw, energy_raw, accept_raw))
        sampling_wall = time.perf_counter() - t0
        print(
            f"[progress] draws {ns}/{ns} (100%) elapsed={sampling_wall:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        positions = {s: np.asarray(a) for s, a in positions_raw.items()}
        step_size = np.asarray(params["step_size"])
        is_div = np.asarray(is_div_raw)
        energy = np.asarray(energy_raw)
        accept = np.asarray(accept_raw)

    e_arr = np.asarray(energy)  # (nc, ns)
    ebfmi = np.mean(np.diff(e_arr, axis=1) ** 2, axis=1) / np.var(e_arr, axis=1)
    diag: dict[str, Any] = {
        "step_size": np.asarray(step_size).tolist(),
        "divergences_per_chain": np.asarray(is_div).sum(axis=1).astype(int).tolist(),
        "e_bfmi_per_chain": [float(x) for x in ebfmi],
        "min_e_bfmi": float(np.min(ebfmi)),
        "mean_acceptance_per_chain": [
            float(x) for x in np.asarray(accept).mean(axis=1)
        ],
        "total_divergences": int(np.asarray(is_div).sum()),
    }
    timing = {"warmup": warmup_wall, "sampling": sampling_wall}
    return positions, diag, timing


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #


def generate_nuts_multichain(
    model_name: str,
    committed_summary: dict,
    out_dir: Path,
    *,
    seed: int | None = None,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_warmup: int | None = None,
    sequential: bool = False,
    smoke: bool = False,
) -> dict:
    """Generate multi-chain NUTS ground-truth draws.

    Handles both ``standard_multichain_nuts`` (init_to_uniform_radius2) and
    ``explicit_positions`` paths; the path is selected from
    ``committed_summary``'s generator field.

    Parameters
    ----------
    model_name
        Registry model name (e.g. ``"radon"`` or ``"lotka_volterra"``).
    committed_summary
        Parsed ``summary_v2.json`` for this model.
    out_dir
        Output directory.
    seed, n_chains, n_draws, n_warmup
        Override committed defaults; ``None`` = use committed value.
    sequential
        Run chains one at a time instead of via vmap.
    smoke
        Tiny-scale run (2 chains × 100 draws × 100 warmup); overrides
        ``n_chains``, ``n_draws``, ``n_warmup``.

    Returns
    -------
    dict
        Parsed ``summary_v2.json`` for the generated GT.
    """
    import jax

    from tuningfork.groundtruth._dispatch import GTMethod, _resolve_gt_method
    from tuningfork.model import MODELS

    entry = MODELS[model_name]

    if entry.requires_x64 and not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            f"Model {model_name!r} requires 64-bit floats but JAX_ENABLE_X64 is not "
            "set.  Set GT_X64=1 before starting the process, or prefix the command "
            "with JAX_ENABLE_X64=1."
        )

    if smoke:
        n_chains = _SMOKE_N_CHAINS
        n_draws = _SMOKE_N_DRAWS
        n_warmup = _SMOKE_N_WARMUP

    sc = committed_summary["sampler_config"]
    _nc = n_chains if n_chains is not None else committed_summary["n_chains"]
    _nd = n_draws if n_draws is not None else committed_summary["n_draws_per_chain"]
    _nw = n_warmup if n_warmup is not None else sc.get("n_warmup_per_chain", 2000)
    _ta = sc.get("target_acceptance", entry.reference_target_acceptance)
    _md = sc.get("max_num_doublings", 10)
    _seed = seed if seed is not None else committed_summary["seeds"]["master_seed"]

    key = jax.random.key(_seed)
    k_init, k_run = jax.random.split(key, 2)

    # Determine init strategy from dispatch
    method = _resolve_gt_method(committed_summary)
    if method is GTMethod.EXPLICIT_POSITIONS:
        positions_dict = _load_explicit_positions(committed_summary)
        inits, ld_fn, actual_nc = _build_perchain_inits_from_positions(
            entry, k_init, positions_dict
        )
        if actual_nc != _nc:
            print(
                f"[note] explicit_positions overrides n_chains: {_nc} → {actual_nc}",
                flush=True,
            )
            _nc = actual_nc
        _init_mode = "explicit_positions"
    else:
        inits, ld_fn = _build_perchain_inits_uniform(entry, k_init, _nc)
        _init_mode = "per_chain_init_to_uniform_radius2"

    print(
        f"[start] {model_name} nc={_nc} nd={_nd} nw={_nw} ta={_ta} "
        f"x64={jax.config.read('jax_enable_x64')} "
        f"device={jax.devices()[0].platform} "
        f"init={_init_mode} sequential={sequential}",
        flush=True,
    )

    t_all = time.perf_counter()
    positions, diag, timing = _run_nuts_multichain(
        k_run,
        inits,
        ld_fn,
        _nc,
        _nw,
        _nd,
        _ta,
        _md,
        sequential,
    )

    sampler_config: dict[str, Any] = {
        "sampler": "nuts",
        "warmup": "window_adaptation_diag_imm_perchain",
        "n_warmup_per_chain": _nw,
        "target_acceptance": _ta,
        "max_num_doublings": _md,
        "init_strategy": _init_mode,
        "execution": "sequential" if sequential else "vmap",
    }
    seeds_meta = {
        "master_seed": _seed,
        "derivation": (
            "key=jax.random.key(seed); split→(k_init, k_run); "
            "warm_keys=split(k_warm, n_chains); samp_keys=split(k_sample, n_chains)"
        ),
    }
    reproduced_from = _provenance_lineage(committed_summary)
    extra_prov: dict[str, Any] = {}
    if method is GTMethod.EXPLICIT_POSITIONS:
        extra_prov["init_positions"] = committed_summary["provenance"]["init_positions"]

    _, summary = write_gt_artifacts(
        out_dir,
        model_name=model_name,
        positions=positions,
        diag=diag,
        timing=timing,
        generator="nuts_perchain",
        space="unconstrained",
        sampler_config=sampler_config,
        seeds=seeds_meta,
        reproduced_from=reproduced_from,
        extra_provenance=extra_prov,
        total_wall=time.perf_counter() - t_all,
    )

    return summary
