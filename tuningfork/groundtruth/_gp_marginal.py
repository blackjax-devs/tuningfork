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
"""GP regression closed-form marginal GT generation.

The gp_regression model is 203-dimensional (3 hyperparameters + 200 GP values),
but the GP values ``f`` can be marginalised out analytically so that NUTS only
runs on the 3-dimensional hyperparameter marginal posterior.  ``f`` is then
reconstructed by drawing from the conditional Gaussian ``f | θ, y`` for each
``θ`` sample.  This reduces the effective sampler dimensionality by 67×.

Algorithm
---------
1. **Marginalise f analytically**: construct
   ``log p(y | θ) = log N(y; 0, K_θ + (JITTER + σ²) I)``
   where ``θ = (log_ls, log_ks, log_ns)`` and ``K_θ`` is the RBF kernel.

2. **Sample θ via NUTS** on this 3-dim log-posterior
   ``log p(θ | y) ∝ log p(y | θ) + log p(θ)``
   using ``window_adaptation(blackjax.nuts)``.  Per-chain starting positions
   are loaded from the committed ``provenance.init_positions.positions`` block.

3. **Reconstruct f | θ, y** by drawing from the conditional Gaussian
   ``N(μ_f, Σ_f)`` where ``μ_f = A (A + σ²I)^{-1} y``.
   Done in chunks of ``CHUNK_F=200`` to cap XLA intermediate buffer at ≤512 MB.

4. **Re-whiten to NCP coordinates**: ``f_raw = L_K^{-1} @ f`` so that the
   output ``draws.npz`` matches the parameterisation of the committed catalog GT.

Environment flags
-----------------
``GT_X64=1`` / ``JAX_ENABLE_X64=1``
    Required (model fails with NaN Cholesky at float32 precision).
``OPENBLAS_NUM_THREADS=1``
    Required on machines with 64+ cores.  OpenBLAS spawns threads for
    Cholesky; with many cores it can trigger a heap corruption (SIGABRT)
    from concurrent allocations inside the f-reconstruction loop.  Set this
    in the shell *before* the process starts.

Reproducibility
---------------
The committed ``provenance.init_positions.positions`` block stores the exact
per-chain starting ``θ`` used in the original run.  Using these positions makes
the regenerated ``θ`` chain statistically equivalent to the committed GT (given
the same seed and chain count) without requiring any external reference file.
``f_raw`` draws differ from the original in the conditional-Gaussian noise
(the ``f_key`` derivation changes if the seed changes), but are still drawn from
the same distribution.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tuningfork.groundtruth._emit import write_gt_artifacts
from tuningfork.groundtruth._nuts_multichain import (
    _load_explicit_positions,
    _run_nuts_multichain,
)

__all__ = ["generate_gp_marginal"]

#: Number of GP observations — must match the committed model.
_N_OBS: int = 200

#: Jitter added to kernel diagonal (same value as ``gp_regression.py``).
_JITTER: float = 1e-4

#: Chunk size for f-reconstruction (keeps XLA intermediates ≤ 512 MB).
_CHUNK_F: int = 200

_SMOKE_N_CHAINS: int = 2
_SMOKE_N_DRAWS: int = 50
_SMOKE_N_WARMUP: int = 100


# --------------------------------------------------------------------------- #
# Math kernel (pure JAX, x64-safe)
# --------------------------------------------------------------------------- #


def _rbf_kernel(X, log_lengthscale, log_kernel_scale):
    """RBF kernel: K[i,j] = ks² exp(-0.5 (xi-xj)² / ls²)."""
    import jax.numpy as jnp

    ls = jnp.exp(log_lengthscale)
    ks = jnp.exp(log_kernel_scale)
    sq = (X[:, None] - X[None, :]) ** 2
    return ks**2 * jnp.exp(-0.5 * sq / ls**2)


def _build_marginal_logdensity(X, y):
    """Return the 3-dim log-posterior for ``(log_ls, log_ks, log_ns)``.

    Marginalises ``f`` analytically:
    ``log p(y | θ) = log N(y; 0, A + σ²I)``
    where ``A = K_θ + JITTER·I`` and adds Gaussian log-priors that match the
    NumPyro model definition (see ``tuningfork.model.gp_regression``).
    """
    import jax.numpy as jnp

    n = len(y)
    eye_n = jnp.eye(n)

    def logdensity(params):
        log_ls = params["log_lengthscale"]
        log_ks = params["log_kernel_scale"]
        log_ns = params["log_noise_scale"]

        K = _rbf_kernel(X, log_ls, log_ks)
        A = K + _JITTER * eye_n
        sig2 = jnp.exp(2.0 * log_ns)
        B = A + sig2 * eye_n  # marginal cov of y

        L_B = jnp.linalg.cholesky(B)
        alpha = jnp.linalg.solve(L_B.T, jnp.linalg.solve(L_B, y))

        log_lik = (
            -0.5 * jnp.dot(y, alpha)
            - jnp.sum(jnp.log(jnp.diag(L_B)))
            - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )
        log_prior = -0.5 * log_ls**2 - 0.5 * log_ks**2 - 0.5 * (log_ns + 2.0) ** 2
        return log_lik + log_prior

    return logdensity


def _build_f_conditional_sampler(X, y):
    """Return a vmappable function that draws ``f | θ, y``.

    Each call allocates approximately ``7 × N_OBS × N_OBS`` float64
    intermediates.  Call in chunks of ``CHUNK_F`` to keep peak allocation
    bounded.
    """
    import jax
    import jax.numpy as jnp

    n = len(y)
    eye_n = jnp.eye(n)

    def sample_f(params, rng_key):
        log_ls = params["log_lengthscale"]
        log_ks = params["log_kernel_scale"]
        log_ns = params["log_noise_scale"]

        K = _rbf_kernel(X, log_ls, log_ks)
        A = K + _JITTER * eye_n
        sig2 = jnp.exp(2.0 * log_ns)
        B = A + sig2 * eye_n

        L_B = jnp.linalg.cholesky(B)
        alpha = jnp.linalg.solve(L_B.T, jnp.linalg.solve(L_B, y))
        mu_f = A @ alpha

        W = jnp.linalg.solve(L_B, A)
        Sigma_f = A - W.T @ W

        L_f = jnp.linalg.cholesky(Sigma_f + 1e-8 * eye_n)
        z = jax.random.normal(rng_key, (n,))
        return mu_f + L_f @ z

    return sample_f


def _f_to_f_raw(X, log_ls, log_ks, f):
    """NCP re-whitening: ``f_raw = L_K^{-1} @ f``.

    Inverse of ``f = L_K @ f_raw`` (lossless).  Vmappable.
    """
    import jax.numpy as jnp
    import jax.scipy.linalg

    K = _rbf_kernel(X, log_ls, log_ks) + _JITTER * jnp.eye(_N_OBS)
    L_K = jnp.linalg.cholesky(K)
    return jax.scipy.linalg.solve_triangular(L_K, f, lower=True)


# --------------------------------------------------------------------------- #
# Combined runner: NUTS on θ + f reconstruction
# --------------------------------------------------------------------------- #


def _run_gp_marginal(
    key,
    X,
    y,
    init_positions: dict,
    nc: int,
    nw: int,
    ns: int,
    target_acceptance: float,
    max_doublings: int,
    sequential: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """Run NUTS on 3-dim θ marginal, then reconstruct f_raw.

    Parameters
    ----------
    key
        Master RNG key (split internally for NUTS and f-reconstruction).
    X, y
        Observed data arrays (float64).
    init_positions
        ``{site: [chain0_val, ...]}`` in unconstrained space.
    nc, nw, ns
        Number of chains, warmup steps, draw steps.
    target_acceptance, max_doublings
        NUTS hyperparameters.
    sequential
        Run chains sequentially (avoids vmap memory pressure).

    Returns
    -------
    positions
        ``{site: ndarray (nc, ns, *event)}`` with sites
        ``log_lengthscale``, ``log_kernel_scale``, ``log_noise_scale``,
        ``f_raw``.
    diag
        Per-chain diagnostics dict.
    timing
        ``{"warmup", "sampling", "f_reconstruction"}`` in seconds.
    """
    import jax
    import jax.numpy as jnp

    logdensity_fn = _build_marginal_logdensity(X, y)

    # Stack per-chain inits
    sites = list(init_positions.keys())
    position_list = [
        {site: jnp.asarray(init_positions[site][i]) for site in sites}
        for i in range(nc)
    ]
    inits = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *position_list)

    k_run, k_f = jax.random.split(key, 2)

    # NUTS on 3-dim θ
    phi_positions, diag, timing = _run_nuts_multichain(
        k_run,
        inits,
        logdensity_fn,
        nc,
        nw,
        ns,
        target_acceptance,
        max_doublings,
        sequential,
    )
    # phi_positions: {log_ls: (nc, ns), log_ks: (nc, ns), log_ns: (nc, ns)}

    # Conditional f | θ, y reconstruction (chunked)
    sample_f_fn = _build_f_conditional_sampler(X, y)
    sample_f_chunk_jit = jax.jit(jax.vmap(sample_f_fn))
    f_to_f_raw_chunk_jit = jax.jit(
        jax.vmap(lambda fi, li, ki: _f_to_f_raw(X, li, ki, fi))
    )

    total_draws = nc * ns
    f_keys_all = jax.random.split(k_f, total_draws)

    # Flatten φ for chunked f-recon
    phi_flat = {
        s: jnp.array(phi_positions[s].reshape(-1), dtype=jnp.float64)
        for s in phi_positions
    }

    n_chunks = (total_draws + _CHUNK_F - 1) // _CHUNK_F
    print(
        f"[f recon] sampling {total_draws} f | θ, y  "
        f"in {n_chunks} chunks of ≤{_CHUNK_F}...",
        flush=True,
    )
    t0_f = time.perf_counter()
    f_chunks = []
    for ci, start in enumerate(range(0, total_draws, _CHUNK_F)):
        end = min(start + _CHUNK_F, total_draws)
        params_chunk = jax.tree.map(
            lambda x: jnp.array(x[start:end], dtype=jnp.float64), phi_flat
        )
        chunk_out = sample_f_chunk_jit(params_chunk, f_keys_all[start:end])
        f_chunks.append(np.array(chunk_out))
        if ci % 20 == 0 or ci == n_chunks - 1:
            elapsed = time.perf_counter() - t0_f
            print(
                f"  chunk {ci + 1}/{n_chunks} "
                f"({end}/{total_draws} draws)  {elapsed:.1f}s elapsed",
                flush=True,
            )

    f_marginal_np = np.concatenate(f_chunks, axis=0)  # (total_draws, N_OBS)

    # NCP re-whitening: f → f_raw, per chain (chunked)
    f_mc = f_marginal_np.reshape(nc, ns, _N_OBS)
    f_raw_mc = np.empty_like(f_mc)

    print(
        f"[f_raw]  converting {total_draws} f → f_raw (NCP re-whitening)...",
        flush=True,
    )
    for chain_idx in range(nc):
        ls_ch = phi_positions["log_lengthscale"][chain_idx]  # (ns,)
        ks_ch = phi_positions["log_kernel_scale"][chain_idx]
        f_ch = f_mc[chain_idx]  # (ns, N_OBS)
        raw_chunks = []
        for start in range(0, ns, _CHUNK_F):
            end = min(start + _CHUNK_F, ns)
            chunk_out = f_to_f_raw_chunk_jit(
                jnp.array(f_ch[start:end], dtype=jnp.float64),
                jnp.array(ls_ch[start:end], dtype=jnp.float64),
                jnp.array(ks_ch[start:end], dtype=jnp.float64),
            )
            raw_chunks.append(np.array(chunk_out))
        f_raw_mc[chain_idx] = np.concatenate(raw_chunks, axis=0)

    t_f = time.perf_counter() - t0_f
    print(f"[f_raw]  done in {t_f:.1f}s", flush=True)

    # Merge φ + f_raw
    positions: dict[str, np.ndarray] = dict(phi_positions)
    positions["f_raw"] = f_raw_mc  # (nc, ns, N_OBS)

    timing["f_reconstruction"] = t_f
    return positions, diag, timing


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def generate_gp_marginal(
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
    """Generate GP regression ground-truth via closed-form marginal sampling.

    Samples the 3-dim hyperparameter marginal posterior with NUTS, then
    reconstructs ``f`` from the conditional Gaussian and re-whitens to NCP
    coordinates.

    Parameters
    ----------
    model_name
        Registry model name — must be ``"gp_regression"``.
    committed_summary
        Parsed ``summary_v2.json`` for this model.
    out_dir
        Directory where ``draws.npz`` and ``summary_v2.json`` are written.
    seed
        Master RNG seed.  Defaults to the committed seed.
    n_chains
        Number of parallel chains.  Defaults to the committed value.
    n_draws
        Draws per chain.  Defaults to the committed value.
    n_warmup
        Warmup steps per chain.  Defaults to the committed value.
    sequential
        Run chains one at a time instead of via vmap.
    smoke
        Run at tiny scale (2 chains × 50 draws × 100 warmup) for fast
        validation.  Overrides ``n_chains``, ``n_draws``, ``n_warmup``.

    Returns
    -------
    dict
        Parsed ``summary_v2.json`` for the generated GT.

    Raises
    ------
    RuntimeError
        If ``JAX_ENABLE_X64`` is not set.
    """
    if model_name != "gp_regression":
        raise ValueError(
            f"generate_gp_marginal only handles 'gp_regression', got {model_name!r}"
        )

    import jax

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "gp_regression requires 64-bit floats (JAX_ENABLE_X64=1).  "
            "Set GT_X64=1 before starting the process, or prefix the command "
            "with JAX_ENABLE_X64=1."
        )

    if smoke:
        n_chains = _SMOKE_N_CHAINS
        n_draws = _SMOKE_N_DRAWS
        n_warmup = _SMOKE_N_WARMUP

    sc = committed_summary["sampler_config"]
    _nw = n_warmup if n_warmup is not None else sc.get("n_warmup_per_chain", 2000)
    _ta = sc.get("target_acceptance", 0.80)
    _md = sc.get("max_num_doublings", 10)
    _seed = seed if seed is not None else committed_summary["seeds"]["master_seed"]

    # Load per-chain θ init positions from committed provenance
    init_positions = _load_explicit_positions(committed_summary)
    sites_available = list(init_positions.keys())
    n_chains_committed = len(init_positions[sites_available[0]])

    _nc = n_chains if n_chains is not None else n_chains_committed

    # Truncate/validate chain count
    if _nc > n_chains_committed:
        raise ValueError(
            f"Requested n_chains={_nc} but committed init_positions only "
            f"has {n_chains_committed} chains."
        )
    if _nc < n_chains_committed:
        print(
            f"[note] using {_nc} of {n_chains_committed} committed init chains",
            file=sys.stderr,
            flush=True,
        )
        init_positions = {s: init_positions[s][:_nc] for s in init_positions}

    _nd = n_draws if n_draws is not None else committed_summary["n_draws_per_chain"]

    # Load data from the model module (x64-correct because JAX_ENABLE_X64 is set)
    from tuningfork.model.gp_regression import X_DATA, Y_DATA  # noqa: PLC0415

    key = jax.random.key(_seed)

    print(
        f"[start] {model_name} nc={_nc} nd={_nd} nw={_nw} ta={_ta} "
        f"x64={jax.config.read('jax_enable_x64')} "
        f"device={jax.devices()[0].platform} "
        f"init=explicit_positions sequential={sequential}",
        flush=True,
    )

    t_all = time.perf_counter()
    positions, diag, timing = _run_gp_marginal(
        key,
        X_DATA,
        Y_DATA,
        init_positions,
        _nc,
        _nw,
        _nd,
        _ta,
        _md,
        sequential,
    )

    _generator_str = "nuts_on_closed_form_gp_marginal_plus_conditional_f_reconstruction"
    sampler_config: dict[str, Any] = {
        "sampler": _generator_str,
        "warmup": "window_adaptation_diag_imm_perchain",
        "n_warmup_per_chain": _nw,
        "target_acceptance": _ta,
        "max_num_doublings": _md,
        "chunk_f": _CHUNK_F,
        "init_strategy": "explicit_positions",
        "execution": "sequential" if sequential else "vmap",
    }
    seeds_meta = {
        "master_seed": _seed,
        "derivation": (
            "key=jax.random.key(seed); split→(k_run, k_f); "
            "warm_keys=split(k_warm, n_chains); samp_keys=split(k_sample, n_chains); "
            "f_keys_all=split(k_f, total_draws)"
        ),
    }
    reproduced_from = {
        "timestamp_utc": committed_summary.get("provenance", {}).get("timestamp_utc"),
        "tuningfork_version": committed_summary.get("provenance", {}).get(
            "tuningfork_version"
        ),
    }
    extra_prov: dict[str, Any] = {
        "init_positions": committed_summary["provenance"]["init_positions"],
    }

    _, summary_path = write_gt_artifacts(
        out_dir,
        model_name=model_name,
        positions=positions,
        diag=diag,
        timing=timing,
        generator=_generator_str,
        space="unconstrained",
        sampler_config=sampler_config,
        seeds=seeds_meta,
        reproduced_from=reproduced_from,
        extra_provenance=extra_prov,
        total_wall=time.perf_counter() - t_all,
    )

    return json.loads(summary_path.read_text())
