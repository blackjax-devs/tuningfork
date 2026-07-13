"""Multi-chain ground-truth regeneration driver (staging only; no catalog writes).

Extends gt_regen_driver_with_target_accept.py with posterior-scale dispersed init.

DESIGN REQUIREMENT (--init-posterior flag):
  Dispersion must be WIDE enough that between-chain R-hat is meaningful
  (un-hollowed — the whole point of #223/#225 multichain GT), but at
  POSTERIOR scale, not the init_to_uniform radius-2 PRIOR scale that threw
  3 of 10 lotka_volterra chains into the ODE-unstable region in the failed
  dispersed run (R-hat 2.67, 8686 divergences).

  The 0.1-jitter point-anchor (gt_regen_driver_mode_anchored.py) was
  REJECTED because it yields R-hat-hollow chains (all start near the same
  point → between-chain variance is an artifact of jitter width, not
  genuine posterior uncertainty).

New flags added to this driver:

  --init-posterior PATH   Path to posterior stats for warm dispersion.
                          Two modes auto-detected by extension:
                          * .npz  -> init-from-draws: subsample num_chains
                            rows from certified reference draws (unconstrained
                            space).  Gold standard: chains are literal
                            posterior samples → maximally honest between-chain
                            R-hat.
                          * .json -> dispersed-Gaussian: per-site
                            N(mean, (dispersion_scale * std)²) using
                            reference/summary.json's mean + std scalars.
                            Fallback when draws are not available.
                          Default None uses overdispersed init_to_uniform
                          (original Stan/posteriordb convention).

  --dispersion-scale F    Multiplier on posterior std for the dispersed-
                          Gaussian mode.  Default 1.5.  Produces chains
                          spread at ≈2.25× posterior variance — genuinely
                          dispersed within the posterior support rather than
                          at prior scale.  Ignored in draws mode.

Also carries --target-accept from the prior patch.

CLI:
  python gt_regen_driver_posterior_dispersed.py <model>
     [--n-chains 10] [--n-draws 10000] [--n-warmup 2000]
     [--seed 20260713] [--max-doublings 10] [--out-dir DIR]
     [--sequential] [--smoke] [--target-accept FLOAT]
     [--init-posterior PATH] [--dispersion-scale FLOAT]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("GT_X64") == "1":
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import initialize_model

import tuningfork
from tuningfork.model import MODELS
from tuningfork.model._base import ReferenceMethod

SCHEMA_VERSION = "gt_v2_multichain"


# --------------------------------------------------------------------------- #
# init strategies
# --------------------------------------------------------------------------- #
def build_perchain_inits(entry, key, num_chains):
    """num_chains independent init_to_uniform draws (Stan/posteriordb convention).

    Each chain gets an independent unconstrained init from numpyro's default
    init strategy (init_to_uniform, radius 2) with a distinct key -> genuinely
    dispersed, per-chain independent starts. Returns stacked pytree
    {site: (num_chains, *shape)} + logdensity_fn + postprocess_fn.
    """
    keys = jax.random.split(key, num_chains)
    positions, ld_fn, pp_fn = [], None, None
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
            ld_fn = lambda p, _pf=pf: -_pf(p)
            pp_fn = mi.postprocess_fn
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *positions)
    return stacked, ld_fn, pp_fn


def build_perchain_inits_from_draws(entry, key, num_chains, draws_path):
    """Inits subsampled from certified reference draws (unconstrained space).

    Uses evenly-spaced indices across the full draw set so the num_chains
    starting points span the full posterior run.  These ARE genuine posterior
    samples -> maximally honest between-chain R-hat.
    """
    k_struct = jax.random.split(key, 1)[0]
    mi = initialize_model(
        k_struct,
        entry.numpyro_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    pf = mi.potential_fn
    ld_fn = lambda p, _pf=pf: -_pf(p)
    pp_fn = mi.postprocess_fn

    raw = np.load(draws_path)
    first_site = raw.files[0]
    n_total = raw[first_site].shape[0]
    idx = np.round(np.linspace(0, n_total - 1, num_chains)).astype(int)

    positions = []
    for i in idx:
        z = {site: jnp.asarray(raw[site][i]) for site in raw.files}
        positions.append(z)

    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *positions)
    return stacked, ld_fn, pp_fn


def build_perchain_inits_posterior_dispersed(
    entry, key, num_chains, summary_stats, dispersion_scale
):
    """Inits drawn from N(mean, (dispersion_scale * std)^2) per site.

    DESIGN REQUIREMENT: dispersion at POSTERIOR scale (not prior scale).
    dispersion_scale=1.5 produces chains spread at ~2.25x posterior variance
    — wide enough for un-hollowed between-chain R-hat, narrow enough to stay
    within the posterior support and avoid ODE-unstable regions.

    summary_stats is a dict with keys 'mean' and 'std', each mapping site
    names to float (scalar params) or list (vector params).  Source:
    catalog/<model>/reference/summary.json.
    """
    k_struct, *k_chains = jax.random.split(key, num_chains + 1)
    mi = initialize_model(
        k_struct,
        entry.numpyro_model,
        model_args=entry.model_args,
        model_kwargs=entry.model_kwargs,
        dynamic_args=False,
    )
    pf = mi.potential_fn
    ld_fn = lambda p, _pf=pf: -_pf(p)
    pp_fn = mi.postprocess_fn

    sites = list(summary_stats["mean"].keys())
    positions = []
    for k_chain in k_chains:
        k_sites = jax.random.split(k_chain, len(sites))
        z = {}
        for k_site, site in zip(k_sites, sites):
            mean_v = jnp.asarray(summary_stats["mean"][site])
            std_v = jnp.asarray(summary_stats["std"][site])
            noise = (
                dispersion_scale * std_v * jax.random.normal(k_site, shape=mean_v.shape)
            )
            z[site] = mean_v + noise
        positions.append(z)

    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *positions)
    return stacked, ld_fn, pp_fn


# --------------------------------------------------------------------------- #
# nuts run
# --------------------------------------------------------------------------- #
def run_nuts_multichain(
    entry,
    key,
    nc,
    nw,
    ns,
    target_acceptance,
    max_doublings,
    sequential=False,
    init_posterior=None,
    dispersion_scale=1.5,
):
    """Per-chain warmup + sampling (vmap, or sequential loop). Returns (positions, diag, timing)."""
    k_init, k_warm, k_sample = jax.random.split(key, 3)
    if init_posterior is not None:
        path = Path(init_posterior)
        if path.suffix == ".npz":
            inits, ld_fn, _pp = build_perchain_inits_from_draws(
                entry, k_init, nc, init_posterior
            )
        else:
            with open(init_posterior) as f:
                summary_stats = json.load(f)
            inits, ld_fn, _pp = build_perchain_inits_posterior_dispersed(
                entry, k_init, nc, summary_stats, dispersion_scale
            )
    else:
        inits, ld_fn, _pp = build_perchain_inits(entry, k_init, nc)

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

    warm_keys = jax.random.split(k_warm, nc)
    samp_keys = jax.random.split(k_sample, nc)

    if sequential:
        wj = jax.jit(one_warmup)
        sj = jax.jit(one_sample)
        pos_l, div_l, en_l, acc_l, ss_l = [], [], [], [], []
        warmup_wall = sampling_wall = 0.0
        for i in range(nc):
            init_i = jax.tree.map(lambda x, _i=i: x[_i], inits)
            t0 = time.perf_counter()
            st, pr = wj(warm_keys[i], init_i)
            jax.block_until_ready((st, pr))
            warmup_wall += time.perf_counter() - t0
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
                f"[seq] chain {i + 1}/{nc} done div={int(np.asarray(d).sum())}",
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
        t0 = time.perf_counter()
        positions, is_div, energy, accept = jax.vmap(one_sample)(
            samp_keys, states, params["step_size"], params["inverse_mass_matrix"]
        )
        jax.block_until_ready((positions, is_div, energy, accept))
        sampling_wall = time.perf_counter() - t0
        positions = {s: np.asarray(a) for s, a in positions.items()}
        step_size = np.asarray(params["step_size"])

    e = np.asarray(energy)  # (nc, ns)
    ebfmi = np.mean(np.diff(e, axis=1) ** 2, axis=1) / np.var(e, axis=1)
    is_div = np.asarray(is_div)
    diag = {
        "step_size": step_size.tolist(),
        "divergences_per_chain": is_div.sum(axis=1).astype(int).tolist(),
        "e_bfmi_per_chain": [float(x) for x in ebfmi],
        "mean_acceptance_per_chain": [
            float(x) for x in np.asarray(accept).mean(axis=1)
        ],
        "total_divergences": int(is_div.sum()),
    }
    return positions, diag, {"warmup": warmup_wall, "sampling": sampling_wall}


def run_analytic_multichain(entry, key, nc, ns):
    """10 independent i.i.d. batches from the exact analytic sampler."""
    keys = jax.random.split(key, nc)
    t0 = time.perf_counter()
    batches = [entry.analytic_sampler(k, ns) for k in keys]
    jax.block_until_ready(batches)
    wall = time.perf_counter() - t0
    sites = batches[0].keys()
    positions = {
        s: np.stack([np.asarray(b[s]) for b in batches], axis=0) for s in sites
    }
    diag = {"generator": "analytic_iid", "total_divergences": 0}
    return positions, diag, {"warmup": 0.0, "sampling": wall}


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def summarize(positions):
    """positions: {site: (nc, ns, *event)} -> per-site stats + gate scalars."""
    idata = az.from_dict({"posterior": positions}, sample_dims=["chain", "draw"])
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    rhat = az.rhat(idata, method="rank")

    per_site, max_rhat, min_bulk = {}, 0.0, np.inf
    for site, arr in positions.items():
        a = np.asarray(arr)
        nc, ns = a.shape[0], a.shape[1]
        flat = a.reshape(nc, ns, -1)
        chain_means = flat.mean(axis=1)
        pooled = flat.reshape(-1, flat.shape[-1])
        be_se = chain_means.std(axis=0, ddof=1) / np.sqrt(nc)
        b_ess = np.atleast_1d(np.asarray(ess_bulk[site])).ravel()
        t_ess = np.atleast_1d(np.asarray(ess_tail[site])).ravel()
        r = np.atleast_1d(np.asarray(rhat[site])).ravel()
        per_site[site] = {
            "mean": pooled.mean(axis=0).tolist(),
            "std": pooled.std(axis=0, ddof=1).tolist(),
            "q05": np.quantile(pooled, 0.05, axis=0).tolist(),
            "q95": np.quantile(pooled, 0.95, axis=0).tolist(),
            "between_chain_se": be_se.tolist(),
            "bulk_ess": b_ess.tolist(),
            "tail_ess": t_ess.tolist(),
            "rhat": r.tolist(),
        }
        max_rhat = max(max_rhat, float(np.nanmax(r)))
        min_bulk = min(min_bulk, float(np.nanmin(b_ess)))
    return per_site, max_rhat, min_bulk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n-chains", type=int, default=10)
    ap.add_argument("--n-draws", type=int, default=10000)
    ap.add_argument("--n-warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--max-doublings", type=int, default=10)
    ap.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help="Override NUTS target acceptance rate. "
        "Default None uses entry.reference_target_acceptance.",
    )
    ap.add_argument("--out-dir", default="/tmp/gt_regen")
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="loop chains instead of vmap (fast for funnel/treedepth-"
        "heterogeneous targets where vmap pays the max-depth penalty)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--init-posterior",
        default=None,
        metavar="PATH",
        help="Path to posterior stats for warm dispersion.  .npz -> draws "
        "subsampling (gold standard).  .json -> dispersed-Gaussian "
        "N(mean, (dispersion_scale*std)^2) from reference/summary.json. "
        "Default None uses overdispersed init_to_uniform.",
    )
    ap.add_argument(
        "--dispersion-scale",
        type=float,
        default=1.5,
        metavar="F",
        help="Multiplier on posterior std for dispersed-Gaussian mode. "
        "Default 1.5 (~2.25x posterior variance).  Ignored for .npz draws.",
    )
    args = ap.parse_args()

    if args.smoke:
        args.n_chains, args.n_draws, args.n_warmup = 4, 200, 200

    entry = MODELS[args.model]
    out = Path(args.out_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    key = jax.random.key(args.seed)

    is_nuts = entry.reference_method == ReferenceMethod.NUTS
    _ta = (
        args.target_accept
        if args.target_accept is not None
        else entry.reference_target_acceptance
    )
    if entry.requires_x64 and not jax.config.read("jax_enable_x64"):
        print(
            f"[FATAL] {args.model} requires_x64 but x64 not enabled "
            f"(set GT_X64=1).",
            flush=True,
        )
        sys.exit(2)

    if args.init_posterior is not None:
        path = Path(args.init_posterior)
        if path.suffix == ".npz":
            _init_mode = "draws"
            _init_note = (
                f"posterior-dispersed init: draws subsampled from {args.init_posterior} "
                "(certified reference draws, unconstrained space). "
                "Between-chain R-hat is un-hollowed: chains start at genuine "
                "posterior samples, maximally dispersed within the posterior."
            )
        else:
            _init_mode = f"dispersed_gaussian_scale{args.dispersion_scale}"
            _init_note = (
                f"posterior-dispersed init: N(mean, ({args.dispersion_scale}*std)^2) "
                f"per site from {args.init_posterior} (reference/summary.json). "
                "Dispersion at posterior scale (~2.25x posterior variance at scale=1.5): "
                "wide enough for un-hollowed between-chain R-hat, narrow enough to "
                "stay in posterior support and avoid ODE-unstable regions."
            )
        print(f"[init] {_init_mode}: {args.init_posterior}", flush=True)
    else:
        _init_mode = "init_to_uniform_radius2"
        _init_note = "overdispersed init_to_uniform radius2 (Stan/posteriordb default)"

    print(
        f"[start] {args.model} nc={args.n_chains} ns={args.n_draws} "
        f"nw={args.n_warmup} x64={jax.config.read('jax_enable_x64')} "
        f"device={jax.devices()[0].platform} method={'nuts' if is_nuts else 'analytic'}",
        flush=True,
    )

    t_all = time.perf_counter()
    if is_nuts:
        positions, diag, timing = run_nuts_multichain(
            entry,
            key,
            args.n_chains,
            args.n_warmup,
            args.n_draws,
            _ta,
            args.max_doublings,
            sequential=args.sequential,
            init_posterior=args.init_posterior,
            dispersion_scale=args.dispersion_scale,
        )
    else:
        positions, diag, timing = run_analytic_multichain(
            entry,
            key,
            args.n_chains,
            args.n_draws,
        )

    # --- PERSIST DRAWS FIRST (before post-processing) ---
    draws_path = out / "draws_10x10k.npz"
    np.savez_compressed(str(draws_path), **positions)
    print(
        f"[draws] saved {draws_path} "
        f"sites={ {s: tuple(a.shape) for s, a in positions.items()} }",  # noqa: E201,E202
        flush=True,
    )
    if is_nuts:
        print(
            f"[diag] total_div={diag['total_divergences']} "
            f"ebfmi={ [round(x, 3) for x in diag['e_bfmi_per_chain']] }",  # noqa: E201,E202
            flush=True,
        )

    # --- post-processing (arviz) ---
    per_site, max_rhat, min_bulk = summarize(positions)
    total_div = diag.get("total_divergences", 0)
    n_total = args.n_chains * args.n_draws
    min_ebfmi = min(diag["e_bfmi_per_chain"]) if is_nuts else None
    gate_pass = (max_rhat <= 1.01) and (
        not is_nuts
        or (
            min_bulk >= 400.0
            and (total_div / n_total) <= 0.001
            and (min_ebfmi is None or min_ebfmi >= 0.3)
        )
    )

    summary = {
        "model_name": args.model,
        "schema_version": SCHEMA_VERSION,
        "generator": "nuts_perchain" if is_nuts else "analytic_iid",
        "space": "unconstrained",
        "n_chains": args.n_chains,
        "n_draws_per_chain": args.n_draws,
        "n_total": n_total,
        "sampler_config": {
            "sampler": "nuts" if is_nuts else "analytic",
            "warmup": "window_adaptation_diag_imm_perchain" if is_nuts else None,
            "n_warmup_per_chain": args.n_warmup if is_nuts else None,
            "target_acceptance": _ta if is_nuts else None,
            "max_num_doublings": args.max_doublings if is_nuts else None,
            "init_strategy": _init_mode,
            "execution": (
                ("sequential" if args.sequential else "vmap") if is_nuts else "iid"
            ),
        },
        "seeds": {
            "master_seed": args.seed,
            "derivation": "key=jax.random.key(seed); split->(init,warmup,sample); "
            "per-chain=split(k,n_chains)",
        },
        "az_method": {
            "bulk_ess": "az.ess(idata, method='bulk') on raw (chain,draw) real chains",
            "tail_ess": "az.ess(idata, method='tail')",
            "rhat": "az.rhat(idata, method='rank')  # Vehtari 2021 rank-norm split-Rhat",
            "between_chain_se": "std(chain_means, ddof=1)/sqrt(n_chains)",
        },
        "quality_gate": {
            "rhat_threshold": 1.01,
            "max_rhat": max_rhat,
            "min_bulk_ess": min_bulk,
            "total_divergences": total_div,
            "divergence_rate": (total_div / n_total) if is_nuts else 0.0,
            "min_e_bfmi": min_ebfmi,
            "passed": bool(gate_pass),
        },
        "diagnostics_per_chain": diag,
        "timing_seconds": {**timing, "total": time.perf_counter() - t_all},
        "provenance": {
            "tuningfork_version": tuningfork.__version__,
            "code_sha": os.environ.get("GT_CODE_SHA", "unset"),
            "jax_version": jax.__version__,
            "blackjax_version": blackjax.__version__,
            "arviz_version": az.__version__,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "device": jax.devices()[0].platform,
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "init_note": _init_note,
        },
        "per_site": per_site,
    }
    summary_path = out / "summary_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[summary] {summary_path}", flush=True)
    print(
        f"[GATE] {'PASS' if gate_pass else 'FAIL'} max_rhat={max_rhat:.5f} "
        f"min_bulk_ess={min_bulk:.0f} total_div={total_div} "
        f"min_ebfmi={min_ebfmi}",
        flush=True,
    )
    print(f"[done] {args.model} wall={time.perf_counter() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
