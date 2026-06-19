"""#25 cert: banana adjusted_mclmc_dynamic at the grid-search PASS window (avg~18).

Production-budget re-cert to confirm the n=500 dynamic-L PASS cell (avg=18, 2mbias 0.079,
rhat 1.003, ESS 726) is a GENUINE clean PASS, not max-over-D 2mbias noise -- and to pick
the most seed-robust filing point in the 18-54 window.

Config (user-approved via tl):
  kernel  : adjusted_mclmc_dynamic, avg PINNED constant (integration_steps_params=(avg,))
  avg     : {18, 24, 36, 54}
  budget  : n=5000/chain, 4 chains, n_warmup=5000
  seeds   : 6 fresh {101..106} (distinct from the sweep's 0,1,2)
  IMM     : banana analytic-GT branch -- diag marginal var [8,9]; golden mean [0,2]
  step    : EEVPD warmup, per-seed (file the median); L = avg * step
  path    : VMAP (parity-gated against the sequential loop on banana avg=18)

GATE (PASS @ medium__): 2mbias<0.1 AND mbias_sd<0.06 AND rhat<1.01 AND minESS>100/ch
                        AND div=0, holding in >=5/6 seeds.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/cert_banana_medium18.py
"""

import json
import os
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.flatten_util import ravel_pytree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

EXPECT_HEAD = "8937e088"
try:
    GIT_HEAD = (
        subprocess.check_output(
            ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
        )
        .decode()
        .strip()
    )
except Exception as e:  # pragma: no cover
    GIT_HEAD = f"UNKNOWN({e})"
print(
    ("git_head OK: " if GIT_HEAD.startswith(EXPECT_HEAD) else "!! WARNING off-pin: ")
    + GIT_HEAD
)

AVG_WINDOW = [18, 24, 36, 54]
SEEDS = [101, 102, 103, 104, 105, 106]
N_SAMPLES = 5000
N_WARMUP = 5000
CH = 4

# medium__ PASS gate
TAU_2M, TAU_MBSD, RHAT_BAD, ESS_MIN, SEED_QUORUM = 0.10, 0.06, 1.01, 100, 5

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    d = 2
    imm = jnp.asarray(BANANA_VAR)  # exact diagonal marginal cov (Cov(x1,x2)=0)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(d), d, imm


def ref_step(ld, init, imm, seed):
    """EEVPD step from warmup. RNG keys derived from seed so each cert seed gets its
    own tuned step -> we report the spread and file the median."""
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 7 + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=N_WARMUP,
            state=st,
            rng_key=jax.random.key(seed * 7 + 2),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


def sample_loop(ld, init, imm, step, avg, n, ch, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for sk in _chain_keys(seed, ch):
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (
                nx.position,
                info.is_divergent,
                info.acceptance_rate,
                info.num_integration_steps,
            )

        _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        pos.append(np.array(jax.vmap(lambda q: ravel_pytree(q)[0])(pt)))
        divs.append(np.array(dv))
        accs.append(np.array(ac))
        nss.append(np.array(ns))
    return (
        np.stack(pos, 0),
        float(np.mean(np.concatenate(divs))),
        float(np.mean(np.concatenate(accs))),
        float(np.mean(np.concatenate(nss))),
    )


def sample_vmap(ld, init, imm, step, avg, n, ch, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = jnp.stack(_chain_keys(seed, ch))

    def run_chain(sk):
        s = adj_dyn_mod.init(init, ld, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,
                integration_steps_params=(float(avg),),
            )
            return nx, (
                nx.position,
                info.is_divergent,
                info.acceptance_rate,
                info.num_integration_steps,
            )

        _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
        flat = jax.vmap(lambda q: ravel_pytree(q)[0])(pt)
        return flat, dv, ac, ns

    flats, dvs, acs, nss = jax.vmap(run_chain)(keys)
    arr = np.asarray(flats)
    return (
        arr,
        float(np.mean(np.asarray(dvs))),
        float(np.mean(np.asarray(acs))),
        float(np.mean(np.asarray(nss))),
    )


def quality(arr, gm, gv):
    vm = np.mean((arr - gm[None, None, :]) ** 2, axis=(0, 1))
    max_2mom_bias = float((np.abs(vm - gv) / np.maximum(gv, 1e-30)).max())
    mean_est = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    max_mean_bias_sd = float(
        (np.abs(mean_est - gm) / np.maximum(np.sqrt(gv), 1e-30)).max()
    )
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], arr)})
    min_ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
    max_rhat = float(np.array(az.rhat(ds, method="rank")["x"]).max())
    return max_2mom_bias, min_ess, max_rhat, max_mean_bias_sd


def seed_pass(b2, mbsd, ess, rhat, div):
    return (
        b2 < TAU_2M
        and mbsd < TAU_MBSD
        and rhat < RHAT_BAD
        and ess > ESS_MIN * CH
        and div == 0.0
    )


# ---- STATISTICAL-EQUIVALENCE GATE (proposed standing gate; vmap FORCED) ----
# Replaces the brittle bit-exact-positions short-circuit. Two parts:
#   (A) STRUCTURAL micro-parity at sample 1 (catches real bugs): identical RNG + kernel
#       => the FIRST kernel step must agree to ~machine-eps. A genuine code/key/broadcast
#       bug shows O(1) here; benign fp-chaos shows ~1e-15.
#   (B) STATISTICAL equivalence over a probe run: KS on marginals + acc/nsteps/div tolerance.
# vmap is FORCED regardless (USE_VMAP=True): the diagnosis (diag_vmap_parity) proved the
# numbers are path-independent; this run produces clean vmap-path provenance.
from scipy.stats import ks_2samp as _ks

print("STAT-EQUIV GATE: vmap vs loop (banana, avg=18, probe n=400, 4 chains)...")
_ld, _init, _d, _imm = load_banana()
_rs0 = ref_step(_ld, _init, _imm, SEEDS[0])
_a_loop, _dl, _al, _nl = sample_loop(_ld, _init, _imm, _rs0, 18, 400, CH, SEEDS[0])
_a_vmap, _dv, _av, _nv = sample_vmap(_ld, _init, _imm, _rs0, 18, 400, CH, SEEDS[0])
# (A) structural micro-parity at sample 1 (chain 0)
_step1 = float(np.max(np.abs(_a_loop[0, 0, :] - _a_vmap[0, 0, :])))
_struct_ok = _step1 < 1e-10
# (B) statistical equivalence over the probe
_poolL, _poolV = _a_loop.reshape(-1, 2), _a_vmap.reshape(-1, 2)
_ks1, _ks2 = _ks(_poolL[:, 0], _poolV[:, 0]), _ks(_poolL[:, 1], _poolV[:, 1])
_ks_D = max(float(_ks1.statistic), float(_ks2.statistic))
_ks_p = min(float(_ks1.pvalue), float(_ks2.pvalue))
_acc_d, _ns_d, _div_d = abs(_al - _av), abs(_nl - _nv), abs(_dl - _dv)
_stat_ok = (
    (_ks_p > 0.05 or _ks_D < 0.05) and _acc_d < 5e-3 and _ns_d < 1e-3 and _div_d < 1e-3
)
_max_abs = float(
    np.max(np.abs(_a_loop - _a_vmap))
)  # full-traj (expected large: fp-chaos)
print(
    f"  (A) struct micro-parity sample-1 |d| = {_step1:.3e}  -> {'OK' if _struct_ok else 'BUG!'} (<1e-10)"
)
print(
    f"  (B) KS D={_ks_D:.4f} p={_ks_p:.3f} | acc|d|={_acc_d:.2e} nsteps|d|={_ns_d:.2e} div|d|={_div_d:.2e} -> {'OK' if _stat_ok else 'FAIL'}"
)
print(
    f"      (full-traj max|loop-vmap|={_max_abs:.3e} — expected large under fp-chaos; NOT gated)"
)
GATE_PASS = _struct_ok and _stat_ok
USE_VMAP = True  # FORCED — diagnosis proved path-independence
sample = sample_vmap
if not GATE_PASS:
    print(
        "  !! STAT-EQUIV GATE FAILED — investigate (NOT a loop-fallback situation). Continuing on vmap for provenance."
    )
else:
    print(
        "  STAT-EQUIV GATE OK -> vmap is the canonical multichain path (no loop fallback)"
    )
sys.stdout.flush()

print(
    f"\ncert_banana_medium18 | path={'VMAP' if USE_VMAP else 'LOOP'} | n={N_SAMPLES}/chain x {CH}ch "
    f"x seeds {SEEDS} | window={AVG_WINDOW} | n_warmup={N_WARMUP}"
)
print(
    f"GATE @medium__: 2mbias<{TAU_2M} & mbias_sd<{TAU_MBSD} & rhat<{RHAT_BAD} & minESS>{ESS_MIN}/ch "
    f"& div=0, in >={SEED_QUORUM}/{len(SEEDS)} seeds\n"
)

# per-seed EEVPD steps (file the median)
STEPS = {s: ref_step(_ld, _init, _imm, s) for s in SEEDS}
step_vals = np.array([STEPS[s] for s in SEEDS])
STEP_MEDIAN = float(np.median(step_vals))
print(f"per-seed EEVPD step: " + ", ".join(f"{s}:{STEPS[s]:.4f}" for s in SEEDS))
print(
    f"step median={STEP_MEDIAN:.5f}  (min={step_vals.min():.5f} max={step_vals.max():.5f})\n"
)
sys.stdout.flush()

cells = []
summary = {}
print(
    f"  {'avg':>4s} {'L_med':>6s} {'seed':>5s} {'realAvg':>7s} {'2mbias':>7s} {'mbias_sd':>8s} "
    f"{'minESS':>7s} {'Rhat':>7s} {'div':>6s} {'acc':>6s}  {'seedPASS':>8s}"
)
for avg in AVG_WINDOW:
    L_med = avg * STEP_MEDIAN
    npass = 0
    b2s, mbsds, esss, rhats, divs, accs, ravgs = [], [], [], [], [], [], []
    for seed in SEEDS:
        # each seed uses its own EEVPD-tuned step (cert-honest); golden is filed at median
        arr, dr, ac, mns = sample(
            _ld, _init, _imm, STEPS[seed], avg, N_SAMPLES, CH, seed
        )
        b2, ess, rhat, mbsd = quality(arr, BANANA_MEAN, BANANA_VAR)
        ok = seed_pass(b2, mbsd, ess, rhat, dr)
        npass += int(ok)
        b2s.append(b2)
        mbsds.append(mbsd)
        esss.append(ess)
        rhats.append(rhat)
        divs.append(dr)
        accs.append(ac)
        ravgs.append(mns)
        cells.append(
            {
                "model": "banana",
                "avg_param": avg,
                "seed": int(seed),
                "d": 2,
                "step_size": STEPS[seed],
                "L": float(avg * STEPS[seed]),
                "realized_avg": float(mns),
                "n_samples": N_SAMPLES,
                "n_warmup": N_WARMUP,
                "max_2mom_bias": float(b2),
                "max_mean_bias_sd": float(mbsd),
                "min_bulk_ess": float(ess),
                "max_rhat": float(rhat),
                "div_rate": float(dr),
                "mean_acc": float(ac),
                "seed_pass": bool(ok),
                "git_head": GIT_HEAD,
            }
        )
        print(
            f"  {avg:>4d} {L_med:>6.2f} {seed:>5d} {mns:>7.1f} {b2:>7.3f} {mbsd:>8.3f} "
            f"{ess:>7.0f} {rhat:>7.3f} {dr:>6.3f} {ac:>6.3f}  {('PASS' if ok else 'fail'):>8s}"
        )
        sys.stdout.flush()
    verdict = "PASS" if npass >= SEED_QUORUM else "FAIL"
    summary[avg] = {
        "L_median": float(L_med),
        "n_pass": npass,
        "n_seeds": len(SEEDS),
        "verdict": verdict,
        "med_2mbias": float(np.median(b2s)),
        "max_2mbias": float(np.max(b2s)),
        "med_mbias_sd": float(np.median(mbsds)),
        "max_mbias_sd": float(np.max(mbsds)),
        "med_minESS": float(np.median(esss)),
        "min_minESS": float(np.min(esss)),
        "max_rhat": float(np.max(rhats)),
        "mean_acc": float(np.mean(accs)),
        "max_div": float(np.max(divs)),
        "med_realAvg": float(np.median(ravgs)),
    }
    print(
        f"  -> avg={avg}: {npass}/{len(SEEDS)} seeds PASS => {verdict} | "
        f"med2mbias={np.median(b2s):.3f} maxMbSd={np.max(mbsds):.3f} "
        f"medESS={np.median(esss):.0f} maxRhat={np.max(rhats):.3f}\n"
    )
    sys.stdout.flush()

# recommended filing point: PASS verdict, then lowest avg with the cleanest/most seed-robust profile
passing = [a for a in AVG_WINDOW if summary[a]["verdict"] == "PASS"]
rec = None
if passing:
    # prefer max seed-quorum, then lowest max_2mbias, then lowest avg (efficiency)
    rec = sorted(
        passing, key=lambda a: (-summary[a]["n_pass"], summary[a]["max_2mbias"], a)
    )[0]

out = os.path.join(_HERE, "cert_banana_medium18_vmap_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "path": "VMAP_FORCED",
            "avg_window": AVG_WINDOW,
            "n_samples": N_SAMPLES,
            "n_warmup": N_WARMUP,
            "seeds": SEEDS,
            "chains": CH,
            "stat_equiv_gate": {
                "struct_sample1_abs": _step1,
                "struct_ok": _struct_ok,
                "ks_D": _ks_D,
                "ks_p": _ks_p,
                "acc_abs": _acc_d,
                "nsteps_abs": _ns_d,
                "div_abs": _div_d,
                "stat_ok": _stat_ok,
                "gate_pass": GATE_PASS,
                "full_traj_max_abs": _max_abs,
            },
            "gate": {
                "tau_2mbias": TAU_2M,
                "tau_mbias_sd": TAU_MBSD,
                "rhat": RHAT_BAD,
                "ess_min_per_chain": ESS_MIN,
                "seed_quorum": SEED_QUORUM,
            },
            "step_per_seed": STEPS,
            "step_median": STEP_MEDIAN,
            "imm_diag": BANANA_VAR.tolist(),
            "golden_mean": BANANA_MEAN.tolist(),
            "golden_var": BANANA_VAR.tolist(),
            "summary": summary,
            "recommended_avg": rec,
            "cells": cells,
        },
        f,
        indent=2,
    )
print(
    f"wrote {out} ({len(cells)} rows) | path=VMAP_FORCED | stat_equiv_gate_pass={GATE_PASS}"
)
print(f"RECOMMENDED_FILING_AVG={rec}")
print("DONE_CERT_BANANA18_VMAP")
