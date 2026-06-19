"""#25 vmap-vs-loop diagnosis, PHASE 1 (banana).

(A) ROOT CAUSE: |Δld|, |Δgrad| unbatched vs vmap-batch (machine-eps?) + controlled
    leapfrog divergence curve max|Δ|_n vs n (exp(λn) amplification-from-eps test).
(B) KEY HANDLING: raw key_data loop vs stacked-vmap, assert bit-identical.
(C) STATISTICAL EQUIVALENCE: actual sampler avg=18, n=3000, 4ch x seeds 10-15, loop AND
    vmap; per-chain nsteps/acc/div + 2mbias/ESS/rhat; KS test on pooled marginals per dim.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/diag_vmap_banana_p1.py
"""

import json
import os
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)  # FLOAT64

import arviz as az
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.flatten_util import ravel_pytree
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

GIT_HEAD = (
    subprocess.check_output(
        ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
print("git_head:", GIT_HEAD, "| x64:", jax.config.jax_enable_x64)

BANANA_MEAN = np.array([0.0, 2.0])
BANANA_VAR = np.array([8.0, 9.0])
SEEDS = [10, 11, 12, 13, 14, 15]
CH = 4
N_EQUIV = 3000
AVG_EQUIV = 18


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), 2, jnp.asarray(BANANA_VAR)


def _chain_keys(seed, ch):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


ld, init, d, imm = load_banana()
gradfn = jax.grad(ld)

# ============================ PART A: ROOT CAUSE =============================
print(
    "\n" + "=" * 78 + "\nPART A: ROOT CAUSE (fp ordering under vmap-batch)\n" + "=" * 78
)
# (A1) ld and grad at a fixed x, unbatched vs vmap-batch-of-4 (identical copies), extract idx0
np.random.seed(0)
x_test = jnp.asarray(np.random.randn(d) * np.sqrt(BANANA_VAR))
ld_unb = ld(x_test)
g_unb = gradfn(x_test)
Xb = jnp.broadcast_to(x_test, (CH, d))
ld_b = jax.vmap(ld)(Xb)[0]
g_b = jax.vmap(gradfn)(Xb)[0]
dld = float(abs(ld_unb - ld_b))
dg = float(jnp.max(jnp.abs(g_unb - g_b)))
print(f"x_test = {np.asarray(x_test)}")
print(f"  |ld_unbatched - ld_vmap[0]|   = {dld:.3e}")
print(f"  max|grad_unb - grad_vmap[0]|  = {dg:.3e}  (machine eps fp64 ~2.2e-16)")

# (A2) controlled leapfrog trajectory: unbatched vs vmapped (identical init), divergence curve
eps = 0.21126  # cert median step
Minv = imm  # diagonal inverse-mass as the kernel uses
N_LF = 400


def leapfrog_traj(x0, p0):
    def stp(carry, _):
        x, p = carry
        p = p + 0.5 * eps * gradfn(x)
        x = x + eps * (Minv * p)
        p = p + 0.5 * eps * gradfn(x)
        return (x, p), x

    _, xs = jax.lax.scan(stp, (x0, p0), None, length=N_LF)
    return xs  # (N_LF, d)


x0 = x_test
p0 = jnp.asarray(np.random.randn(d))
traj_unb = np.asarray(leapfrog_traj(x0, p0))  # (N, d)
X0b = jnp.broadcast_to(x0, (CH, d))
P0b = jnp.broadcast_to(p0, (CH, d))
traj_b = np.asarray(jax.vmap(leapfrog_traj)(X0b, P0b))[0]  # (N, d)
delta = np.max(np.abs(traj_unb - traj_b), axis=1)  # (N,)
print(f"\n  controlled leapfrog (eps={eps}, N={N_LF}): max|Δ| per step")
report_steps = [1, 2, 5, 10, 20, 50, 100, 200, 399]
for n in report_steps:
    print(f"    step {n:>4d}: max|Δ| = {delta[n]:.3e}")
# log-slope fit on the growth region (until it saturates near O(sqrt(var)))
sat = np.sqrt(BANANA_VAR).max()
grow = np.where(delta < 0.1 * sat)[0]
if len(grow) > 5:
    ns = grow[grow >= 1]
    logd = np.log(np.maximum(delta[ns], 1e-300))
    A = np.vstack([ns, np.ones_like(ns)]).T
    lam, b = np.linalg.lstsq(A, logd, rcond=None)[0]
    print(
        f"  growth-region log-slope λ = {lam:.4f} /step (exp(λn) from eps); saturates ~{sat:.2f}"
    )
else:
    lam = None
    print("  (no clean growth region — see curve)")

# ============================ PART B: KEY HANDLING ==========================
print(
    "\n" + "=" * 78 + "\nPART B: KEY HANDLING (loop vs stacked-vmap keys)\n" + "=" * 78
)
key_mismatch = 0
for seed in SEEDS:
    loop_keys = _chain_keys(seed, CH)
    stacked = jnp.stack(loop_keys)
    loop_data = np.asarray(jnp.stack([jax.random.key_data(k) for k in loop_keys]))
    vmap_data = np.asarray(jax.random.key_data(stacked))
    same = bool(np.array_equal(loop_data, vmap_data))
    if not same:
        key_mismatch += 1
print(
    f"  loop key_data == stacked vmap key_data for all {len(SEEDS)} seeds x {CH} chains: "
    f"{'YES (bit-identical)' if key_mismatch == 0 else f'NO ({key_mismatch} mismatch)'}"
)

# ===================== PART C: STATISTICAL EQUIVALENCE ======================
print(
    "\n"
    + "=" * 78
    + f"\nPART C: STAT EQUIVALENCE (avg={AVG_EQUIV}, n={N_EQUIV}, {CH}ch, seeds {SEEDS})\n"
    + "=" * 78
)


def ref_step(seed, num_steps=2000):
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 1000 + 500))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=num_steps,
            state=st,
            rng_key=jax.random.key(seed * 1000 + 501),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def sample_loop(step, avg, n, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    pos, divs, accs, nss = [], [], [], []
    for sk in _chain_keys(seed, CH):
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
    return np.stack(pos, 0), np.array(divs), np.array(accs), np.array(nss)


def sample_vmap(step, avg, n, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = jnp.stack(_chain_keys(seed, CH))

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
    return np.asarray(flats), np.asarray(dvs), np.asarray(acs), np.asarray(nss)


def per_chain_stats(arr):
    # arr (ch, n, d)
    gm, gv = BANANA_MEAN, BANANA_VAR
    out = []
    for c in range(arr.shape[0]):
        a = arr[c]  # (n, d)
        vm = np.mean((a - gm[None, :]) ** 2, axis=0)
        b2 = float((np.abs(vm - gv) / gv).max())
        ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], a[None])})
        ess = float(np.array(az.ess(ds, method="bulk")["x"]).min())
        out.append((b2, ess))
    return out


rows = []
print(
    f"  {'seed':>4s} {'path':>4s} {'chain':>5s} {'nsteps':>8s} {'acc':>7s} {'div':>5s} {'2mbias':>7s} {'ESS':>7s}"
)
for seed in SEEDS:
    step = ref_step(seed)
    aL, dL, cL, nL = sample_loop(step, AVG_EQUIV, N_EQUIV, seed)
    aV, dV, cV, nV = sample_vmap(step, AVG_EQUIV, N_EQUIV, seed)
    sL = per_chain_stats(aL)
    sV = per_chain_stats(aV)
    # per-chain scalar diffs (acc/nsteps/div)
    dacc = float(np.max(np.abs(cL.mean(1) - cV.mean(1))))
    dnst = float(np.max(np.abs(nL.mean(1) - nV.mean(1))))
    ddiv = float(np.max(np.abs(dL.mean(1) - dV.mean(1))))
    # KS test on pooled marginals per dim
    ks = []
    for dim in range(d):
        s_loop = aL[:, :, dim].ravel()
        s_vmap = aV[:, :, dim].ravel()
        st_, p_ = ks_2samp(s_loop, s_vmap)
        ks.append((float(st_), float(p_)))
    for c in range(CH):
        print(
            f"  {seed:>4d} loop {c:>5d} {nL[c].mean():>8.4f} {cL[c].mean():>7.4f} {dL[c].mean():>5.2f} {sL[c][0]:>7.3f} {sL[c][1]:>7.0f}"
        )
        print(
            f"  {seed:>4d} vmap {c:>5d} {nV[c].mean():>8.4f} {cV[c].mean():>7.4f} {dV[c].mean():>5.2f} {sV[c][0]:>7.3f} {sV[c][1]:>7.0f}"
        )
    print(
        f"   seed {seed}: max|Δacc|={dacc:.2e} max|Δnsteps|={dnst:.2e} max|Δdiv|={ddiv:.2e} | "
        f"KS x0 (D={ks[0][0]:.4f},p={ks[0][1]:.3f}) x1 (D={ks[1][0]:.4f},p={ks[1][1]:.3f})"
    )
    rows.append(
        {
            "seed": seed,
            "step": step,
            "max_dacc": dacc,
            "max_dnsteps": dnst,
            "max_ddiv": ddiv,
            "ks_x0": ks[0],
            "ks_x1": ks[1],
            "loop_2mbias": [s[0] for s in sL],
            "vmap_2mbias": [s[0] for s in sV],
            "loop_ess": [s[1] for s in sL],
            "vmap_ess": [s[1] for s in sV],
        }
    )
    sys.stdout.flush()

# verdict on KS: all p-values >> 0.05 => same distribution
all_p = [r["ks_x0"][1] for r in rows] + [r["ks_x1"][1] for r in rows]
ks_ok = all(p > 0.01 for p in all_p)
print(
    f"\n  KS verdict: min p-value across all seeds/dims = {min(all_p):.4f} -> "
    f"{'SAME DISTRIBUTION (no rejection at 1%)' if ks_ok else 'REJECTED somewhere'}"
)

out = os.path.join(_HERE, "diag_vmap_banana_p1_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "partA": {
                "dld": dld,
                "dgrad": dg,
                "leapfrog_delta_at": {str(n): float(delta[n]) for n in report_steps},
                "lyapunov_logslope": (float(lam) if lam is not None else None),
                "saturation": float(sat),
            },
            "partB_key_mismatch": key_mismatch,
            "partC": rows,
            "ks_min_p": float(min(all_p)),
            "ks_ok": ks_ok,
        },
        f,
        indent=2,
    )
print(f"\nwrote {out}")
print("DONE_DIAG_P1")
