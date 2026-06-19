"""Injected-bug test for the two-part vmap parity gate (banana).

Asserts: clean vmap path -> VMAP_OK; each injected bug -> BLOCK_BUG.
Bugs injected into the VMAP path only: step×1.05, IMM×1.5, wrong key (seed+1),
nsteps off (avg+1), gradient sign flip.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/inject_bug_gate_check.py
"""

import os
import subprocess
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
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
print("git_head:", GIT_HEAD)

BANANA_VAR = np.array([8.0, 9.0])
CH, N_REF, AVG, SEED = 4, 2000, 18, 10
K_SE, MICRO_TOL, KS_P, KS_D = 4.0, 1e-10, 0.05, 0.05


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), 2, jnp.asarray(BANANA_VAR)


ld, init, d, imm = load_banana()


def _keys(seed):
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(CH)]


def ref_step():
    st = mclmc_mod.init(init, ld, jax.random.key(SEED * 1000 + 500))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=2000,
            state=st,
            rng_key=jax.random.key(SEED * 1000 + 501),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


STEP = ref_step()


def run(path_imm, step, avg, ld_fn, keys, n, vmap):
    dyn = _make_fixed_imm_adj_dyn_kernel(path_imm)

    def run_chain(sk):
        s = adj_dyn_mod.init(init, ld_fn, sk)

        def stp(c, key):
            nx, info = dyn(
                rng_key=key,
                state=c,
                logdensity_fn=ld_fn,
                step_size=step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=path_imm,
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

    if vmap:
        f, dv, ac, ns = jax.vmap(run_chain)(jnp.stack(keys))
        return np.asarray(f), np.asarray(dv), np.asarray(ac), np.asarray(ns)
    out = [run_chain(k) for k in keys]
    return (
        np.stack([np.asarray(o[0]) for o in out]),
        np.stack([np.asarray(o[1]) for o in out]),
        np.stack([np.asarray(o[2]) for o in out]),
        np.stack([np.asarray(o[3]) for o in out]),
    )


def gate(vmap_step=None, vmap_imm=None, vmap_avg=None, vmap_ld=None, vmap_keyseed=None):
    """Loop is the clean reference; vmap_* inject a bug into the vmap path only."""
    loop_keys = _keys(SEED)
    vkeys = _keys(vmap_keyseed if vmap_keyseed is not None else SEED)
    vstep = vmap_step if vmap_step is not None else STEP
    vimm = vmap_imm if vmap_imm is not None else imm
    vavg = vmap_avg if vmap_avg is not None else AVG
    vld = vmap_ld if vmap_ld is not None else ld
    # (A.1) key identity
    kd_loop = np.asarray(jnp.stack([jax.random.key_data(k) for k in loop_keys]))
    kd_vmap = np.asarray(jax.random.key_data(jnp.stack(vkeys)))
    if not np.array_equal(kd_loop, kd_vmap):
        return "BLOCK_BUG", "A.key"
    # (A.2) micro-parity n=1
    a1L, *_ = run(imm, STEP, AVG, ld, loop_keys, 1, False)
    a1V, *_ = run(vimm, vstep, vavg, vld, vkeys, 1, True)
    if float(np.max(np.abs(a1L - a1V))) >= MICRO_TOL:
        return "BLOCK_BUG", "A.micro"
    # reference cell
    aL, dL, cL, nL = run(imm, STEP, AVG, ld, loop_keys, N_REF, False)
    aV, dV, cV, nV = run(vimm, vstep, vavg, vld, vkeys, N_REF, True)
    if float(abs(nL.mean() - nV.mean())) != 0.0:
        return "BLOCK_BUG", "B.nsteps"
    if float(abs(dL.mean() - dV.mean())) != 0.0:
        return "BLOCK_BUG", "B.div"
    acc = 0.5 * (cL.mean() + cV.mean())
    se = float(np.sqrt(acc * (1 - acc) / N_REF))
    if float(abs(cL.mean() - cV.mean())) >= K_SE * se:
        return "BLOCK_BUG", "B.acc"
    for j in range(d):
        D, p = ks_2samp(aL[:, :, j].ravel(), aV[:, :, j].ravel())
        if not (p > KS_P or D < KS_D):
            return "BLOCK_BUG", f"B.KS_x{j}"
    return "VMAP_OK", "-"


def bad_ld(xf):
    g = ld(xf)
    return g  # placeholder; sign-flip injected via grad path below


# gradient sign flip: wrap logdensity so its gradient flips sign on dim 0
def ld_gradflip(xf):
    # add a term whose gradient cancels then doubles-negative dim-0 slope -> different traj
    return ld(xf) - 2.0 * (xf[0] ** 2) * 0.0 + 0.0  # no-op; real flip below


tests = [
    ("CLEAN (no bug)", dict(), "VMAP_OK"),
    ("step x1.05", dict(vmap_step=STEP * 1.05), "BLOCK_BUG"),
    ("IMM x1.5", dict(vmap_imm=imm * 1.5), "BLOCK_BUG"),
    ("wrong key (seed+1)", dict(vmap_keyseed=SEED + 1), "BLOCK_BUG"),
    ("nsteps off (avg+1)", dict(vmap_avg=AVG + 1), "BLOCK_BUG"),
    ("gradient sign flip (ld -> -ld)", dict(vmap_ld=(lambda xf: -ld(xf))), "BLOCK_BUG"),
]

print(
    f"\ngate config: n_ref={N_REF}, avg={AVG}, K_SE={K_SE}, micro_tol={MICRO_TOL}, step={STEP:.4f}\n"
)
print(
    f"{'injected':32s} {'expect':>10s} {'got':>10s} {'caught_by':>10s}  {'result':>5s}"
)
ok = True
for name, kw, expect in tests:
    verdict, where = gate(**kw)
    passed = verdict == expect
    ok = ok and passed
    print(
        f"{name:32s} {expect:>10s} {verdict:>10s} {where:>10s}  {'PASS' if passed else 'XXXX'}"
    )

print("\n" + ("ALL_GATE_TESTS_PASS" if ok else "GATE_TEST_FAILURE"))
print("DONE_INJECT_BUG")
