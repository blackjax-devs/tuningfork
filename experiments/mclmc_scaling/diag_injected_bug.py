"""Injected-bug validation for the two-part parity gate (test #3 of the patch spec).

Confirms the STRUCTURAL micro-parity check (A) discriminates a real key-handling bug
(O(1) at sample 1 -> BLOCK_BUG) from benign fp-chaos (~machine-eps at sample 1 -> OK).
Reference numbers for SWE to match once the gate is wired into the harness.

Cases (banana adjusted_mclmc_dynamic, float64, chain 0, seed 101):
  CORRECT     : vmap fed the SAME per-chain keys as the loop  -> sample-1 |d| ~ 1e-15  -> VMAP_OK
  BUG_SHIFT   : vmap chain-0 slot fed chain-1's key (off-by-one) -> sample-1 |d| ~ O(1) -> BLOCK_BUG
  BUG_TRANSPOSE: vmap keys reversed                              -> sample-1 |d| ~ O(1) -> BLOCK_BUG

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/diag_injected_bug.py
"""

import json
import os
import sys
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from run_fixed_imm import _make_fixed_imm_adj_dyn_kernel, _make_fixed_imm_kernel

BANANA_VAR = np.array([8.0, 9.0])
CH, AVG, N = 4, 18, 50
STRUCT_TOL = 1e-10


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), jnp.asarray(BANANA_VAR)


def ref_step(ld, init, imm, seed=101):
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 7 + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=2000,
            state=st,
            rng_key=jax.random.key(seed * 7 + 2),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


def _scan_chain(ld, init, imm, dyn, step, sk):
    s = adj_dyn_mod.init(init, ld, sk)

    def stp(c, key):
        nx, info = dyn(
            rng_key=key,
            state=c,
            logdensity_fn=ld,
            step_size=step,
            L_proposal_factor=jnp.inf,
            inverse_mass_matrix=imm,
            integration_steps_params=(float(AVG),),
        )
        return nx, nx.position

    _, pt = jax.lax.scan(stp, s, jax.random.split(sk, N))
    return jax.vmap(lambda q: ravel_pytree(q)[0])(pt)


def chain0_loop(ld, init, imm, step, seed):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    k0 = jax.random.key(seed * 1000 + 0 + 1)  # chain 0's correct key
    return np.asarray(_scan_chain(ld, init, imm, dyn, step, k0))


def chain0_vmap(ld, init, imm, step, seed, mode):
    dyn = _make_fixed_imm_adj_dyn_kernel(imm)
    keys = [jax.random.key(seed * 1000 + ci + 1) for ci in range(CH)]
    if mode == "correct":
        pass
    elif mode == "shift":  # off-by-one: chain-0 slot gets chain-1's key
        keys = keys[1:] + keys[:1]
    elif mode == "transpose":  # reversed order
        keys = keys[::-1]
    stacked = jnp.stack(keys)
    flats = jax.vmap(lambda sk: _scan_chain(ld, init, imm, dyn, step, sk))(stacked)
    return np.asarray(flats)[0]  # chain-0 slot


ld, init, imm = load_banana()
step = ref_step(ld, init, imm, 101)
loop0 = chain0_loop(ld, init, imm, step, 101)
print(f"banana ref_step={step:.5f}  AVG={AVG}  struct_tol={STRUCT_TOL}\n")
print(f"{'case':>12} {'sample1_|d|':>14} {'verdict':>12}")
out = {}
for mode in ["correct", "shift", "transpose"]:
    v0 = chain0_vmap(ld, init, imm, step, 101, mode)
    s1 = float(np.max(np.abs(loop0[0] - v0[0])))
    verdict = "VMAP_OK" if s1 < STRUCT_TOL else "BLOCK_BUG"
    print(f"{mode:>12} {s1:>14.3e} {verdict:>12}")
    out[mode] = {"sample1_abs": s1, "verdict": verdict}

ok = (
    out["correct"]["verdict"] == "VMAP_OK"
    and out["shift"]["verdict"] == "BLOCK_BUG"
    and out["transpose"]["verdict"] == "BLOCK_BUG"
)
print(f"\nGATE DISCRIMINATES BUG vs fp-CHAOS: {ok}")
with open(os.path.join(_HERE, "diag_injected_bug_results.json"), "w") as f:
    json.dump(
        {"step": step, "struct_tol": STRUCT_TOL, "cases": out, "discriminates": ok},
        f,
        indent=2,
    )
print("DONE_INJECTED_BUG")
