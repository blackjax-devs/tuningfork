"""Instrument warmup_grad_evals for the avg=24 banana medium__ recipe.

tl authorized (option b): run the exact warmup used by the cert (mclmc_find_L_and_step_size,
n_warmup=5000, default fracs 0.1/0.1/0.1, diagonal_preconditioning=False, banana, fixed-IMM
unadjusted mclmc kernel) and report the EXACT integration-step count it consumes — the tuner
returns it as its 3rd value (total_num_tuning_integrator_steps), which the cert discarded.

warmup_grad_evals = 2 * total_num_tuning_integrator_steps  (same "2x integration steps"
convention as the sampling headline; mclachlan = 2 grad-evals/step amortized).

Step count is deterministic in (num_steps, fracs), seed-INDEPENDENT — we run several seeds
to confirm that empirically rather than asserting it.

Run: JAX_PLATFORM_NAME=cpu .venv/bin/python experiments/mclmc_scaling/warmup_gradevals_probe.py
"""

import json
import os
import subprocess
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

import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from run_fixed_imm import _make_fixed_imm_kernel

EXPECT_HEAD = "8937e088"
GIT_HEAD = (
    subprocess.check_output(
        ["git", "-C", "/home/jp/blackjax-devs/blackjax", "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
print(
    ("git_head OK: " if GIT_HEAD.startswith(EXPECT_HEAD) else "!! off-pin: ") + GIT_HEAD
)

N_WARMUP = 5000
SEEDS = [10, 11, 12, 13, 14, 15]
GRAD_PER_STEP = (
    2  # mclachlan minimal-norm, amortized; matches sampling headline convention
)

BANANA_VAR = np.array([8.0, 9.0])


def load_banana():
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS as _M

    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), _M["banana"])
    _, unravel = ravel_pytree(init_dict)
    return (lambda xf: ld_raw(unravel(xf))), jnp.zeros(2), 2, jnp.asarray(BANANA_VAR)


ld, init, d, imm = load_banana()


def warmup_steps(seed, num_steps):
    """Return (step_size, total_num_tuning_integrator_steps) for one seed, exactly as the cert."""
    st = mclmc_mod.init(init, ld, jax.random.key(seed * 1000 + 500))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, n_tune = mclmc_find_L_and_step_size(
            mclmc_kernel=_make_fixed_imm_kernel(imm),
            num_steps=num_steps,
            state=st,
            rng_key=jax.random.key(seed * 1000 + 501),
            logdensity_fn=ld,
            diagonal_preconditioning=False,
        )
    return float(p.step_size), int(n_tune)


print(f"\n  {'seed':>5s} {'step':>9s} {'tune_steps':>11s} {'grad_evals':>11s}")
rows = []
for seed in SEEDS:
    step, n_tune = warmup_steps(seed, N_WARMUP)
    ge = GRAD_PER_STEP * n_tune
    rows.append(
        {
            "seed": seed,
            "step_size": step,
            "tune_integration_steps": n_tune,
            "warmup_grad_evals": ge,
        }
    )
    print(f"  {seed:>5d} {step:>9.5f} {n_tune:>11d} {ge:>11d}")
    sys.stdout.flush()

tune_counts = sorted({r["tune_integration_steps"] for r in rows})
ge_counts = sorted({r["warmup_grad_evals"] for r in rows})
seed_independent = len(tune_counts) == 1
print(
    f"\n  distinct tune_integration_steps across seeds: {tune_counts} "
    f"-> {'SEED-INDEPENDENT (deterministic)' if seed_independent else 'VARIES BY SEED'}"
)
print(f"  warmup_grad_evals = 2 * tune_steps = {ge_counts}")
print(
    f"  EXCLUDED from headline_metric (re-runner pays no warmup); provenance field only."
)

out = os.path.join(_HERE, "warmup_gradevals_probe_results.json")
with open(out, "w") as f:
    json.dump(
        {
            "git_head": GIT_HEAD,
            "n_warmup": N_WARMUP,
            "frac_tune": [0.1, 0.1, 0.1],
            "diagonal_preconditioning": False,
            "grad_per_step": GRAD_PER_STEP,
            "model": "banana",
            "seeds": SEEDS,
            "tune_integration_steps": tune_counts,
            "warmup_grad_evals": ge_counts,
            "seed_independent": seed_independent,
            "rows": rows,
        },
        f,
        indent=2,
    )
print(f"\nwrote {out}")
print("DONE_WARMUP_PROBE")
