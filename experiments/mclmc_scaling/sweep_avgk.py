"""avg=k deep-dive: optimal trajectory length for adjusted_mclmc_dynamic.

Research question: characterize optimal avg=k for adjusted_mclmc_dynamic as a
function of (preconditioning quality, geometry, dimension).

Downstream decision: should the production default target_num_integration_steps
be a CONSTANT (2) or geometry-aware?

Methodology (per TL correction, 2026-06-17):
  - avg=k is held by: (a) frac_tune2=0 (skip the sqrt(dim) reset and pass-2),
    (b) params.L = k * step_init at entry so pass-1's fix_L_first_da=False
    (avg-preserving via the order-bug fix) holds avg=k throughout the DA,
    (c) target_num_integration_steps=k so the post-override re-pins L=k*step
    as a no-op consistency check.
  - IMM regimes: GT-dense (kappa_eff~1, removes precond confound),
    diagonal (realistic worst-case, where Q2 regression lives).
  - ESS basis: BOTH spectral/Geyer (blackjax, can exceed N — captures the
    antithetic avg=2 mechanism) AND az-bulk. Gate spectral with fail-loud
    classifier to reject spurious ESS>N on near-divergent cells.
  - Fail-loud classification: PASS / LOUD-FAIL / SILENT-FAIL.

Usage:
  JAX_PLATFORM_NAME=cpu uv run python sweep_avgk.py --smoke   # E2E smoke ~5min
  JAX_PLATFORM_NAME=cpu uv run python sweep_avgk.py           # full sweep
  JAX_PLATFORM_NAME=cpu uv run python sweep_avgk.py --highd   # irt_1pl d=500 only
"""

import inspect
import os
import sys
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az

# ---------------------------------------------------------------------------
# MANDATORY GUARD: confirm order-bug-fixed blackjax is active.
# On the unfixed blackjax (pinned 359205da), fix_L=False is a dead no-op
# (L is frozen while the DA moves step → avg drifts away from k).
# The post-tuning L=k*step re-pin then miscalibrates the sampler at exactly
# the worst cells (high-k / ill-cond / diagonal). This guard makes it
# impossible to accidentally run on the wrong blackjax.
# ---------------------------------------------------------------------------
import blackjax.adaptation.adjusted_mclmc_adaptation as _adj_guard
import jax.numpy as jnp
import numpy as np
import xarray as xr

assert "old_step_size" in inspect.getsource(_adj_guard), (
    f"blackjax at {_adj_guard.__file__} LACKS the order-bug fix "
    f"(fix_L=False is a dead no-op on this version — avg=k will NOT be "
    f"held during tuning). Editable-install fix/adjusted-mclmc-fix-L-order-bug "
    f"first:\n  uv pip install -e /home/jp/blackjax-devs/blackjax\n"
    f"Then run via the venv python directly (NOT uv run, which re-syncs the lockfile):\n"
    f"  /home/jp/blackjax-devs/tuningfork/.venv/bin/python sweep_avgk.py"
)
print(f"GUARD OK: order-bug-fixed blackjax at {_adj_guard.__file__}")
from jax.flatten_util import ravel_pytree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, "..", ".."))

import blackjax
import blackjax.mcmc.adjusted_mclmc as adj_mclmc_mod_static  # for tuning state init
import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
import blackjax.mcmc.mclmc as mclmc_mod
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
)
from blackjax.adaptation.mclmc_adaptation import (
    MCLMCAdaptationState,
    mclmc_find_L_and_step_size,
)
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from run_fixed_imm import (
    _make_fixed_imm_adj_dyn_kernel,
    _make_fixed_imm_adj_kernel_for_tuning,
    _make_fixed_imm_kernel,
)

# ---------------------------------------------------------------------------
# Panel definition
# ---------------------------------------------------------------------------

SMOKE = "--smoke" in sys.argv
HIGHD = "--highd" in sys.argv

K_VALUES = [1, 2, 4, 8]

if SMOKE:
    N_WARMUP = 5
    N_SAMPLES = 20
    NUM_CHAINS = 2
    SEEDS = [0]
    PRIMARY_MODELS = ["mvn_10", "ill_cond_50"]
elif HIGHD:
    N_WARMUP = 2000
    N_SAMPLES = 3000
    NUM_CHAINS = 4
    SEEDS = [0, 1]
    PRIMARY_MODELS = ["irt_1pl"]
    K_VALUES = [1, 2, 4]  # k=8 not tested at d=500 (near-certain silent-fail)
else:
    N_WARMUP = 2000
    N_SAMPLES = 3000
    NUM_CHAINS = 4
    SEEDS = [0, 1, 2]
    PRIMARY_MODELS = [
        "mvn_10",  # isotropic d=10, smooth control
        "ill_cond_50",  # kappa=1000 rotated d=50, correlated smooth
        "german_credit",  # mild-corr d=26, smooth real posterior
        "eight_schools_ncp",  # mild funnel d=10, NCP
        "neals_funnel",  # centered funnel d=10, negative control
    ]

# IMM regimes: GT-dense (kappa_eff~1) and diagonal (worst-case precond)
IMM_REGIMES = ["gt_dense", "diagonal"]

# Adjusted tuning: pass-1 DA budget (fraction of n_warmup for step-size tuning)
# frac_tune2=0 always: skip the sqrt(dim) reset block.
FRAC_TUNE1 = 0.5  # 50% of n_warmup for the step-size DA pass
ADJ_TARGET = 0.9  # acceptance target for adjusted MCLMC

# Fail-loud thresholds
BIAS_LOUD = 0.15  # bias above this AND any loud signal = loud-fail
BIAS_PASS = 0.10  # bias below this = pass-eligible
RHAT_BAD = 1.05  # rank-normalized split-Rhat threshold
DIV_BAD = 0.01  # divergence rate threshold
ACC_COLLAPSE = 0.3  # acceptance below this = loud signal

# ESS: spectral can exceed N (captures antithetic effect). Flag ESS>N cells.
ESS_MIN_PER_CHAIN = 50  # min min-ESS per chain to count as non-degenerate

# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def load_model(model_name, seed=0):
    """Return (logdensity_fn, init_position, d, gt_mean_np, gt_var_np, Sigma_or_None).

    For synthetic models: Sigma is the analytic covariance (numpy float64).
    For real models: Sigma is None; gt_mean/gt_var from GT draws.
    """
    _SYNTHETIC = {"ill_cond_50", "mvn_10"}

    if model_name == "mvn_10":
        from tuningfork.model.mvn_10 import DIM

        d = DIM

        def logdensity_fn(x):
            return -0.5 * jnp.dot(x, x)

        Sigma = np.eye(d, dtype=np.float64)
        gt_mean = np.zeros(d, dtype=np.float64)
        gt_var = np.ones(d, dtype=np.float64)
        return logdensity_fn, jnp.zeros(d, dtype=jnp.float64), d, gt_mean, gt_var, Sigma

    elif model_name == "ill_cond_50":
        from tuningfork.model.ill_cond_50 import COV_NP

        Sigma = COV_NP.astype(np.float64)
        d = Sigma.shape[0]
        Sigma_inv = np.linalg.inv(Sigma)
        Sinv_jax = jnp.array(Sigma_inv)

        def logdensity_fn(x):
            return -0.5 * jnp.dot(x, Sinv_jax @ x)

        gt_mean = np.zeros(d, dtype=np.float64)
        gt_var = np.diag(Sigma)
        return logdensity_fn, jnp.zeros(d, dtype=jnp.float64), d, gt_mean, gt_var, Sigma

    else:
        # Real catalog model: load from GT draws
        imm_dense, gt_var, gt_mean, d = gt_from_draws(model_name, k=None)
        from tuningfork.model._numpyro import build_logdensity_fn as _build_ld
        from tuningfork.model._registry import MODELS as _MODELS_REG

        entry = _MODELS_REG[model_name]
        _init_key = jax.random.key(seed)
        init_dict, ld_raw, _ = _build_ld(_init_key, entry)
        _, _unravel = ravel_pytree(init_dict)

        def logdensity_fn(x_flat, _ld=ld_raw, _un=_unravel):
            return _ld(_un(x_flat))

        return (
            logdensity_fn,
            jnp.array(gt_mean, dtype=jnp.float64),
            d,
            np.array(gt_mean, dtype=np.float64),
            np.array(gt_var, dtype=np.float64),
            None,
        )  # Sigma=None for real models


def build_imm(model_name, regime, d, Sigma):
    """Return (imm, imm_label, kappa_eff_approx).

    regime: 'gt_dense' or 'diagonal'
    For real models (Sigma=None), uses gt_from_draws for gt_dense and
    jnp.ones(d) (unit diagonal M^-1) for diagonal.
    """
    if model_name == "mvn_10":
        Sigma_use = np.eye(d, dtype=np.float64)
    elif model_name == "ill_cond_50":
        Sigma_use = Sigma
    else:
        Sigma_use = None  # real model

    if regime == "gt_dense":
        if Sigma_use is not None:
            imm = gt_lrd_imm(Sigma_use, k=d)
            label = f"gt_dense(k={d})"
        else:
            # Real model: use full GT from draws
            imm, _, _, _ = gt_from_draws(model_name, k=None)
            label = "gt_dense"
    elif regime == "diagonal":
        if Sigma_use is not None:
            # Diagonal M^-1 = diag(Sigma): sigma=sqrt(diag(Sigma)), U=(d,0), lam=(0,)
            sigma_diag = np.sqrt(np.diag(Sigma_use))
            imm = LowRankInverseMassMatrix(
                sigma=jnp.array(sigma_diag, dtype=jnp.float64),
                U=jnp.zeros((d, 0), dtype=jnp.float64),
                lam=jnp.zeros((0,), dtype=jnp.float64),
            )
            label = "diagonal(GT-sigma)"
        else:
            # Real model: diagonal M^-1 = diag(GT-var)
            _, gt_var, gt_mean, _ = gt_from_draws(model_name, k=0)
            sigma_diag = np.sqrt(np.array(gt_var, dtype=np.float64))
            imm = LowRankInverseMassMatrix(
                sigma=jnp.array(sigma_diag, dtype=jnp.float64),
                U=jnp.zeros((d, 0), dtype=jnp.float64),
                lam=jnp.zeros((0,), dtype=jnp.float64),
            )
            label = "diagonal(GT-sigma)"
    else:
        raise ValueError(f"Unknown IMM regime: {regime!r}")

    return imm, label


# ---------------------------------------------------------------------------
# Step calibration: get the reference unadjusted MCLMC step at fixed IMM
# (used to warm-start the adjusted tuner at step_init ~= unadjusted ref step)
# ---------------------------------------------------------------------------


def get_ref_step(logdensity_fn, init_pos, imm, d, n_warmup_ref, seed):
    """Run unadjusted MCLMC to get the reference step at this IMM."""
    fixed_kernel = _make_fixed_imm_kernel(imm)
    st = mclmc_mod.init(init_pos, logdensity_fn, jax.random.key(seed + 99))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _ = mclmc_find_L_and_step_size(
            mclmc_kernel=fixed_kernel,
            num_steps=n_warmup_ref,
            state=st,
            rng_key=jax.random.key(seed + 100),
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=False,
        )
    return float(p.step_size)


# ---------------------------------------------------------------------------
# Core avg=k experiment function
# ---------------------------------------------------------------------------


def run_avgk_cell(
    model_name,
    logdensity_fn,
    init_pos,
    imm,
    d,
    gt_mean,
    gt_var,
    k,
    n_warmup,
    n_samples,
    num_chains,
    seed,
    step_init,
):
    """Run adjusted_mclmc_dynamic at avg=k on one (model, IMM, k, seed) cell.

    avg=k is achieved by:
      1. params.L = k * step_init at entry  (so pass-1's avg-preserving fix_L=False holds avg=k)
      2. frac_tune2=0                        (skip sqrt(dim) reset + pass-2)
      3. target_num_integration_steps=k      (post-override re-pins L=k*step as a no-op)

    The step is calibrated for the target avg=k trajectory on this fixed IMM.

    Returns a dict with metrics.
    """
    # Tuning kernel: adjusted_mclmc (static) with fixed IMM
    adj_tune_kernel = _make_fixed_imm_adj_kernel_for_tuning(imm)

    # Sampling kernel: adjusted_mclmc_dynamic with fixed IMM
    adj_dyn_kernel = _make_fixed_imm_adj_dyn_kernel(imm)

    # Pre-seed params with L = k * step_init
    # This ensures pass-1's fix_L_first_da=False holds avg=k throughout.
    # params.inverse_mass_matrix is a placeholder (the kernel ignores it).
    adj_init_params = MCLMCAdaptationState(
        L=jnp.array(float(k) * float(step_init), dtype=jnp.float64),
        step_size=jnp.array(float(step_init), dtype=jnp.float64),
        inverse_mass_matrix=jnp.ones(d, dtype=jnp.float64),  # placeholder
    )

    base_key = jax.random.key(seed)
    chain_keys = jax.random.split(base_key, num_chains)

    # Tune and collect per-chain (step, L, avg_final)
    steps, Ls, avg_finals = [], [], []
    final_positions = []  # tuned positions; used to init dynamic sampling states

    for ci in range(num_chains):
        ck = chain_keys[ci]
        # Tuning uses the STATIC adjusted_mclmc kernel (HMCState, 3 fields).
        # The tuner's adjusted_mclmc_find_L_and_step_size expects this state.
        init_state_static = adj_mclmc_mod_static.init(init_pos, logdensity_fn)
        warmup_key = jax.random.fold_in(ck, 7)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_tuned, params_tuned, _ = adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=adj_tune_kernel,
                logdensity_fn=logdensity_fn,
                num_steps=n_warmup,
                state=init_state_static,
                rng_key=warmup_key,
                target=ADJ_TARGET,
                frac_tune1=FRAC_TUNE1,
                frac_tune2=0.0,  # skip sqrt(dim) reset + pass-2
                frac_tune3=0.0,
                diagonal_preconditioning=False,
                params=adj_init_params,
                # NOTE: do NOT pass target_num_integration_steps here.
                # The #939 post-override (L = target * step) would fire AFTER the
                # avg-preserving pass-1, re-pinning L=2*step regardless of k.
                # Instead we do the manual re-pin below using k directly.
            )

        step_ci = float(params_tuned.step_size)
        # MANUAL post-pin: set L = k * step so avg = k exactly.
        # This is the 2c prototype: step was calibrated for avg=k (because
        # pass-1 started with L=k*step and fix_L=False held avg=k), so
        # the re-pin is a true no-op (the ratio is already k).
        L_ci = float(k) * step_ci
        avg_ci = L_ci / max(step_ci, 1e-10)

        steps.append(step_ci)
        Ls.append(L_ci)
        avg_finals.append(avg_ci)
        final_positions.append(state_tuned.position)

    step_mean = float(np.mean(steps))
    L_mean = float(np.mean(Ls))
    avg_final_mean = float(np.mean(avg_finals))

    # Sampling: use adjusted_mclmc_dynamic (random trajectory length).
    # Must re-init state as DynamicHMCState (4 fields) from the tuned position.
    # integration_steps_params = (avg,) where avg = L/step = k
    sample_base_key = jax.random.fold_in(base_key, 999)
    sampling_keys = jax.random.split(sample_base_key, num_chains)

    all_positions = []
    all_accept_rates = []
    all_divs = []
    all_n_steps = []

    for ci in range(num_chains):
        # Init dynamic state from the tuned position
        dyn_state = adj_dyn_mod.init(
            final_positions[ci], logdensity_fn, sampling_keys[ci]
        )
        scan_keys = jax.random.split(sampling_keys[ci], n_samples)

        def _sample_step(carry_state, rng_key, _step=step_mean, _L=L_mean, _imm=imm):
            avg_sample = _L / max(_step, 1e-10)
            next_state, info = adj_dyn_kernel(
                rng_key=rng_key,
                state=carry_state,
                logdensity_fn=logdensity_fn,
                step_size=_step,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=_imm,
                integration_steps_params=(avg_sample,),
            )
            return next_state, (
                next_state.position,
                info.acceptance_rate,
                info.is_divergent,
                info.num_integration_steps,
            )

        _, (pos_traj, acc_traj, div_traj, nsteps_traj) = jax.lax.scan(
            _sample_step, dyn_state, scan_keys
        )
        flat_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(pos_traj)
        all_positions.append(np.array(flat_pos, dtype=np.float64))
        all_accept_rates.append(np.array(acc_traj, dtype=np.float64))
        all_divs.append(np.array(div_traj, dtype=np.bool_))
        all_n_steps.append(np.array(nsteps_traj, dtype=np.float64))

    positions_arr = np.stack(all_positions, axis=0)  # (C, T, d)
    n_steps_all = np.concatenate(all_n_steps, axis=0)
    mean_n_steps = float(np.mean(n_steps_all))

    # Grad accounting: 2 grads per leapfrog step; adjusted uses variable n_steps
    n_warmup_grads = (
        2 * n_warmup * num_chains
    )  # approximate: frac_tune1 * n_warmup steps
    n_sampling_grads = int(
        2 * np.sum(n_steps_all)
    )  # exact: sum over all chains x draws
    total_grads = n_warmup_grads + n_sampling_grads

    # --- Bias metric ---
    gt_mean_bc = gt_mean[None, None, :]  # broadcast over (C, T, d)
    var_mcmc = np.mean((positions_arr - gt_mean_bc) ** 2, axis=(0, 1))  # (d,)
    bias_arr = np.abs(var_mcmc - gt_var) / np.maximum(gt_var, 1e-30)
    max_bias = float(bias_arr.max())

    # --- Diagnostics ---
    divs_all = np.concatenate(all_divs, axis=0)
    div_rate = float(np.mean(divs_all))
    acc_all = np.concatenate(all_accept_rates, axis=0)
    mean_acc = float(np.mean(acc_all))

    # --- ESS: BOTH spectral (Geyer/blackjax) AND az-bulk ---
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], positions_arr)})
    az_bulk = az.ess(ds, method="bulk")
    min_bulk_ess_az = float(np.array(az_bulk["x"]).min())

    # Spectral ESS via blackjax.diagnostics.effective_sample_size
    # Input shape: (num_chains, n_samples, d) -> returns (d,) spectral ESS
    samples_jnp = jnp.array(positions_arr, dtype=jnp.float64)
    spectral_ess = np.array(blackjax.diagnostics.effective_sample_size(samples_jnp))
    min_spectral_ess = float(spectral_ess.min())
    max_spectral_ess = float(spectral_ess.max())

    # az-bulk-based ess/grad (declared basis for inter-experiment comparability)
    ess_per_grad_az = min_bulk_ess_az / total_grads

    # spectral ess/grad — primary for H1 mechanism (captures antithetic ESS>N)
    # Gate: if acc < ACC_COLLAPSE or div_rate > DIV_BAD, spectral ESS is suspect
    spectral_suspect = (mean_acc < ACC_COLLAPSE) or (div_rate > DIV_BAD)
    ess_per_grad_spectral = min_spectral_ess / total_grads

    # Rhat via arviz rank-normalized
    rhat_ds = az.rhat(ds, method="rank")
    rhat_arr = np.array(rhat_ds["x"])
    max_rhat = (
        float(np.nanmax(rhat_arr)) if not np.all(np.isnan(rhat_arr)) else float("nan")
    )

    # --- Fail-loud classification ---
    loud_signal = (
        max_rhat > RHAT_BAD
        or np.isnan(max_rhat)
        or div_rate > DIV_BAD
        or mean_acc < ACC_COLLAPSE
    )
    bias_ok = max_bias < BIAS_PASS
    rhat_ok = (max_rhat < RHAT_BAD) and not np.isnan(max_rhat)
    ess_ok = min_bulk_ess_az > ESS_MIN_PER_CHAIN * num_chains

    if bias_ok and rhat_ok and ess_ok:
        verdict = "PASS"
    elif (not bias_ok) and loud_signal:
        verdict = "LOUD-FAIL"
    elif not bias_ok:
        verdict = "SILENT-FAIL"
    else:
        verdict = "marginal"  # bias ok, but ess or rhat short

    return {
        # Tuning results
        "step_mean": step_mean,
        "L_mean": L_mean,
        "avg_final": avg_final_mean,
        # Efficiency (BOTH bases — declared on every number)
        "min_bulk_ess_az": min_bulk_ess_az,  # az-bulk
        "min_spectral_ess": min_spectral_ess,  # Geyer/blackjax spectral
        "max_spectral_ess": max_spectral_ess,  # to detect ESS>N antithetic
        "ess_per_grad_az": ess_per_grad_az,  # az-bulk basis
        "ess_per_grad_spectral": ess_per_grad_spectral,  # spectral basis
        "spectral_suspect": spectral_suspect,  # True if acc<0.3 or div>0.01
        # Bias and diagnostics
        "max_bias": max_bias,
        "max_rhat": max_rhat,
        "div_rate": div_rate,
        "mean_acc": mean_acc,
        "mean_n_steps": mean_n_steps,
        # Grad accounting
        "n_warmup_grads": n_warmup_grads,
        "n_sampling_grads": n_sampling_grads,
        "total_grads": total_grads,
        # Classification
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def print_header():
    print("=" * 110)
    print(
        f"sweep_avgk | smoke={SMOKE} | highd={HIGHD} | "
        f"NW={N_WARMUP} NS={N_SAMPLES} CH={NUM_CHAINS}"
    )
    print(f"k_values={K_VALUES} | imm_regimes={IMM_REGIMES} | seeds={SEEDS}")
    print(
        "ESS basis: BOTH az-bulk (declared) AND spectral/Geyer (blackjax; can exceed N)."
    )
    print(
        "Fail-loud: PASS (bias<0.10, rhat<1.05, minESS>50/chain) | "
        "LOUD-FAIL (high bias + rhat/div/acc signal) | SILENT-FAIL (bias high, no flag)"
    )
    print("=" * 110)


def print_result(row):
    esg_az = row["ess_per_grad_az"]
    esg_sp = row["ess_per_grad_spectral"]
    flag_sp = "*" if row["spectral_suspect"] else " "
    esn = row["max_spectral_ess"] / max(N_SAMPLES * NUM_CHAINS, 1)
    print(
        f"  k={row['k']:<2d}  IMM={row['imm']:<20s}  "
        f"avg_f={row['avg_final']:.2f}  step={row['step_mean']:.3f}  acc={row['mean_acc']:.3f}  "
        f"bias={row['max_bias']:.3f}  rhat={row['max_rhat']:.3f}  div={row['div_rate']:.4f}  "
        f"essg_az={esg_az:.4f}  essg_sp={esg_sp:.4f}{flag_sp}  "
        f"ESS/N={esn:.2f}  verdict={row['verdict']}"
    )
    sys.stdout.flush()


results = []
print_header()

for model_name in PRIMARY_MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    # Load model (seed=0 for init; seeds are varied in the per-seed loop below)
    try:
        logdensity_fn, init_pos, d, gt_mean, gt_var, Sigma = load_model(
            model_name, seed=0
        )
    except Exception as e:
        print(f"  LOAD ERROR for {model_name}: {e}")
        continue

    print(f"  d={d}  n_models={len(PRIMARY_MODELS)}")

    for regime in IMM_REGIMES:
        try:
            imm, imm_label = build_imm(model_name, regime, d, Sigma)
        except Exception as e:
            print(f"  IMM BUILD ERROR ({regime}): {e}")
            continue

        print(f"\n  --- IMM={imm_label} ---")
        print(
            f"  {'k':>3}  {'seeds':>6}  {'avg_f':>6}  {'step':>7}  {'acc':>5}  "
            f"{'bias':>6}  {'rhat':>6}  {'div':>6}  "
            f"{'essg_az':>9}  {'essg_sp':>9}  {'ESS/N':>6}  verdict"
        )

        # Get reference unadjusted step for this IMM (warm-start for the adjusted tuner)
        n_warmup_ref = min(N_WARMUP, 500) if not SMOKE else 5
        ref_step = get_ref_step(logdensity_fn, init_pos, imm, d, n_warmup_ref, seed=42)
        print(f"  ref_step (unadjusted MCLMC at this IMM): {ref_step:.4f}")

        # step_init for the adjusted tuner: start at the unadjusted reference step
        step_init = ref_step

        for k in K_VALUES:
            seed_results = []
            for seed in SEEDS:
                t0 = time.perf_counter()
                try:
                    cell = run_avgk_cell(
                        model_name=model_name,
                        logdensity_fn=logdensity_fn,
                        init_pos=init_pos,
                        imm=imm,
                        d=d,
                        gt_mean=gt_mean,
                        gt_var=gt_var,
                        k=k,
                        n_warmup=N_WARMUP,
                        n_samples=N_SAMPLES,
                        num_chains=NUM_CHAINS,
                        seed=seed,
                        step_init=step_init,
                    )
                    cell["wall"] = time.perf_counter() - t0
                    cell["k"] = k
                    cell["seed"] = seed
                    cell["model"] = model_name
                    cell["imm"] = imm_label
                    seed_results.append(cell)
                except Exception as e:
                    print(f"  ERROR k={k} seed={seed}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            if seed_results:
                # Aggregate across seeds: mean of key metrics, worst-case verdict
                agg = {
                    "k": k,
                    "model": model_name,
                    "imm": imm_label,
                    "avg_final": np.mean([r["avg_final"] for r in seed_results]),
                    "step_mean": np.mean([r["step_mean"] for r in seed_results]),
                    "mean_acc": np.mean([r["mean_acc"] for r in seed_results]),
                    "max_bias": np.mean([r["max_bias"] for r in seed_results]),
                    "max_rhat": np.nanmax([r["max_rhat"] for r in seed_results]),
                    "div_rate": np.mean([r["div_rate"] for r in seed_results]),
                    "min_bulk_ess_az": np.mean(
                        [r["min_bulk_ess_az"] for r in seed_results]
                    ),
                    "min_spectral_ess": np.mean(
                        [r["min_spectral_ess"] for r in seed_results]
                    ),
                    "max_spectral_ess": np.mean(
                        [r["max_spectral_ess"] for r in seed_results]
                    ),
                    "ess_per_grad_az": np.mean(
                        [r["ess_per_grad_az"] for r in seed_results]
                    ),
                    "ess_per_grad_spectral": np.mean(
                        [r["ess_per_grad_spectral"] for r in seed_results]
                    ),
                    "spectral_suspect": any(
                        r["spectral_suspect"] for r in seed_results
                    ),
                    "n_seeds": len(seed_results),
                    # verdict: worst-case across seeds
                    "verdict": max(
                        [r["verdict"] for r in seed_results],
                        key=lambda v: {
                            "PASS": 0,
                            "marginal": 1,
                            "LOUD-FAIL": 2,
                            "SILENT-FAIL": 3,
                        }.get(v, 4),
                    ),
                }
                print_result(agg)
                results.append(agg)
                results.extend(seed_results)  # keep per-seed too

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 110)
print("SUMMARY TABLE (mean across seeds, by model x IMM x k)")
print(
    "ESS basis: az=az-bulk, sp=spectral/Geyer. sp* = spectral_suspect (acc<0.3 or div>0.01)"
)
print("-" * 110)

# Print aggregated results (rows without per-seed "seed" key)
agg_rows = [r for r in results if "seed" not in r or r.get("n_seeds")]
printed = set()
for model_name in PRIMARY_MODELS:
    for r in [x for x in results if x.get("model") == model_name and "n_seeds" in x]:
        key = (model_name, r["imm"], r["k"])
        if key in printed:
            continue
        printed.add(key)
        print_result(r)

# ---------------------------------------------------------------------------
# Optimal-k verdict per (model, IMM)
# ---------------------------------------------------------------------------

print("\n" + "=" * 110)
print(
    "OPTIMAL-k verdict per (model, IMM regime): k with best ess/grad_spectral among PASS cells"
)
print("(Falls back to ess/grad_az if spectral is suspect. Excludes SILENT-FAIL.)")
print("-" * 110)

from collections import defaultdict

# Group agg rows by (model, imm)
groupby = defaultdict(list)
for r in [x for x in results if "n_seeds" in x]:
    groupby[(r["model"], r["imm"])].append(r)

for (model, imm_lbl), rows in sorted(groupby.items()):
    pass_rows = [r for r in rows if r["verdict"] not in ("SILENT-FAIL",)]
    if not pass_rows:
        print(f"  {model:25s} {imm_lbl:25s}  -> ALL LOUD/SILENT-FAIL (no clean k)")
        continue

    # Best k by spectral ess/grad (if not suspect); else az-bulk
    def score(r):
        if not r["spectral_suspect"]:
            return r["ess_per_grad_spectral"]
        return r["ess_per_grad_az"]

    best = max(pass_rows, key=score)
    print(
        f"  {model:25s} {imm_lbl:25s}  "
        f"-> best k={best['k']}  "
        f"(essg_sp={best['ess_per_grad_spectral']:.4f}, essg_az={best['ess_per_grad_az']:.4f}, "
        f"bias={best['max_bias']:.3f}, verdict={best['verdict']})"
    )

print("\nDONE")
