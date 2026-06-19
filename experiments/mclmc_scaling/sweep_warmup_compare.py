"""Controlled A/B comparison: shipped mclmc_lrd_warmup (baseline) vs enhanced (E1+E2).

For each model x warmup-budget combination, runs BOTH warmup variants and records:

  Metric 1 -- accuracy vs gold:
    |step - step_gold| / step_gold
    |L - L_gold| / L_gold
    kappa_eff(GT_Sigma, warmup_IMM)   (IMM accuracy; lower = closer to GT whitening)
    For synthetic models: GT_Sigma is the analytic covariance.
    For real models: GT_Sigma is the DENSE sample covariance of the GT draws
    (full (d,d) matrix from np.cov of flat_draws), giving a correct kappa_eff
    measurement on correlated real posteriors.

  Metric 0 -- sample quality:
    Feed each warmup's (step, L, IMM) into run_fixed_imm.run_mclmc_fixed_imm
    with fixed_step_size / fixed_L to get: min_bulk_ESS, ess_per_grad, max_bias.

  Metric 2 -- speed:
    Captured implicitly by the accuracy-vs-budget ladder.

MODELS (smooth panel):
  mvn_10         -- Gaussian d=10, isotropic (analytic Sigma=I)
  ill_cond_50    -- Gaussian d=50, kappa=1000 rotated (analytic Sigma from COV_NP)
  german_credit  -- real model d=26 (GT from catalog draws)
  irt_1pl        -- real model d=500 (GT from catalog draws)

BUDGETS (pilot_num_warmup / pilot_num_samples, lrd_num_steps fixed at 500):
  Per-model ladders (ill_cond_50 needs more pilot to reach n_eff >= 2*k*~40):
    mvn_10, german_credit, irt_1pl: (200,200), (500,500), (1000,1000), (2000,2000)
    ill_cond_50:                    (2000,2000), (5000,5000), (10000,10000)

GOLD REFERENCE:
  Computed once per model from the BASELINE at the LARGEST budget in its ladder.
  step_gold = baseline step at largest budget
  L_gold    = baseline L at largest budget

Run:
  python sweep_warmup_compare.py --smoke      # mvn_10 + ill_cond_50, 2 smallest budgets each
  python sweep_warmup_compare.py              # full panel (TL runs)

Output: sweep_warmup_compare_results.json (or _smoke.json for --smoke)
"""

import argparse
import json
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Path setup: make experiment helpers importable
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import lrd_warmup_baseline as _baseline_mod
import lrd_warmup_enhanced as _enhanced_mod
from gt_imm import gt_cov, gt_from_draws, gt_lrd_imm
from gt_imm import kappa_eff as kappa_eff_fn
from run_fixed_imm import run_mclmc_fixed_imm

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_SYNTHETIC_MODELS = {"mvn_10", "ill_cond_50"}


def _get_model_logdensity_and_gt(model_name: str, seed: int = 0):
    """Return GT components for a named model.

    For synthetic models: gt_Sigma is the analytic covariance matrix.
    For real models: gt_Sigma is the DENSE sample covariance of the GT draws
    (np.cov of flat draws, shape (d,d)) — used by _compute_kappa_eff_of_imm
    for a correct kappa_eff measurement on correlated real posteriors.

    Returns
    -------
    logdensity_fn : callable
        Flat-input log-density.
    init_position : jnp.ndarray, shape (d,)
    d : int
    gt_Sigma : np.ndarray, shape (d, d)
        Dense GT covariance (analytic for synthetic; sample cov for real).
    gt_imm_dense : LowRankInverseMassMatrix
        GT dense IMM.
    gt_var : np.ndarray, shape (d,)
    gt_mean : np.ndarray, shape (d,)
    """
    if model_name == "mvn_10":
        from tuningfork.model.mvn_10 import DIM

        d = DIM
        Sigma = np.eye(d, dtype=np.float64)

        def logdensity_fn(x):
            return -0.5 * jnp.dot(x, x)

        init_position = jnp.zeros(d, dtype=jnp.float64)
        gt_imm_dense = gt_lrd_imm(Sigma, k=d)
        return (
            logdensity_fn,
            init_position,
            d,
            Sigma,
            gt_imm_dense,
            np.diag(Sigma),
            np.zeros(d),
        )

    elif model_name == "ill_cond_50":
        from tuningfork.model.ill_cond_50 import COV_NP

        Sigma = COV_NP.astype(np.float64)
        Sigma_inv_jax = jnp.array(np.linalg.inv(Sigma), dtype=jnp.float64)
        d = Sigma.shape[0]

        def logdensity_fn(x):
            return -0.5 * jnp.dot(x, Sigma_inv_jax @ x)

        init_position = jnp.zeros(d, dtype=jnp.float64)
        gt_imm_dense = gt_lrd_imm(Sigma, k=d)
        return (
            logdensity_fn,
            init_position,
            d,
            Sigma,
            gt_imm_dense,
            np.diag(Sigma),
            np.zeros(d),
        )

    else:
        # Real catalog model
        import os

        from blackjax.adaptation.mclmc_lrd_adaptation import _extract_lrd_from_samples
        from jax.flatten_util import ravel_pytree as _ravel_pytree

        from tuningfork.model._numpyro import (
            build_logdensity_fn as _build_logdensity_fn,
        )
        from tuningfork.model._registry import MODELS

        entry = MODELS[model_name]
        d = entry.dim

        _init_key = jax.random.key(seed)
        _init_pos_dict, _logdensity_fn_raw, _postprocess_fn = _build_logdensity_fn(
            _init_key, entry
        )
        _flat0, _unravel_fn = _ravel_pytree(_init_pos_dict)

        def logdensity_fn(x_flat):
            return _logdensity_fn_raw(_unravel_fn(x_flat))

        gt_imm_dense, gt_var, gt_mean, d2 = gt_from_draws(model_name, k=None)
        assert d2 == d, f"dim mismatch for {model_name}: {d2} != {d}"

        # Compute the DENSE sample covariance of the GT draws for kappa_eff.
        # gt_from_draws builds the IMM from flat_draws but doesn't expose the
        # raw flat array.  Re-load the draws to get the (d,d) sample cov.
        _catalog_root = os.path.join(_HERE, "..", "..", "tuningfork", "catalog")
        _draws_path = os.path.join(
            _catalog_root, model_name, "groundtruth_samples", "blackjax", "draws.npz"
        )
        _data = np.load(_draws_path)
        _draw_keys = list(_data.files)
        # vmap-ravel to (n_total, d) — mirrors gt_from_draws
        from jax.flatten_util import ravel_pytree as _rvp2

        _sample_0 = {k: jnp.array(_data[k][0], dtype=jnp.float64) for k in _draw_keys}
        _flat_batch = {k: jnp.array(_data[k].astype(np.float64)) for k in _draw_keys}
        _flat_draws = jax.vmap(lambda pos: _rvp2(pos)[0])(_flat_batch)  # (n, d)
        _flat_draws_np = np.array(_flat_draws, dtype=np.float64)
        gt_Sigma = np.cov(_flat_draws_np, rowvar=False)  # (d, d)

        init_position = jnp.array(gt_mean, dtype=jnp.float64)
        return logdensity_fn, init_position, d, gt_Sigma, gt_imm_dense, gt_var, gt_mean


def _compute_kappa_eff_of_imm(model_name: str, gt_Sigma, warmup_imm) -> float:
    """Compute kappa_eff(GT_Sigma, warmup_IMM).

    gt_Sigma is always a (d,d) dense matrix:
      - Synthetic models: analytic covariance (exact).
      - Real models: dense sample covariance of the GT draws (from gt_from_draws
        + np.cov), giving a correct kappa_eff on correlated real posteriors.
    """
    return kappa_eff_fn(gt_Sigma, warmup_imm)


# ---------------------------------------------------------------------------
# Core run function: run both warmups, collect all metrics
# ---------------------------------------------------------------------------


def run_one(
    model_name: str,
    pilot_num_warmup: int,
    pilot_num_samples: int,
    k_requested: int,
    num_chains: int,
    seed: int,
    n_sampling: int,
    gold_step: float | None,
    gold_L: float | None,
) -> dict:
    """Run baseline and enhanced warmups on one (model, budget) point.

    Returns a dict with keys 'baseline' and 'enhanced', each containing
    the warmup result + downstream metrics.
    """
    lrd_num_steps = 500  # fixed across all experiments

    logdensity_fn, init_position, d, gt_Sigma, gt_imm_dense, gt_var, gt_mean = (
        _get_model_logdensity_and_gt(model_name, seed=seed)
    )

    _is_real = model_name not in _SYNTHETIC_MODELS
    gt_mean_kwarg = gt_mean if _is_real else None
    gt_var_kwarg = gt_var if _is_real else None

    results = {}

    for warmup_tag, warmup_mod in [
        ("baseline", _baseline_mod),
        ("enhanced", _enhanced_mod),
    ]:
        rng_key = jax.random.key(seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warmup_result = warmup_mod.mclmc_lrd_warmup(
                logdensity_fn=logdensity_fn,
                position=init_position,
                rng_key=rng_key,
                k=k_requested,
                pilot_num_warmup=pilot_num_warmup,
                pilot_num_samples=pilot_num_samples,
                lrd_num_steps=lrd_num_steps,
                num_chains=num_chains,
                inner_kernel="mclmc",
            )

        step_out = float(warmup_result.step_size)
        L_out = float(warmup_result.L)
        diag = warmup_result.diagnostics

        # Metric 1a: accuracy vs gold (if gold is available)
        if gold_step is not None:
            step_err = abs(step_out - gold_step) / max(abs(gold_step), 1e-10)
            L_err = abs(L_out - gold_L) / max(abs(gold_L), 1e-10)
        else:
            step_err = None
            L_err = None

        # Metric 1b: IMM accuracy (kappa_eff of warmup IMM vs GT Sigma)
        warmup_imm = warmup_result.inverse_mass_matrix
        keff_imm = _compute_kappa_eff_of_imm(model_name, gt_Sigma, warmup_imm)

        # Metric 0: sample quality — feed (step, L, IMM) into fixed run
        sample_result = run_mclmc_fixed_imm(
            model_name=model_name,
            imm=warmup_imm,
            n_warmup=200,  # minimal warmup (just to init state)
            n_samples=n_sampling,
            num_chains=num_chains,
            seed=seed + 7,
            fixed_step_size=step_out,
            fixed_L=L_out,
            gt_mean=gt_mean_kwarg,
            gt_var=gt_var_kwarg,
        )

        results[warmup_tag] = {
            # Warmup outputs
            "step": step_out,
            "L": L_out,
            "k_used": diag.get("k_used"),
            "k_safe": diag.get("k_safe"),
            # E2 extras (only in enhanced)
            "k_star": diag.get("k_star"),
            "kappa_eff_at_k_star": diag.get("kappa_eff_at_k_star"),
            # E1 gate (only in enhanced)
            "e1_fired": diag.get("e1_fired"),
            "e1_kappa_eff_at_k_used": diag.get("e1_kappa_eff_at_k_used"),
            # Metric 1
            "step_err_vs_gold": step_err,
            "L_err_vs_gold": L_err,
            "kappa_eff_imm": keff_imm,
            # Metric 0
            "ess_per_grad": sample_result["ess_per_grad"],
            "min_bulk_ess": sample_result["min_bulk_ess"],
            "max_bias": sample_result["max_bias"],
        }
        _e1_fired = diag.get("e1_fired")
        _e1_keff = diag.get("e1_kappa_eff_at_k_used", float("nan"))
        if _e1_fired is not None:
            e1_tag = f"  e1={'Y' if _e1_fired else 'N'}(keff={_e1_keff:.1f})"
        else:
            e1_tag = ""
        print(
            f"  [{warmup_tag:8s}] step={step_out:.3f}  L={L_out:.3f}  "
            f"k_used={diag.get('k_used')}  kappa_eff_imm={keff_imm:.2f}  "
            f"ess/grad={sample_result['ess_per_grad']:.5f}{e1_tag}"
        )
        sys.stdout.flush()

    return results


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

FULL_MODELS = ["mvn_10", "ill_cond_50", "german_credit", "irt_1pl"]
SMOKE_MODELS = ["mvn_10", "ill_cond_50"]

# Per-model budget ladders.
# ill_cond_50 needs n_eff >= 2*k*~40 => pilot >= ~2000 draws to allow E2 room.
# Other smooth models certify with smaller budgets.
MODEL_BUDGETS = {
    "mvn_10": [(200, 200), (500, 500), (1000, 1000), (2000, 2000)],
    "ill_cond_50": [(2000, 2000), (5000, 5000), (10000, 10000)],
    "german_credit": [(200, 200), (500, 500), (1000, 1000), (2000, 2000)],
    "irt_1pl": [(200, 200), (500, 500), (1000, 1000), (2000, 2000)],
}

# Smoke: the TWO smallest budgets from each model's ladder
MODEL_SMOKE_BUDGETS = {model: budgets[:2] for model, budgets in MODEL_BUDGETS.items()}

# k to REQUEST for each model (generous; E2 will find the right one)
MODEL_K = {
    "mvn_10": 10,
    "ill_cond_50": 49,
    "german_credit": 20,
    "irt_1pl": 100,
}

NUM_CHAINS = 4
SEED = 42
N_SAMPLING = 500  # short fixed sampling run for Metric 0


def run_sweep(models, budget_map, outfile):
    """Run the A/B sweep.

    Parameters
    ----------
    models : list[str]
        Model names to run.
    budget_map : dict[str, list[tuple[int,int]]]
        Per-model list of (pilot_num_warmup, pilot_num_samples) budgets.
        Gold reference = baseline at the LARGEST budget in each model's list.
    outfile : str
        Path for the JSON output.
    """
    all_results = {}

    for model_name in models:
        print(f"\n=== MODEL: {model_name} ===")
        sys.stdout.flush()

        k_req = MODEL_K[model_name]
        budgets = budget_map[model_name]
        model_results = {}

        # First pass: compute gold reference from BASELINE at the largest budget
        largest_budget = budgets[-1]
        n_pilot_w_gold, n_pilot_s_gold = largest_budget
        gold_key = f"{n_pilot_w_gold}/{n_pilot_s_gold}"
        print(f"  Computing gold reference (baseline at {gold_key})...")
        sys.stdout.flush()

        logdensity_fn, init_position, d, gt_Sigma, gt_imm_dense, gt_var, gt_mean = (
            _get_model_logdensity_and_gt(model_name, seed=SEED)
        )
        _is_real = model_name not in _SYNTHETIC_MODELS
        gt_mean_kwarg = gt_mean if _is_real else None
        gt_var_kwarg = gt_var if _is_real else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gold_warmup = _baseline_mod.mclmc_lrd_warmup(
                logdensity_fn=logdensity_fn,
                position=init_position,
                rng_key=jax.random.key(SEED),
                k=k_req,
                pilot_num_warmup=n_pilot_w_gold,
                pilot_num_samples=n_pilot_s_gold,
                lrd_num_steps=500,
                num_chains=NUM_CHAINS,
                inner_kernel="mclmc",
            )
        gold_step = float(gold_warmup.step_size)
        gold_L = float(gold_warmup.L)
        print(f"  Gold: step={gold_step:.4f}  L={gold_L:.4f}")
        sys.stdout.flush()

        # Second pass: run both warmups at each budget
        for n_pilot_w, n_pilot_s in budgets:
            budget_key = f"{n_pilot_w}/{n_pilot_s}"
            print(f"  Budget {budget_key}:")
            sys.stdout.flush()

            result = run_one(
                model_name=model_name,
                pilot_num_warmup=n_pilot_w,
                pilot_num_samples=n_pilot_s,
                k_requested=k_req,
                num_chains=NUM_CHAINS,
                seed=SEED,
                n_sampling=N_SAMPLING,
                gold_step=gold_step,
                gold_L=gold_L,
            )
            model_results[budget_key] = {
                "n_pilot_warmup": n_pilot_w,
                "n_pilot_samples": n_pilot_s,
                "gold_step": gold_step,
                "gold_L": gold_L,
                **result,
            }

        all_results[model_name] = model_results

    # Save results
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nResults saved to: {outfile}")
    sys.stdout.flush()
    return all_results


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------


def print_summary(results):
    """Print a compact summary table."""
    print("\n" + "=" * 90)
    print("SUMMARY — baseline vs enhanced warmup comparison")
    print("=" * 90)
    header = (
        f"{'model':<16} {'budget':>10}  "
        f"{'b_step':>8} {'e_step':>8}  "
        f"{'b_L':>7} {'e_L':>7}  "
        f"{'b_keff':>8} {'e_keff':>8}  "
        f"{'b_ess/g':>10} {'e_ess/g':>10}"
    )
    print(header)
    print("-" * 90)

    for model_name, model_results in results.items():
        for budget_key, br in model_results.items():
            b = br.get("baseline", {})
            e = br.get("enhanced", {})
            print(
                f"{model_name:<16} {budget_key:>10}  "
                f"{b.get('step', float('nan')):>8.3f} {e.get('step', float('nan')):>8.3f}  "
                f"{b.get('L', float('nan')):>7.3f} {e.get('L', float('nan')):>7.3f}  "
                f"{b.get('kappa_eff_imm', float('nan')):>8.2f} {e.get('kappa_eff_imm', float('nan')):>8.2f}  "
                f"{b.get('ess_per_grad', float('nan')):>10.5f} {e.get('ess_per_grad', float('nan')):>10.5f}"
            )
    print("=" * 90)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="A/B comparison: baseline vs enhanced mclmc_lrd_warmup"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: only mvn_10 + ill_cond_50 at two smallest budgets.",
    )
    args = parser.parse_args()

    if args.smoke:
        models = SMOKE_MODELS
        budget_map = MODEL_SMOKE_BUDGETS
        outfile = os.path.join(_HERE, "sweep_warmup_compare_smoke.json")
        print(
            "SMOKE MODE: models={}, budgets per model={}".format(
                models, {m: budget_map[m] for m in models}
            )
        )
    else:
        models = FULL_MODELS
        budget_map = MODEL_BUDGETS
        outfile = os.path.join(_HERE, "sweep_warmup_compare_results.json")
        print(
            "FULL SWEEP: models={}, budgets per model={}".format(
                models, {m: budget_map[m] for m in models}
            )
        )

    sys.stdout.flush()

    results = run_sweep(models, budget_map, outfile)
    print_summary(results)


if __name__ == "__main__":
    main()
