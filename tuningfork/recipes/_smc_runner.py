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
"""SMC recipe emit pipeline — particle init + SMC run + smc_gate + SMCRecipe.

Parallel to ``_recipe_runner.emit_low_recipe_for_cell`` for MCMC, this
module provides:

  ``emit_smc_recipe_for_cell(model_name, smc_method_name, inner_method_name, ...)``
      Full pipeline: build logfns → init particles → build SMC algorithm →
      run_smc → smc_gate → save SMCRecipe.

Build order (Phase 8B.1):
  W1/W2 (build_smc_logfns, build_prior_sample_fn) →
  W3/W6 (SMCRecipe, parameter_update_registry) →
  W4 (smc_gate) → **this file** →
  S4 smoke (logistic_synthetic + HMC).

Primary primitive: ``inner_kernel_tuning_smc(adaptive_tempered_smc, HMC)``
  with combined step_size + diagonal-IMM adaptation from particle cloud and
  acceptance rates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.smc_gate import SMCGateVerdict, smc_gate
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_prior_sample_fn, build_smc_logfns
from tuningfork.recipes._base_smc import SMCRecipe
from tuningfork.runner.smc import init_particles_from_prior, run_smc
from tuningfork.smc import SMC_METHODS
from tuningfork.smc.parameter_update_registry import build_parameter_update_fn

__all__ = ["SMCCellResult", "emit_smc_recipe_for_cell"]

_CATALOG_ROOT: Path = Path(__file__).parent.parent / "catalog"
_OUTCOMES_FILE: Path = _CATALOG_ROOT / "_smc_outcomes.md"

# Default SMC configuration
_DEFAULT_NUM_PARTICLES: int = 1000
_DEFAULT_MAX_STEPS: int = 200

# Default initial inner-kernel params for HMC inner kernel.
# These are updated by the parameter_update_fn during the tempering ladder.
_DEFAULT_HMC_STEP_SIZE: float = 0.1
_DEFAULT_HMC_NUM_STEPS: int = 10  # fixed-L for HMC inner kernel


# ---------------------------------------------------------------------------
# SMCCellResult
# ---------------------------------------------------------------------------


@dataclass
class SMCCellResult:
    """Outcome of one SMC emit attempt."""

    model_name: str
    smc_method_name: str
    inner_method_name: str
    verdict: str  # "PASS" | "FAIL" | "ERROR"
    recipe_path: Path | None = None
    gate_verdict: SMCGateVerdict | None = None
    particle_ess: float | None = None
    max_abs_mean_z: float | None = None
    headline_metric: float | None = None
    wall_seconds: float = 0.0
    note: str = ""

    def __repr__(self) -> str:
        return (
            f"SMCCellResult({self.model_name}/{self.smc_method_name}/"
            f"{self.inner_method_name} verdict={self.verdict} "
            f"wall={self.wall_seconds:.1f}s)"
        )


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def _append_outcome(
    model: str,
    smc_method: str,
    inner: str,
    message: str,
    outcomes_file: Path,
) -> None:
    try:
        outcomes_file.parent.mkdir(parents=True, exist_ok=True)
        with outcomes_file.open("a") as f:
            f.write(f"- [{model}/{smc_method}/{inner}] {message}\n")
    except Exception:  # noqa: BLE001
        pass  # non-fatal


# ---------------------------------------------------------------------------
# emit_smc_recipe_for_cell
# ---------------------------------------------------------------------------


def emit_smc_recipe_for_cell(
    model_name: str,
    smc_method_name: str,
    inner_method_name: str,
    *,
    num_particles: int = _DEFAULT_NUM_PARTICLES,
    max_steps: int = _DEFAULT_MAX_STEPS,
    smc_params: dict[str, Any] | None = None,
    inner_params_init: dict[str, Any] | None = None,
    parameter_update_strategy: str = "step_size_and_imm_from_particles",
    parameter_update_strategy_kwargs: dict[str, Any] | None = None,
    seed: int = 20260517,
    catalog_root: Path = _CATALOG_ROOT,
    outcomes_file: Path = _OUTCOMES_FILE,
    verbose: bool = True,
) -> SMCCellResult:
    """Run the full SMC cert pipeline and emit an SMCRecipe on PASS.

    Pipeline:
      1. Build logprior_fn + loglikelihood_fn from the NumPyro model (W1).
      2. Build prior_sample_fn for particle initialisation (W2).
      3. Initialise particles from the prior.
      4. Build the SMC algorithm (SMC_METHODS[smc_method_name] with the
         inner kernel from BASE_METHODS[inner_method_name]).
      5. Run the SMC tempering ladder via ``run_smc``.
      6. Compute cert metrics via ``smc_gate`` (W4).
      7. Compute headline metric: particle_ess / total_grad_evals (HMC) or
         / total_likelihood_evals (RWM).
      8. Save SMCRecipe on PASS.

    Parameters
    ----------
    model_name
        Model registry key (must be in MODELS).
    smc_method_name
        SMC_METHODS key (e.g. ``"inner_kernel_tuning"`` or
        ``"adaptive_tempered_smc"``).
    inner_method_name
        BASE_METHODS key for the inner MCMC kernel (e.g. ``"hmc"``).
    num_particles
        Number of SMC particles (default 1000).
    max_steps
        Maximum tempering steps for the while_loop (default 200).
    smc_params
        SMC-level HPs (target_ess, num_mcmc_steps, ...).  ``None`` uses
        the SMCMethod's default HP midpoints.
    inner_params_init
        Initial inner-kernel params ``{"step_size": float,
        "inverse_mass_matrix": array}`` before tempering starts.  ``None``
        → use defaults based on model dimensionality.
    parameter_update_strategy
        W6 registry key for the ``mcmc_parameter_update_fn``.
        Default ``"step_size_and_imm_from_particles"`` (HMC recommended).
    parameter_update_strategy_kwargs
        Extra kwargs for ``build_parameter_update_fn``.  Default:
        ``{"target_acceptance": 0.65}`` for HMC strategies.
    seed
        Master JAX random seed (default 20260517).
    catalog_root
        Catalog root for saving the recipe.
    outcomes_file
        Append FAIL / ERROR notes here.
    verbose
        Print progress messages.

    Returns
    -------
    SMCCellResult
        Verdict + gate metrics + path to saved recipe (on PASS).
    """
    _log(f"\n=== {model_name} x {smc_method_name} x {inner_method_name} ===")
    t_start = time.perf_counter()

    posterior = MODELS[model_name]
    smc_entry = SMC_METHODS[smc_method_name]
    inner_entry = BASE_METHODS[inner_method_name]

    # Compatibility check.
    if inner_method_name not in smc_entry.compatible_inner_methods:
        note = (
            f"SKIP: inner {inner_method_name!r} not in "
            f"{smc_method_name}.compatible_inner_methods"
        )
        _log(f"  {note}", verbose)
        return SMCCellResult(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            verdict="ERROR",
            note=note,
        )

    # Requires x64 for some models (e.g. gp_regression).
    if posterior.requires_x64 and not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    master_key = jax.random.key(seed)
    logfns_key, init_key, run_key = jax.random.split(master_key, 3)

    # --- W1: build logprior_fn + loglikelihood_fn ---
    _log("  Building SMC logfns (logprior + loglikelihood)...", verbose)
    try:
        init_position, logprior_fn, loglikelihood_fn, _postprocess_fn = (
            build_smc_logfns(logfns_key, posterior)
        )
    except Exception as exc:
        note = f"ERROR: build_smc_logfns failed: {type(exc).__name__}: {exc}"
        _log(f"  {note}", verbose)
        _append_outcome(
            model_name, smc_method_name, inner_method_name, note, outcomes_file
        )
        return SMCCellResult(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            verdict="ERROR",
            note=note,
            wall_seconds=time.perf_counter() - t_start,
        )

    # --- W2: build prior_sample_fn ---
    prior_sample_fn = build_prior_sample_fn(posterior)

    # --- Resolve dimension from init_position ---
    leaves = jax.tree_util.tree_leaves(init_position)
    model_dim = int(sum(jnp.asarray(leaf).size for leaf in leaves))

    # --- Inner-kernel params init ---
    if inner_params_init is None:
        # Default: identity IMM, small step size.
        inner_params_init = {
            "step_size": jnp.full(num_particles, _DEFAULT_HMC_STEP_SIZE),
            "inverse_mass_matrix": jnp.ones((num_particles, model_dim)),
        }
    else:
        # Broadcast scalar/global params to per-particle shape if needed.
        if "step_size" in inner_params_init:
            ss = jnp.asarray(inner_params_init["step_size"])
            if ss.ndim == 0:
                inner_params_init = {
                    **inner_params_init,
                    "step_size": jnp.full(num_particles, float(ss)),
                }
        if "inverse_mass_matrix" in inner_params_init:
            imm = jnp.asarray(inner_params_init["inverse_mass_matrix"])
            if imm.ndim == 1:
                inner_params_init = {
                    **inner_params_init,
                    "inverse_mass_matrix": jnp.tile(imm, (num_particles, 1)),
                }

    # --- Parameter update fn (W6) ---
    _update_kwargs = parameter_update_strategy_kwargs or (
        {"target_acceptance": 0.65} if "step_size" in parameter_update_strategy else {}
    )
    mcmc_parameter_update_fn = build_parameter_update_fn(
        parameter_update_strategy, **_update_kwargs
    )

    # --- Initialise particles ---
    _log(f"  Initialising {num_particles} particles from prior...", verbose)
    try:
        # prior_sample_fn returns dict of per-site arrays (N, *shape).
        prior_particles = init_particles_from_prior(
            init_key,
            prior_sample_fn=prior_sample_fn,
            num_particles=num_particles,
        )
    except Exception as exc:
        note = f"ERROR: init_particles_from_prior failed: {type(exc).__name__}: {exc}"
        _log(f"  {note}", verbose)
        _append_outcome(
            model_name, smc_method_name, inner_method_name, note, outcomes_file
        )
        return SMCCellResult(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            verdict="ERROR",
            note=note,
            wall_seconds=time.perf_counter() - t_start,
        )

    # --- Resolve SMC-level params ---
    from tuningfork.calibration.tune import default_value_for_space  # noqa: PLC0415

    resolved_smc_params: dict[str, Any] = {
        space.name: default_value_for_space(space)
        for space in smc_entry.default_hp_space
    }
    if smc_params:
        resolved_smc_params.update(smc_params)

    # Derive num_mcmc_steps for the inner kernel.
    num_mcmc_steps = int(resolved_smc_params.get("num_mcmc_steps", 10))
    target_ess = float(resolved_smc_params.get("target_ess", 0.5))

    # --- Build inner kernel (HMC/RWM/…) ---
    # The inner kernel is used by the SMC method to mutate particles.
    # For inner_kernel_tuning, it also extracts .step and .init.
    # We build a representative kernel (with identity IMM, default step_size)
    # for the factory; the actual per-particle params come from inner_params_init.
    from tuningfork.calibration.tune import default_params_for  # noqa: PLC0415

    inner_defaults = default_params_for(inner_entry)
    _imm_default = jnp.ones(model_dim)

    # For HMC/mhmc/rmhmc: needs step_size + inverse_mass_matrix + num_integration_steps.
    factory_kwargs: dict[str, Any] = dict(inner_defaults)
    if inner_entry.needs_mass_matrix:
        factory_kwargs["inverse_mass_matrix"] = _imm_default
    inner_kernel = inner_entry.factory(logprior_fn, **factory_kwargs)
    # Note: for SMC the logdensity_fn passed to the inner kernel at EACH step
    # is the tempered log density built by the SMC layer (not logprior_fn).
    # We build with logprior_fn here just to get the right .step / .init types.

    # --- Build mcmc_parameters dict (per-particle params, JAX arrays only) ---
    mcmc_parameters = dict(inner_params_init)  # step_size + inverse_mass_matrix

    # Add num_integration_steps for HMC-family inner kernels.
    if "num_integration_steps" in inner_defaults:
        nis = inner_defaults.get("num_integration_steps", _DEFAULT_HMC_NUM_STEPS)
        mcmc_parameters["num_integration_steps"] = jnp.full(num_particles, int(nis))

    # --- Build SMC algorithm ---
    _log(
        f"  Building {smc_method_name}({inner_method_name}, "
        f"N={num_particles}, target_ess={target_ess:.2f}, "
        f"num_mcmc_steps={num_mcmc_steps})...",
        verbose,
    )
    try:
        # For inner_kernel_tuning: needs extra kwargs (smc_algorithm, update_fn).
        extra_factory_kwargs: dict[str, Any] = {}
        if smc_method_name == "inner_kernel_tuning":
            import blackjax  # noqa: PLC0415

            extra_factory_kwargs["smc_algorithm"] = blackjax.adaptive_tempered_smc
            extra_factory_kwargs["mcmc_parameter_update_fn"] = mcmc_parameter_update_fn
            extra_factory_kwargs["initial_parameter_value"] = mcmc_parameters
            extra_factory_kwargs["target_ess"] = target_ess

        smc_alg = smc_entry.factory(
            logprior_fn,
            loglikelihood_fn,
            inner_kernel=inner_kernel,
            mcmc_parameters=mcmc_parameters,
            num_mcmc_steps=num_mcmc_steps,
            **(
                {
                    k: v
                    for k, v in resolved_smc_params.items()
                    if k not in ("num_mcmc_steps",) and k not in extra_factory_kwargs
                }
            ),
            **extra_factory_kwargs,
        )
        smc_init_state = smc_alg.init(prior_particles)
    except Exception as exc:
        note = f"ERROR: SMC algorithm build failed: {type(exc).__name__}: {exc}"
        _log(f"  {note}", verbose)
        _append_outcome(
            model_name, smc_method_name, inner_method_name, note, outcomes_file
        )
        return SMCCellResult(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            verdict="ERROR",
            note=note,
            wall_seconds=time.perf_counter() - t_start,
        )

    # --- Run SMC ---
    _log(f"  Running SMC (max_steps={max_steps})...", verbose)
    t_run0 = time.perf_counter()
    try:
        final_state, history = run_smc(
            run_key,
            smc_init_state=smc_init_state,
            smc_step_fn=smc_alg.step,
            max_steps=max_steps,
            lambda_target=1.0,
        )
        jax.block_until_ready(final_state)
    except Exception as exc:
        note = f"FAIL SMC run error: {type(exc).__name__}: {exc}"
        _log(f"  {note}", verbose)
        _append_outcome(
            model_name, smc_method_name, inner_method_name, note, outcomes_file
        )
        return SMCCellResult(
            model_name=model_name,
            smc_method_name=smc_method_name,
            inner_method_name=inner_method_name,
            verdict="FAIL",
            note=note,
            wall_seconds=time.perf_counter() - t_start,
        )
    t_run = time.perf_counter() - t_run0
    n_smc_steps = int(len(history["lmbda"])) if len(history["lmbda"]) > 0 else max_steps
    _log(f"  SMC done in {t_run:.1f}s ({n_smc_steps} tempering steps).", verbose)

    # --- Extract particles + weights from final state ---
    # inner_kernel_tuning: particles at state.sampler_state.particles
    # plain adaptive_tempered_smc: state.particles
    try:
        particles_out = final_state.sampler_state.particles
        weights_out = final_state.sampler_state.weights
        final_inner_params_raw = final_state.parameter_override
    except AttributeError:
        try:
            particles_out = final_state.particles
            weights_out = final_state.weights
            final_inner_params_raw = None
        except AttributeError as exc:
            note = f"FAIL: could not extract particles from state: {exc}"
            _log(f"  {note}", verbose)
            _append_outcome(
                model_name, smc_method_name, inner_method_name, note, outcomes_file
            )
            return SMCCellResult(
                model_name=model_name,
                smc_method_name=smc_method_name,
                inner_method_name=inner_method_name,
                verdict="FAIL",
                note=note,
                wall_seconds=time.perf_counter() - t_start,
            )

    particles_np = (
        {k: np.array(v) for k, v in particles_out.items()}
        if isinstance(particles_out, dict)
        else np.array(particles_out)
    )
    weights_np = np.array(weights_out)

    # --- Load GT summary for z-gate ---
    gt_summary_path = catalog_root / model_name / "reference" / "summary.json"
    _gt_summary: dict | None = None
    if gt_summary_path.exists():
        import json  # noqa: PLC0415

        raw = json.loads(gt_summary_path.read_text())
        _gt_summary = {}
        for site in raw.get("mean", {}):
            _gt_summary[site] = {
                "mean": raw["mean"][site],
                "std": raw["std"][site],
            }

    # --- W4: smc_gate ---
    _log("  Running smc_gate...", verbose)
    gate = smc_gate(
        particles_np,
        weights_np,
        _gt_summary,
        model_name=model_name,
        num_particles=num_particles,
    )
    _log(
        (
            f"  Gate: {gate.verdict}, particle_ess={gate.particle_ess:.1f}, "
            f"z={gate.max_abs_mean_z:.3f}"
            if gate.max_abs_mean_z is not None
            else f"  Gate: {gate.verdict}, particle_ess={gate.particle_ess:.1f}"
        ),
        verbose,
    )

    # --- Compute headline metric ---
    # total_grad_evals for HMC: N_particles × n_smc_steps × num_mcmc_steps × num_int_steps
    # total_likelihood_evals for RWM: N_particles × n_smc_steps × num_mcmc_steps
    headline: float | None = None
    _num_integration_steps = (
        int(inner_defaults.get("num_integration_steps", 1))
        if "num_integration_steps" in inner_defaults
        else 1
    )
    is_gradient_free = inner_entry.grad_count_per_step is not None and (
        inner_entry.grad_count_convention.startswith("0")
    )
    if is_gradient_free:
        total_cost = num_particles * n_smc_steps * num_mcmc_steps
    else:
        # HMC-family: each mcmc step = num_integration_steps gradient evals.
        total_cost = (
            num_particles * n_smc_steps * num_mcmc_steps * _num_integration_steps
        )
    if gate.particle_ess is not None and total_cost > 0:
        headline = gate.particle_ess / total_cost

    _log(
        f"  headline={headline:.5f}" if headline is not None else "  headline=None",
        verbose,
    )

    # --- Build final inner_params_final for recipe ---
    inner_params_final: dict[str, Any] | None = None
    if final_inner_params_raw is not None:
        try:
            inner_params_final = {
                k: v[0].tolist() if hasattr(v, "tolist") else float(v)
                for k, v in final_inner_params_raw.items()
                if k in ("step_size", "inverse_mass_matrix")
            }
        except Exception:  # noqa: BLE001
            pass  # non-fatal; recipe saved without final params

    # --- Save recipe on PASS ---
    t_total = time.perf_counter() - t_start
    gate_evidence = {
        "auto": gate.to_dict(),
        "override": {"reason": "", "statistician_id": "", "decision": ""},
    }

    recipe = SMCRecipe(
        model_name=model_name,
        smc_method_name=smc_method_name,
        inner_method_name=inner_method_name,
        num_particles=num_particles,
        max_steps=max_steps,
        smc_params=resolved_smc_params,
        inner_params_init={
            k: (v[0].tolist() if hasattr(v[0], "tolist") else float(v[0]))
            for k, v in inner_params_init.items()
        },
        inner_params_final=inner_params_final,
        parameter_update_strategy=parameter_update_strategy,
        parameter_update_strategy_kwargs=_update_kwargs,
        headline_metric=headline,
        gate_evidence=gate_evidence,
        calibration_budget={
            "n_particles": num_particles,
            "n_smc_steps": n_smc_steps,
            "num_mcmc_steps": num_mcmc_steps,
            "wall_seconds_total": round(t_total, 3),
            "wall_seconds_run": round(t_run, 3),
        },
    )

    recipe_path: Path | None = None
    if gate.verdict == "PASS":
        _log("  Building recipe (PASS)...", verbose)
        recipe_path = recipe.save(catalog_root)
        _log(f"  Saved: {recipe_path}", verbose)
    else:
        _log(f"  => gate {gate.verdict} — not saving recipe.", verbose)
        _append_outcome(
            model_name,
            smc_method_name,
            inner_method_name,
            f"FAIL z={gate.max_abs_mean_z:.3f} ess={gate.particle_ess:.1f}",
            outcomes_file,
        )

    return SMCCellResult(
        model_name=model_name,
        smc_method_name=smc_method_name,
        inner_method_name=inner_method_name,
        verdict=gate.verdict,
        recipe_path=recipe_path,
        gate_verdict=gate,
        particle_ess=gate.particle_ess,
        max_abs_mean_z=gate.max_abs_mean_z,
        headline_metric=headline,
        wall_seconds=t_total,
        note=(
            f"{gate.verdict} z={gate.max_abs_mean_z:.3f} ess={gate.particle_ess:.1f}"
            if gate.max_abs_mean_z is not None
            else gate.verdict
        ),
    )
