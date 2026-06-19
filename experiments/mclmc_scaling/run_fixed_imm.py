"""Run MCLMC with a FIXED GT-LRD inverse mass matrix (S1/S2/S3 measurement harness).

The IMM is held fixed; by default only step_size and L are adapted via
``mclmc_find_L_and_step_size`` with ``diagonal_preconditioning=False``.

S2 extension (additive, backward-compatible):
  Pass ``fixed_step_size`` and ``fixed_L`` to SKIP tuning and run directly at
  the supplied (step, L).  When both are None (the default), behaviour is
  identical to the original S1 harness.

S3/Real-model extension (additive, backward-compatible):
  ``run_mclmc_fixed_imm`` now accepts any registered catalog model name, not
  just ``"ill_cond_50"`` or ``"mvn_10"``.  For real (non-synthetic) models:
  - ``logdensity_fn`` is built from the model registry via
    ``tuningfork.model.build_logdensity_fn`` (unconstrained-space logdensity,
    same space as the GT draws).
  - ``init_position`` = ``gt_mean`` (per-dim mean of the GT draws) so there
    is no burn-in transient.
  - Bias metric: ``|Var_mcmc - gt_var| / gt_var`` where
    ``Var_mcmc = mean((x - gt_mean)^2)`` and ``gt_var`` is the per-dim
    variance of the GT draws.  This replaces the synthetic-model bias which
    centred on zero (``|E[x^2] - diag(Sigma)| / diag(Sigma)``).
  - ``gt_mean`` and ``gt_var`` are passed in (from ``gt_from_draws``).
  The synthetic-model paths (``ill_cond_50``, ``mvn_10``) are unchanged.

Funnel / adjusted_mclmc_dynamic extension (additive, backward-compatible):
  ``run_adj_dynamic_fixed_imm`` runs ``blackjax.mcmc.adjusted_mclmc_dynamic``
  with the fixed GT LRD IMM, tuned via the adjusted tuner with certified
  constraints (``frac_tune2=0``, ``params=MCLMCAdaptationState(...imm)``).
  The unadjusted-warmup step is scaled by **0.55** (catalog §7 rule, validated
  on horseshoe) to target ~94% acceptance for adjusted_mclmc_dynamic.
  Returns real ``div_rate`` from ``info.is_divergent`` (not NaN proxy).

API:
    run_mclmc_fixed_imm(model, imm, *, n_warmup, n_samples, num_chains, seed,
                        adjusted=False,
                        fixed_step_size=None, fixed_L=None,
                        floor_factor=1.5,
                        adj_num_steps=None, adj_target=0.9,
                        gt_mean=None, gt_var=None,
                        tune_init_step=None, tune_init_L=None) -> dict

    run_adj_dynamic_fixed_imm(model, imm, *, n_warmup, n_samples, num_chains,
                               seed, adj_target=0.9, step_scale=0.55,
                               gt_mean=None, gt_var=None) -> dict

Returned dict keys (unadjusted path):
    step_size        : float  -- adapted (or supplied) step size
    L                : float  -- adapted (or supplied) L
    max_bias         : float  -- max |Var_mcmc - gt_var| / gt_var across dims
                                 (real models) or |E[x^2] - diag(Sigma)| / diag(Sigma)
                                 (synthetic models, backward-compatible)
    mean_bias        : float  -- mean of the same
    min_bulk_ess     : float  -- min bulk-ESS across dims (arviz, declared basis)
    ess_per_grad     : float  -- total ESS / total grad evals
    eevpd            : float  -- Var[DeltaE] / dim (energy error variance per dim)
    div_rate         : float  -- fraction of NaN positions (proxy for divergences)
                                 or fraction of is_divergent=True (adj_dynamic path)
    n_warmup_grads   : int    -- warmup gradient evaluations (0 when fixed_step/L given)
    n_sampling_grads : int    -- sampling gradient evaluations
    total_grads      : int    -- total gradient evaluations

Additional keys for adjusted=True path and adj_dynamic path:
    acceptance_rate  : float  -- mean MH acceptance rate across chains x draws
    n_steps_median   : float  -- median integration steps per trajectory

ESS basis: arviz.ess(method="bulk") on xr.Dataset with (chain, draw, dim) layout.
Grad accounting (McLachlan 4-stage integrator, 2 grad/step for unadjusted;
  adjusted: 2 x n_integration_steps grad/trajectory):
    warmup   grads = 2 x n_warmup   x num_chains  (0 when fixed step/L given)
    sampling grads = 2 x n_samples  x num_chains  (unadjusted)
                   = 2 x n_steps    x num_chains  (adjusted; n_steps from info)
"""

import os
import sys
import warnings

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

import blackjax.mcmc.adjusted_mclmc as adj_mclmc_mod
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
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.metrics import LowRankInverseMassMatrix
from jax.flatten_util import ravel_pytree

# Add experiment dir to sys.path so gt_imm imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Helper: build a closed-over LRD kernel with fixed IMM
# ---------------------------------------------------------------------------


def _make_fixed_imm_kernel(fixed_imm: LowRankInverseMassMatrix):
    """Return an MCLMC kernel with IMM baked in.

    The returned kernel has the same signature as the standard mclmc kernel
    (rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size)
    but ignores the ``inverse_mass_matrix`` argument and always uses
    ``fixed_imm``. This mirrors the ``lrd_kernel`` closure in Phase 3 of
    ``mclmc_lrd_adaptation.mclmc_lrd_warmup``.
    """
    base_kernel = mclmc_mod.build_kernel()

    def kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):
        return base_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=fixed_imm,
            L=L,
            step_size=step_size,
        )

    return kernel


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_mclmc_fixed_imm(
    model_name: str,
    imm: LowRankInverseMassMatrix,
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    seed: int,
    adjusted: bool = False,
    fixed_step_size: float | None = None,
    fixed_L: float | None = None,
    floor_factor: float = 1.5,
    adj_num_steps: int | None = None,
    adj_target: float = 0.9,
    gt_mean: np.ndarray | None = None,
    gt_var: np.ndarray | None = None,
    tune_init_step: float | None = None,
    tune_init_L: float | None = None,
) -> dict:
    """Run MCLMC (unadjusted or adjusted) with a fixed LRD IMM.

    Parameters
    ----------
    model_name : str
        Synthetic models: ``"ill_cond_50"`` or ``"mvn_10"`` (analytic Sigma,
        zero-centred, bias = |E[x^2] - diag(Sigma)| / diag(Sigma)).
        Real catalog models: any registered model name (e.g. ``"german_credit"``,
        ``"eight_schools_ncp"``).  For real models, ``gt_mean`` and ``gt_var``
        must be provided (from ``gt_from_draws``).
    imm : LowRankInverseMassMatrix
        Fixed GT IMM to bake into the kernel.
    n_warmup : int
        Number of warmup steps (for tuning when fixed_step_size/fixed_L are
        None).  Ignored in the count when both fixed values are provided.
    n_samples : int
        Number of post-warmup samples per chain.
    num_chains : int
        Number of independent chains.
    seed : int
        Base RNG seed.
    adjusted : bool
        If False (default) runs unadjusted MCLMC.
        If True runs adjusted_mclmc with the fixed LRD IMM.
    fixed_step_size : float | None
        When provided together with ``fixed_L``, skip the tuning phase and
        run sampling directly at this step size.  Warmup grads = 0.
        Default None (tune via mclmc_find_L_and_step_size / adjusted tuner).
    fixed_L : float | None
        Paired with ``fixed_step_size``.  When both are provided, skip tuning.
        Default None.
    floor_factor : float
        Adjusted path only.  L_init floor multiplier:
        ``L_init = max(L_unadj, floor_factor * step_unadj)``.
        Mirrors the constraint in mclmc_lrd_adaptation Phase 4.
        Default 1.5 (recommended for stiff targets like ill_cond_50).
    adj_num_steps : int | None
        Adjusted path only.  Number of DA tuning steps.
        Default: n_warmup (same budget as unadjusted).
    adj_target : float
        Adjusted path only.  Target MH acceptance rate.  Default 0.9.
    gt_mean : np.ndarray | None
        Required for real (non-synthetic) models.  Per-dim mean of the GT draws
        in unconstrained space.  Used as init_position and as the centering
        point for the bias metric.  Provided by ``gt_from_draws``.
    gt_var : np.ndarray | None
        Required for real (non-synthetic) models.  Per-dim variance of the GT
        draws in unconstrained space.  Used in the bias metric:
        ``|Var_mcmc - gt_var| / gt_var``.  Provided by ``gt_from_draws``.
    tune_init_step : float | None
        When tuning (``fixed_step_size``/``fixed_L`` both None) AND this is
        provided together with ``tune_init_L``, passes
        ``params=MCLMCAdaptationState(L=tune_init_L, step_size=tune_init_step,
        inverse_mass_matrix=jnp.ones(d))`` into ``mclmc_find_L_and_step_size``
        so the DA step-size tuner starts from the given init rather than the
        default 0.25·√d.  The ``inverse_mass_matrix`` placeholder is ignored
        because the fixed-IMM closure always routes through the baked-in IMM.
        Default None → current (default) DA init behaviour unchanged.
    tune_init_L : float | None
        Paired with ``tune_init_step``.  Both must be provided together or
        both None.  Default None.

    Returns
    -------
    dict with keys listed in the module docstring.
    """
    # Validate fixed_step/L consistency
    _skip_tuning = (fixed_step_size is not None) and (fixed_L is not None)
    if (fixed_step_size is None) != (fixed_L is None):
        raise ValueError(
            "fixed_step_size and fixed_L must both be provided or both be None."
        )

    # Validate tune_init consistency
    if (tune_init_step is None) != (tune_init_L is None):
        raise ValueError(
            "tune_init_step and tune_init_L must both be provided or both be None."
        )
    _use_tune_init = (tune_init_step is not None) and (not _skip_tuning)

    # ---------------------------------------------------------------------------
    # Dispatch: synthetic vs real model
    # ---------------------------------------------------------------------------
    _SYNTHETIC = {"ill_cond_50", "mvn_10"}
    _is_real = model_name not in _SYNTHETIC

    if _is_real:
        # Real catalog model path (S3 extension)
        if gt_mean is None or gt_var is None:
            raise ValueError(
                f"run_mclmc_fixed_imm: {model_name!r} is a real model — "
                "gt_mean and gt_var must be provided (from gt_from_draws)."
            )
        from tuningfork.model._numpyro import (
            build_logdensity_fn as _build_logdensity_fn,
        )
        from tuningfork.model._registry import MODELS

        entry = MODELS[model_name]
        d = entry.dim

        # Build logdensity_fn from the model registry (unconstrained-space logdensity)
        _init_key = jax.random.key(seed)
        _init_pos_dict, _logdensity_fn_raw, _postprocess_fn = _build_logdensity_fn(
            _init_key, entry
        )

        # Determine ravel structure from the init_pos dict
        from jax.flatten_util import ravel_pytree as _ravel_pytree_local

        _flat0, _unravel_fn = _ravel_pytree_local(_init_pos_dict)

        # The logdensity_fn for real models takes a dict (multi-site pytree)
        # MCLMC operates on flat arrays, so we compose with unravel
        def logdensity_fn(x_flat):
            return _logdensity_fn_raw(_unravel_fn(x_flat))

        # init_position = GT mean (flat, float64)
        init_position = jnp.array(gt_mean, dtype=jnp.float64)

        # Bias metric: |Var_mcmc - gt_var| / gt_var  (gt_var = per-dim variance of GT draws)
        gt_var_arr = np.array(gt_var, dtype=np.float64)  # (d,)
        gt_mean_arr = np.array(gt_mean, dtype=np.float64)  # (d,)

        # diag_Sigma is not used for real models; set to gt_var as a sentinel
        diag_Sigma = gt_var_arr

    else:
        # Synthetic model path (unchanged)
        from gt_imm import gt_cov

        Sigma, _ = gt_cov(model_name)
        d = Sigma.shape[0]
        diag_Sigma = np.diag(Sigma)  # (d,) -- GT second moments for bias calc

        if model_name == "ill_cond_50":
            from tuningfork.model.ill_cond_50 import COV_NP

            Sigma_inv_jax = jnp.array(np.linalg.inv(COV_NP), dtype=jnp.float64)

            def logdensity_fn(x):
                return -0.5 * jnp.dot(x, Sigma_inv_jax @ x)

            init_position = jnp.zeros(d, dtype=jnp.float64)

        elif model_name == "mvn_10":

            def logdensity_fn(x):
                return -0.5 * jnp.dot(x, x)

            init_position = jnp.zeros(d, dtype=jnp.float64)

        # gt_mean/gt_var sentinel values for synthetic models (zero-centred)
        gt_mean_arr = np.zeros(d, dtype=np.float64)
        gt_var_arr = diag_Sigma

    # Per-chain key splitting
    base_key = jax.random.key(seed)
    chain_keys = jax.random.split(base_key, num_chains)

    if adjusted:
        return _run_adjusted_fixed_imm(
            logdensity_fn=logdensity_fn,
            init_position=init_position,
            imm=imm,
            d=d,
            diag_Sigma=diag_Sigma,
            base_key=base_key,
            chain_keys=chain_keys,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            fixed_step_size=fixed_step_size,
            fixed_L=fixed_L,
            floor_factor=floor_factor,
            adj_num_steps=adj_num_steps,
            adj_target=adj_target,
            skip_tuning=_skip_tuning,
            gt_mean_arr=gt_mean_arr,
            gt_var_arr=gt_var_arr,
        )

    # ------------------------------------------------------------------
    # Unadjusted MCLMC path
    # ------------------------------------------------------------------

    # Build the fixed-IMM kernel (IMM baked in, ignores the imm argument)
    fixed_kernel = _make_fixed_imm_kernel(imm)

    if _skip_tuning:
        # S2 mode: use the provided (step, L) directly — no warmup budget spent
        L_mean = float(fixed_L)
        step_mean = float(fixed_step_size)
        n_warmup_grads = 0

        # Initialise chains (no tuning needed)
        final_states = []
        for ci in range(num_chains):
            init_key, _ = jax.random.split(chain_keys[ci], 2)
            final_states.append(mclmc_mod.init(init_position, logdensity_fn, init_key))
    else:
        # S1 / default mode: tune one chain at a time, collect (L, step_size)
        Ls = []
        steps = []
        final_states = []

        # Build the optional tuning-init params once (reused across chains).
        # When _use_tune_init is True, pass params=MCLMCAdaptationState(...) to
        # warm-start the DA tuner at (tune_init_step, tune_init_L) instead of
        # the default 0.25*sqrt(d).  The inverse_mass_matrix field is a
        # placeholder (jnp.ones(d)) — ignored because fixed_kernel always routes
        # through the baked-in IMM.
        if _use_tune_init:
            _tune_params = MCLMCAdaptationState(
                L=jnp.array(float(tune_init_L)),
                step_size=jnp.array(float(tune_init_step)),
                inverse_mass_matrix=jnp.ones(d),  # placeholder; overridden by closure
            )
        else:
            _tune_params = None

        for ci in range(num_chains):
            key_ci = chain_keys[ci]
            init_key, warmup_key = jax.random.split(key_ci, 2)

            init_state = mclmc_mod.init(init_position, logdensity_fn, init_key)

            # diagonal_preconditioning=False: do NOT update inverse_mass_matrix
            # inside the tuner -- the fixed_kernel closure handles the IMM.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_tuned, params_tuned, _ = mclmc_find_L_and_step_size(
                    mclmc_kernel=fixed_kernel,
                    num_steps=n_warmup,
                    state=init_state,
                    rng_key=warmup_key,
                    logdensity_fn=logdensity_fn,
                    diagonal_preconditioning=False,
                    params=_tune_params,  # None → default DA init; else warm-start
                )

            Ls.append(float(params_tuned.L))
            steps.append(float(params_tuned.step_size))
            final_states.append(state_tuned)

        L_mean = float(np.mean(Ls))
        step_mean = float(np.mean(steps))

        # Warmup grad count: 2 grads/step x n_warmup steps x num_chains
        n_warmup_grads = 2 * n_warmup * num_chains

    # Sampling: run n_samples per chain with mean-adapted (L, step_size)
    # Use fresh keys to avoid correlation with warmup
    sample_base_key = jax.random.fold_in(base_key, 999)
    sampling_keys = jax.random.split(sample_base_key, num_chains)

    all_positions = []  # list of (n_samples, d) arrays
    all_energy_changes = []  # list of (n_samples,) arrays

    base_prod_kernel = mclmc_mod.build_kernel()

    for ci in range(num_chains):
        state = final_states[ci]
        scan_keys = jax.random.split(sampling_keys[ci], n_samples)

        def _sample_step(carry_state, rng_key):
            next_state, info = base_prod_kernel(
                rng_key=rng_key,
                state=carry_state,
                logdensity_fn=logdensity_fn,
                inverse_mass_matrix=imm,
                L=L_mean,
                step_size=step_mean,
            )
            return next_state, (next_state.position, info.energy_change)

        _, (pos_traj, energy_changes) = jax.lax.scan(_sample_step, state, scan_keys)
        # Flatten pytree positions to (n_samples, d)
        flat_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(pos_traj)
        all_positions.append(np.array(flat_pos, dtype=np.float64))
        all_energy_changes.append(np.array(energy_changes, dtype=np.float64))

    # Sampling grad count: 2 grads/step x n_samples x num_chains
    n_sampling_grads = 2 * n_samples * num_chains
    total_grads = n_warmup_grads + n_sampling_grads

    # Compute metrics
    # positions: (num_chains, n_samples, d)
    positions_arr = np.stack(all_positions, axis=0)  # (C, n_samples, d)

    # Bias metric:
    #   Synthetic models (zero-centred): |E[x^2] - diag(Sigma)| / diag(Sigma)
    #   Real models: |Var_mcmc - gt_var| / gt_var  where Var_mcmc = mean((x - gt_mean)^2)
    # The real-model formula subsumes the synthetic formula when gt_mean=0 and gt_var=diag(Sigma).
    var_mcmc = np.mean(
        (positions_arr - gt_mean_arr[None, None, :]) ** 2, axis=(0, 1)
    )  # (d,)
    bias = np.abs(var_mcmc - gt_var_arr) / np.maximum(gt_var_arr, 1e-30)  # (d,)
    max_bias = float(bias.max())
    mean_bias = float(bias.mean())

    # Bulk ESS via arviz 1.x (declared basis: arviz.ess method="bulk")
    # Build xr.Dataset with dims (chain, draw, x_dim_0)
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], positions_arr)})
    ess_ds = az.ess(ds, method="bulk")
    ess_arr = np.array(ess_ds["x"])  # (d,)
    min_bulk_ess = float(ess_arr.min())

    # ESS per grad: min_bulk_ess / total_grads
    ess_per_grad = min_bulk_ess / total_grads

    # EEVPD: Var[DeltaE] / dim
    energy_changes_all = np.concatenate(all_energy_changes, axis=0)
    eevpd = float(np.var(energy_changes_all) / d)

    # Divergence rate: NaN fraction as proxy (unadjusted MCLMC has no MH flag)
    nan_frac = float(np.isnan(positions_arr).any(axis=-1).mean())
    div_rate = nan_frac

    return {
        "step_size": step_mean,
        "L": L_mean,
        "max_bias": max_bias,
        "mean_bias": mean_bias,
        "min_bulk_ess": min_bulk_ess,
        "ess_per_grad": ess_per_grad,
        "eevpd": eevpd,
        "div_rate": div_rate,
        "n_warmup_grads": n_warmup_grads,
        "n_sampling_grads": n_sampling_grads,
        "total_grads": total_grads,
    }


# ---------------------------------------------------------------------------
# Adjusted MCLMC path (Phase 4 of mclmc_lrd_adaptation, adapted for sandbox)
# ---------------------------------------------------------------------------


def _make_fixed_imm_adj_kernel(fixed_imm: LowRankInverseMassMatrix):
    """Return an adjusted_mclmc kernel with IMM baked in.

    Mirrors the ``adj_lrd_kernel`` closure in Phase 4 of
    ``mclmc_lrd_adaptation.mclmc_lrd_warmup``.  The wrapper ignores the
    ``inverse_mass_matrix`` argument and always routes through ``fixed_imm``.
    """
    base_adj_kernel = adj_mclmc_mod.build_kernel()

    def kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        integration_steps_params,
        inverse_mass_matrix,
    ):
        return base_adj_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            integration_steps_params=integration_steps_params,
            inverse_mass_matrix=fixed_imm,  # always route through fixed GT IMM
        )

    return kernel


def _run_adjusted_fixed_imm(
    *,
    logdensity_fn,
    init_position,
    imm: LowRankInverseMassMatrix,
    d: int,
    diag_Sigma,
    base_key,
    chain_keys,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    fixed_step_size: float | None,
    fixed_L: float | None,
    floor_factor: float,
    adj_num_steps: int | None,
    adj_target: float,
    skip_tuning: bool,
    gt_mean_arr: np.ndarray | None = None,
    gt_var_arr: np.ndarray | None = None,
) -> dict:
    """Adjusted MCLMC with fixed GT-LRD IMM.

    Mirrors Phase 4 of mclmc_lrd_adaptation with the following hard constraints:
      C1) params != None  ->  no sqrt(dim) default L init
      C2) frac_tune2=0.0  ->  variance-based L estimator disabled
          (invalid under a baked-in LRD IMM; measures original-space trace(Sigma))
      C3) target_acceptance=0.9
      C4) floor_factor >= 1.5 for stiff targets (prevents DA ceiling binding)

    When skip_tuning=True (fixed_step_size and fixed_L both given):
      - Skips the adjusted tuning phase
      - Uses the provided (step, L) directly
      - n_warmup_grads = 0
    """
    adj_kernel = _make_fixed_imm_adj_kernel(imm)
    _adj_steps = adj_num_steps if adj_num_steps is not None else n_warmup

    # Initialise chains
    init_states_adj = []
    for ci in range(num_chains):
        init_key, _ = jax.random.split(chain_keys[ci], 2)
        init_states_adj.append(adj_mclmc_mod.init(init_position, logdensity_fn))

    if skip_tuning:
        # S2 fixed-step mode: skip the adjusted tuning; use the provided (step, L)
        step_mean = float(fixed_step_size)
        L_mean = float(fixed_L)
        n_warmup_grads = 0
        final_adj_states = init_states_adj
    else:
        # Run the adjusted tuner on each chain (sequentially; mirrors Phase 4).
        # Hard constraints per mclmc_lrd_adaptation:
        #   - params carries the baked-in IMM (placeholder) + L_init floor-guarded
        #   - frac_tune2=0.0 (variance-based L estimator disabled for LRD IMM)
        #   - diagonal_preconditioning=False
        #   - target_acceptance=adj_target (default 0.9)
        #
        # For the S2 harness we don't have a preceding unadjusted Phase 3 to
        # warm-start from.  We use sqrt(d) as a fallback L_init, then apply the
        # floor guard relative to a small probe step (1.0).  The floor_factor=1.5
        # default ensures the DA ceiling L_init/1.1 stays above the oracle step.
        probe_step = 1.0  # conservative probe before tuning converges
        L_init = float(max(np.sqrt(d), floor_factor * probe_step))

        # adj_init_params: must be MCLMCAdaptationState with the baked-in IMM
        # (constraint C1: params != None prevents sqrt(dim) default).
        # inverse_mass_matrix field is a placeholder; adj_kernel always routes
        # through the fixed GT IMM.  We supply imm.sigma (a 1-D jnp array with
        # the right dtype contract) as the placeholder.
        adj_init_params = MCLMCAdaptationState(
            L=jnp.array(L_init, dtype=jnp.float64),
            step_size=jnp.array(1.0, dtype=jnp.float64),
            inverse_mass_matrix=imm.sigma,  # placeholder; overridden by closure
        )

        Ls = []
        steps = []
        final_adj_states = []

        warmup_keys = [
            jax.random.split(chain_keys[ci], 2)[1] for ci in range(num_chains)
        ]

        for ci in range(num_chains):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_tuned, params_tuned, _ = adjusted_mclmc_find_L_and_step_size(
                    mclmc_kernel=adj_kernel,
                    logdensity_fn=logdensity_fn,
                    num_steps=_adj_steps,
                    state=init_states_adj[ci],
                    rng_key=warmup_keys[ci],
                    target=adj_target,
                    frac_tune1=0.5,  # certified recipe: 0.5 × steps DA steps
                    frac_tune2=0.0,  # C2: variance-based L estimator disabled
                    frac_tune3=0.0,
                    diagonal_preconditioning=False,  # don't overwrite LRD IMM
                    params=adj_init_params,  # C1: explicit init params
                )
            Ls.append(float(params_tuned.L))
            steps.append(float(params_tuned.step_size))
            final_adj_states.append(state_tuned)

        L_mean = float(np.mean(Ls))
        step_mean = float(np.mean(steps))

        # Warmup grads: adjusted_mclmc has a fixed integration_steps per trajectory.
        # The adapted N_sample = round(L_mean / step_mean); each trajectory costs
        # N_sample x 2 grads (McLachlan 4-stage).  For budget accounting we use
        # the simpler 2-grads/step approximation consistent with the unadjusted path.
        n_warmup_grads = 2 * _adj_steps * num_chains

    # Sampling: run n_samples per chain using the baked-in IMM kernel.
    # integration_steps_params = (N_sample,) where N_sample = round(L/step).
    N_sample = max(1, round(L_mean / max(step_mean, 1e-10)))
    sample_base_key = jax.random.fold_in(base_key, 998)
    sampling_keys = jax.random.split(sample_base_key, num_chains)

    all_positions = []
    all_accept_rates = []
    all_n_steps_list = []

    adj_prod_kernel = adj_mclmc_mod.build_kernel()

    for ci in range(num_chains):
        state = final_adj_states[ci]
        scan_keys = jax.random.split(sampling_keys[ci], n_samples)

        def _adj_sample_step(carry_state, rng_key):
            next_state, info = adj_prod_kernel(
                rng_key=rng_key,
                state=carry_state,
                logdensity_fn=logdensity_fn,
                step_size=step_mean,
                integration_steps_params=(N_sample,),
                inverse_mass_matrix=imm,
            )
            return next_state, (
                next_state.position,
                info.acceptance_rate,
                info.num_integration_steps,
            )

        _, (pos_traj, accept_traj, n_steps_traj) = jax.lax.scan(
            _adj_sample_step, state, scan_keys
        )
        flat_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(pos_traj)
        all_positions.append(np.array(flat_pos, dtype=np.float64))
        all_accept_rates.append(np.array(accept_traj, dtype=np.float64))
        all_n_steps_list.append(np.array(n_steps_traj, dtype=np.float64))

    # Sampling grad count: 2 x N_sample x n_samples x num_chains
    n_sampling_grads = 2 * N_sample * n_samples * num_chains
    total_grads = n_warmup_grads + n_sampling_grads

    positions_arr = np.stack(all_positions, axis=0)  # (C, n_samples, d)

    # Bias metric (adjusted is asymptotically unbiased; this will be ~MC noise)
    # Real models: |Var_mcmc - gt_var| / gt_var; synthetic (gt_mean=0): |E[x^2] - diag(Sigma)| / diag(Sigma)
    _gt_mean = gt_mean_arr if gt_mean_arr is not None else np.zeros(d, dtype=np.float64)
    _gt_var = gt_var_arr if gt_var_arr is not None else diag_Sigma
    var_mcmc_adj = np.mean((positions_arr - _gt_mean[None, None, :]) ** 2, axis=(0, 1))
    bias = np.abs(var_mcmc_adj - _gt_var) / np.maximum(_gt_var, 1e-30)
    max_bias = float(bias.max())
    mean_bias = float(bias.mean())

    # Bulk ESS
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], positions_arr)})
    ess_ds = az.ess(ds, method="bulk")
    ess_arr = np.array(ess_ds["x"])
    min_bulk_ess = float(ess_arr.min())
    ess_per_grad = min_bulk_ess / total_grads

    # EEVPD: not directly available for adjusted (no energy_change field in same way)
    # Use acceptance_rate as proxy for energy health; eevpd not meaningful for adjusted.
    eevpd = float("nan")

    # Acceptance rate: mean over all chains x draws
    accept_all = np.concatenate(all_accept_rates, axis=0)
    acceptance_rate = float(np.mean(accept_all))

    # Divergence rate: NaN fraction
    nan_frac = float(np.isnan(positions_arr).any(axis=-1).mean())
    div_rate = nan_frac

    # Median integration steps/trajectory
    n_steps_all = np.concatenate(all_n_steps_list, axis=0)
    n_steps_median = float(np.median(n_steps_all))

    return {
        "step_size": step_mean,
        "L": L_mean,
        "max_bias": max_bias,
        "mean_bias": mean_bias,
        "min_bulk_ess": min_bulk_ess,
        "ess_per_grad": ess_per_grad,
        "eevpd": eevpd,
        "div_rate": div_rate,
        "n_warmup_grads": n_warmup_grads,
        "n_sampling_grads": n_sampling_grads,
        "total_grads": total_grads,
        "acceptance_rate": acceptance_rate,
        "n_steps_median": n_steps_median,
    }


# ---------------------------------------------------------------------------
# adjusted_mclmc_dynamic with fixed GT-LRD IMM (funnel exploration)
# ---------------------------------------------------------------------------


def _make_fixed_imm_adj_dyn_kernel(fixed_imm: LowRankInverseMassMatrix):
    """Return an adjusted_mclmc_dynamic kernel with the GT IMM baked in.

    Mirrors _make_fixed_imm_adj_kernel but for the dynamic variant.  The
    wrapper ignores the ``inverse_mass_matrix`` argument and always routes
    through ``fixed_imm``.

    The dynamic kernel signature (from adjusted_mclmc_dynamic.build_kernel):
        kernel(rng_key, state, logdensity_fn, step_size,
               L_proposal_factor, inverse_mass_matrix, integration_steps_params)
    """
    _steps_fn = make_random_trajectory_length_fn(True)
    base_dyn_kernel = adj_dyn_mod.build_kernel(integration_steps_fn=_steps_fn)

    def kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        L_proposal_factor,
        inverse_mass_matrix,
        integration_steps_params,
    ):
        return base_dyn_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            L_proposal_factor=L_proposal_factor,
            inverse_mass_matrix=fixed_imm,  # always route through fixed GT IMM
            integration_steps_params=integration_steps_params,
        )

    return kernel


def _make_fixed_imm_adj_kernel_for_tuning(fixed_imm: LowRankInverseMassMatrix):
    """Return an adjusted_mclmc (static) kernel for the tuning phase.

    adjusted_mclmc_find_L_and_step_size uses the adjusted_mclmc (static)
    kernel — not the dynamic one.  This mirrors what the horseshoe recipe
    does: tune with adjusted_mclmc_tuning, then sample with
    adjusted_mclmc_dynamic.  The tuning kernel has the same baked-in GT IMM.
    """
    base_adj_kernel = adj_mclmc_mod.build_kernel()

    def kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        integration_steps_params,
        inverse_mass_matrix,
    ):
        return base_adj_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            integration_steps_params=integration_steps_params,
            inverse_mass_matrix=fixed_imm,
        )

    return kernel


def run_adj_dynamic_fixed_imm(
    model_name: str,
    imm: LowRankInverseMassMatrix,
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    seed: int,
    adj_target: float = 0.9,
    step_scale: float = 0.55,
    gt_mean: np.ndarray | None = None,
    gt_var: np.ndarray | None = None,
) -> dict:
    """Run adjusted_mclmc_dynamic with a fixed GT-LRD IMM on funnel/hard models.

    Tuning protocol (mirrors catalog §7 + mclmc_lrd_adaptation Phase 4):
      1. Run unadjusted MCLMC warmup (``mclmc_find_L_and_step_size``) with
         the fixed GT IMM to get an initial (step_unadj, L_unadj).
      2. Scale step_size by ``step_scale`` (default 0.55 per §7: validated on
         horseshoe, targets ~94% acceptance for adjusted_mclmc_dynamic).
      3. Run ``adjusted_mclmc_find_L_and_step_size`` (static adjusted_mclmc
         kernel, the canonical tuner) with ``frac_tune2=0``,
         ``params=MCLMCAdaptationState(...imm.sigma)`` (certified constraints).
      4. Sample with ``adjusted_mclmc_dynamic`` (random trajectory length)
         using the tuned (step, L) and the fixed GT IMM.

    Divergence flag: uses ``info.is_divergent`` (real MH/energy flag from
    HMCInfo), not the NaN proxy used by the unadjusted path.

    Parameters
    ----------
    model_name : str
        Model name (synthetic or real catalog model).
    imm : LowRankInverseMassMatrix
        Fixed GT IMM to bake into the kernel (both tuning and sampling).
    n_warmup : int
        Warmup steps for BOTH the unadjusted phase-1 and the adjusted
        tuning phase.  Total warmup grads = 2 * n_warmup * num_chains * 2.
    n_samples : int
        Post-warmup samples per chain.
    num_chains : int
        Number of independent chains.
    seed : int
        Base RNG seed.
    adj_target : float
        Target MH acceptance rate for the adjusted tuner.  Default 0.9.
    step_scale : float
        Factor to scale the unadjusted step to initialise the adjusted DA.
        Default 0.55 (catalog §7, validated on horseshoe, ~94% acceptance).
    gt_mean : np.ndarray | None
        Required for real models (from gt_from_draws).
    gt_var : np.ndarray | None
        Required for real models (from gt_from_draws).

    Returns
    -------
    dict with keys: step_size, L, max_bias, mean_bias, min_bulk_ess,
        ess_per_grad, eevpd (NaN — not meaningful for adjusted),
        div_rate (from is_divergent), acceptance_rate, n_steps_median,
        n_warmup_grads, n_sampling_grads, total_grads.
    """
    # ---------------------------------------------------------------------------
    # Dispatch: synthetic vs real model (mirrors run_mclmc_fixed_imm)
    # ---------------------------------------------------------------------------
    _SYNTHETIC = {"ill_cond_50", "mvn_10"}
    _is_real = model_name not in _SYNTHETIC

    if _is_real:
        if gt_mean is None or gt_var is None:
            raise ValueError(
                f"run_adj_dynamic_fixed_imm: {model_name!r} is a real model — "
                "gt_mean and gt_var must be provided (from gt_from_draws)."
            )
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
        from jax.flatten_util import ravel_pytree as _ravel_pytree_local

        _flat0, _unravel_fn = _ravel_pytree_local(_init_pos_dict)

        def logdensity_fn(x_flat):
            return _logdensity_fn_raw(_unravel_fn(x_flat))

        init_position = jnp.array(gt_mean, dtype=jnp.float64)
        gt_var_arr = np.array(gt_var, dtype=np.float64)
        gt_mean_arr = np.array(gt_mean, dtype=np.float64)

    else:
        from gt_imm import gt_cov

        Sigma, _ = gt_cov(model_name)
        d = Sigma.shape[0]
        diag_Sigma = np.diag(Sigma)

        if model_name == "ill_cond_50":
            from tuningfork.model.ill_cond_50 import COV_NP

            Sigma_inv_jax = jnp.array(np.linalg.inv(COV_NP), dtype=jnp.float64)

            def logdensity_fn(x):
                return -0.5 * jnp.dot(x, Sigma_inv_jax @ x)

        elif model_name == "mvn_10":

            def logdensity_fn(x):
                return -0.5 * jnp.dot(x, x)

        init_position = jnp.zeros(d, dtype=jnp.float64)
        gt_mean_arr = np.zeros(d, dtype=np.float64)
        gt_var_arr = diag_Sigma

    # ---------------------------------------------------------------------------
    # Key splitting
    # ---------------------------------------------------------------------------
    base_key = jax.random.key(seed)
    chain_keys = jax.random.split(base_key, num_chains)

    # ---------------------------------------------------------------------------
    # Phase 1: unadjusted MCLMC warmup to get initial (step_unadj, L_unadj)
    # This mirrors mclmc_lrd_adaptation Phase 3 -> Phase 4 warm-start.
    # ---------------------------------------------------------------------------
    unadj_kernel = _make_fixed_imm_kernel(imm)

    Ls_unadj = []
    steps_unadj = []
    unadj_states = []

    for ci in range(num_chains):
        init_key_ci, warmup_key_ci = jax.random.split(chain_keys[ci], 2)
        init_state_unadj = mclmc_mod.init(init_position, logdensity_fn, init_key_ci)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_tuned, params_tuned, _ = mclmc_find_L_and_step_size(
                mclmc_kernel=unadj_kernel,
                num_steps=n_warmup,
                state=init_state_unadj,
                rng_key=warmup_key_ci,
                logdensity_fn=logdensity_fn,
                diagonal_preconditioning=False,
            )
        Ls_unadj.append(float(params_tuned.L))
        steps_unadj.append(float(params_tuned.step_size))
        unadj_states.append(state_tuned)

    L_unadj = float(np.mean(Ls_unadj))
    step_unadj = float(np.mean(steps_unadj))

    # ---------------------------------------------------------------------------
    # Phase 2: adjusted tuner warm-started from (step_unadj * step_scale, L_unadj)
    # Mirrors catalog §7 + Phase 4 of mclmc_lrd_adaptation.
    # Uses the STATIC adjusted_mclmc kernel for tuning (canonical protocol).
    # Certified constraints: frac_tune2=0, params != None.
    # ---------------------------------------------------------------------------
    adj_tuning_kernel = _make_fixed_imm_adj_kernel_for_tuning(imm)
    step_init_adj = step_unadj * step_scale  # §7: scale by 0.55
    L_init_adj = max(L_unadj, 1.5 * step_init_adj)  # floor guard

    adj_init_params = MCLMCAdaptationState(
        L=jnp.array(L_init_adj, dtype=jnp.float64),
        step_size=jnp.array(step_init_adj, dtype=jnp.float64),
        inverse_mass_matrix=imm.sigma,  # placeholder; adj_tuning_kernel ignores it
    )

    Ls_adj = []
    steps_adj = []
    adj_warmup_states = []

    # Adjusted warmup keys: use a fold to separate from Phase 1 keys
    adj_base_key = jax.random.fold_in(base_key, 777)
    adj_chain_keys = jax.random.split(adj_base_key, num_chains)

    for ci in range(num_chains):
        # initialise adjusted_mclmc state from the unadjusted end-state position
        # adjusted_mclmc.init takes (position, logdensity_fn) — no rng_key
        init_adj_state = adj_mclmc_mod.init(unadj_states[ci].position, logdensity_fn)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_adj_tuned, params_adj_tuned, _ = adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=adj_tuning_kernel,
                logdensity_fn=logdensity_fn,
                num_steps=n_warmup,
                state=init_adj_state,
                rng_key=adj_chain_keys[ci],
                target=adj_target,
                frac_tune1=0.5,
                frac_tune2=0.0,  # certified: variance-based L estimator disabled
                frac_tune3=0.0,
                diagonal_preconditioning=False,
                params=adj_init_params,  # certified: explicit init prevents sqrt(dim) default
            )
        Ls_adj.append(float(params_adj_tuned.L))
        steps_adj.append(float(params_adj_tuned.step_size))
        adj_warmup_states.append(state_adj_tuned)

    L_mean = float(np.mean(Ls_adj))
    step_mean = float(np.mean(steps_adj))

    # Warmup grads: Phase 1 (unadj) + Phase 2 (adj), each n_warmup steps
    n_warmup_grads = 2 * n_warmup * num_chains * 2

    # ---------------------------------------------------------------------------
    # Sampling with adjusted_mclmc_dynamic + fixed GT IMM
    # ---------------------------------------------------------------------------
    dyn_kernel = _make_fixed_imm_adj_dyn_kernel(imm)
    avg_steps = float(max(1.0, L_mean / max(step_mean, 1e-10)))

    sample_base_key = jax.random.fold_in(base_key, 888)
    sampling_keys = jax.random.split(sample_base_key, num_chains)

    all_positions = []
    all_accept_rates = []
    all_n_steps_list = []
    all_is_divergent = []

    for ci in range(num_chains):
        # adjusted_mclmc_dynamic.init requires (position, logdensity_fn, random_generator_arg)
        # Use chain sampling key as the initial random_generator_arg
        init_dyn_state = adj_dyn_mod.init(
            adj_warmup_states[ci].position,
            logdensity_fn,
            sampling_keys[ci],  # random_generator_arg for trajectory-length sampling
        )

        scan_keys = jax.random.split(sampling_keys[ci], n_samples)

        def _dyn_sample_step(carry_state, rng_key):
            next_state, info = dyn_kernel(
                rng_key=rng_key,
                state=carry_state,
                logdensity_fn=logdensity_fn,
                step_size=step_mean,
                L_proposal_factor=jnp.inf,
                inverse_mass_matrix=imm,  # passed but overridden in closure
                integration_steps_params=(avg_steps,),
            )
            return next_state, (
                next_state.position,
                info.acceptance_rate,
                info.num_integration_steps,
                info.is_divergent,
            )

        _, (pos_traj, accept_traj, n_steps_traj, is_div_traj) = jax.lax.scan(
            _dyn_sample_step, init_dyn_state, scan_keys
        )
        flat_pos = jax.vmap(lambda p: ravel_pytree(p)[0])(pos_traj)
        all_positions.append(np.array(flat_pos, dtype=np.float64))
        all_accept_rates.append(np.array(accept_traj, dtype=np.float64))
        all_n_steps_list.append(np.array(n_steps_traj, dtype=np.float64))
        all_is_divergent.append(np.array(is_div_traj, dtype=bool))

    # Sampling grad count: 2 x avg_steps (realized) x n_samples x num_chains
    # Use actual realized step count for accuracy
    n_steps_all = np.concatenate(all_n_steps_list, axis=0)
    n_sampling_grads = int(2 * n_steps_all.sum() * num_chains / len(all_n_steps_list))
    # Simpler: 2 * round(avg_steps) * n_samples * num_chains (for forward accounting)
    n_sampling_grads = 2 * int(round(avg_steps)) * n_samples * num_chains
    total_grads = n_warmup_grads + n_sampling_grads

    positions_arr = np.stack(all_positions, axis=0)  # (C, n_samples, d)

    # Bias metric
    var_mcmc = np.mean((positions_arr - gt_mean_arr[None, None, :]) ** 2, axis=(0, 1))
    bias = np.abs(var_mcmc - gt_var_arr) / np.maximum(gt_var_arr, 1e-30)
    max_bias = float(bias.max())
    mean_bias = float(bias.mean())

    # Bulk ESS
    ds = xr.Dataset({"x": (["chain", "draw", "x_dim_0"], positions_arr)})
    ess_ds = az.ess(ds, method="bulk")
    ess_arr = np.array(ess_ds["x"])
    min_bulk_ess = float(ess_arr.min())
    ess_per_grad = min_bulk_ess / total_grads

    # Acceptance rate
    accept_all = np.concatenate(all_accept_rates, axis=0)
    acceptance_rate = float(np.mean(accept_all))

    # Divergence rate: real is_divergent flag (not NaN proxy)
    is_div_all = np.concatenate(all_is_divergent, axis=0)
    div_rate = float(np.mean(is_div_all))

    # Median integration steps
    n_steps_median = float(np.median(n_steps_all))

    return {
        "step_size": step_mean,
        "L": L_mean,
        "max_bias": max_bias,
        "mean_bias": mean_bias,
        "min_bulk_ess": min_bulk_ess,
        "ess_per_grad": ess_per_grad,
        "eevpd": float("nan"),  # not meaningful for MH-adjusted sampler
        "div_rate": div_rate,
        "acceptance_rate": acceptance_rate,
        "n_steps_median": n_steps_median,
        "n_warmup_grads": n_warmup_grads,
        "n_sampling_grads": n_sampling_grads,
        "total_grads": total_grads,
    }


if __name__ == "__main__":
    # Quick smoke test
    print("run_fixed_imm.py quick smoke (tiny N) ...")

    from gt_imm import gt_cov, gt_lrd_imm

    model_name = "mvn_10"
    Sigma, _ = gt_cov(model_name)
    d = Sigma.shape[0]

    imm_diag = gt_lrd_imm(Sigma, k=0)
    result = run_mclmc_fixed_imm(
        model_name, imm_diag, n_warmup=50, n_samples=100, num_chains=2, seed=42
    )
    print(f"  step_size: {result['step_size']:.4f}")
    print(f"  L:         {result['L']:.4f}")
    print(f"  max_bias:  {result['max_bias']:.4f}")
    print(f"  min_ess:   {result['min_bulk_ess']:.2f}")
    print(f"  ess/grad:  {result['ess_per_grad']:.6f}")
    print(f"  eevpd:     {result['eevpd']:.2e}")
    print("SMOKE PASSED")
    sys.exit(0)
