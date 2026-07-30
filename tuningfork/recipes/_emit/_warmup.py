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
"""Descriptor-driven Python emit-functions for warmup sections.

Replaces the 8 ``.tmpl`` files in
``_templates/warmups/{no_warmup,window_adaptation,window_adaptation_multichain,
pathfinder,multipathfinder,multipathfinder_window_adaptation,
vi_warmup,laplace_multiphase_warmup}.py.tmpl``
(~376 LOC total) with a single Python entry point.

All routing is resolved at generation time (P1 straight-line principle).
D8 compliant: emitted strings contain no ``import tuningfork``.

Warmup groupings
----------------
- **no_warmup**: sets ``_adapted_params = {}`` and the single-chain init flag.
- **WA pair** (``window_adaptation``, ``window_adaptation_multichain``):
  single-chain vs multichain variant, controlled by ``_multichain`` flag.
- **Pathfinder pair** (``pathfinder``, ``multipathfinder``):
  single-path vs multi-path variant, controlled by ``_multi`` flag.
- **multipathfinder_window_adaptation**: composition warmup (stage 1 MPF + stage 2 WA).
- **vi_warmup**: VI-based IMM + init-positions + adapted step_size; unified
  meanfield/fullrank variant (resolved via ctx keys populated by _emit_script.py).
- **laplace_multiphase_warmup**: multi-phase laplace warmup (2 WA phases
  with different LaplaceMarginal factories).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tuningfork.base_method._base import BaseMethod

__all__ = ["emit_warmup"]


def emit_warmup(warmup_name: str, base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit the warmup section for a recipe reproduction script.

    Replaces the per-warmup ``.py.tmpl`` template files with a single
    descriptor-driven Python function.

    Parameters
    ----------
    warmup_name : str
        Warmup identifier string (e.g. ``"window_adaptation_diag_imm"``).
        Used for outer dispatch only — not a family proxy.
    base_method : BaseMethod
        Registry entry for the sampler.  Descriptors consumed:
        - ``base_method.name`` — sampler identifier.
        - ``base_method.extra_required_kwargs`` — drives laplace detection.
    ctx : dict
        Substitution context from ``emit_script()``.  Keys consumed depend
        on the warmup family; see per-warmup helpers below.

    Returns
    -------
    str
        Python source for the warmup block (D8 compliant — no tuningfork
        inference imports).
    """
    if warmup_name == "no_warmup":
        body = _emit_no_warmup()
    elif warmup_name in (
        "window_adaptation_diag_imm",
        "window_adaptation_dense_imm",
        "window_adaptation_low_rank_imm",
    ):
        # multichain flag is in ctx — determined by _emit_script.py before calling here.
        _multichain = ctx.get("_warmup_is_multichain", False)
        body = _emit_window_adaptation(warmup_name, ctx, multichain=_multichain)
    elif warmup_name == "pathfinder":
        body = _emit_pathfinder(ctx, multi=False)
    elif warmup_name == "multipathfinder":
        body = _emit_pathfinder(ctx, multi=True)
    elif warmup_name == "multipathfinder_window_adaptation":
        body = _emit_multipathfinder_window_adaptation(ctx)
    elif warmup_name in ("meanfield_vi", "fullrank_vi"):
        body = _emit_vi_warmup(ctx)
    elif warmup_name == "laplace_multiphase_warmup":
        body = _emit_laplace_multiphase_warmup(ctx)
    elif warmup_name == "mclmc_tuning":
        body = _emit_mclmc_tuning(ctx)
    elif warmup_name == "mclmc_lrd_tuning":
        body = _emit_mclmc_lrd_tuning(ctx)
    else:
        raise ValueError(
            f"emit_warmup: unsupported warmup_name {warmup_name!r}. "
            "If this is a new warmup, add it to _warmup.py."
        )
    if not body.endswith("\n"):
        body += "\n"
    return body


# ---------------------------------------------------------------------------
# no_warmup
# ---------------------------------------------------------------------------


def _emit_no_warmup() -> str:
    """Emit the no_warmup section.

    Sets _adapted_params = {} and _warmup_init_is_single_chain = True.
    The sampler template provides _state_post_warmup via its NameError fallback
    (preserved for no_warmup path; stripped for all other warmups by _emit_script.py).
    """
    lines: list[str] = []
    a = lines.append
    a(
        "# === WARMUP: no_warmup (no adaptation; sampler params taken from recipe base_method_params) ==="
    )
    a(
        "# _adapted_params is empty; each sampler template falls back to $bm_* recipe defaults."
    )
    a(
        "# _state_post_warmup will be bound in the sampler template after the kernel is constructed."
    )
    a("_adapted_params = {}")
    a("# Signal to the inference loop that _state_post_warmup is a single-chain")
    a(
        "# state (set by the sampler template's NameError fallback) and needs broadcasting"
    )
    a("# to (num_chains, ...) before the vmap-scan loop.")
    a("_warmup_init_is_single_chain = True")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Window adaptation (single-chain + multichain variants)
# ---------------------------------------------------------------------------


def _emit_window_adaptation(
    warmup_name: str, ctx: dict[str, Any], *, multichain: bool
) -> str:
    """Emit single-chain or multichain window_adaptation warmup.

    Covers diag_imm, dense_imm, and low_rank_imm variants.
    The distinction between variants comes from ``window_adaptation_fn`` and
    ``window_adaptation_extra_kwargs`` already resolved in ctx by _emit_script.py.

    Parameters
    ----------
    warmup_name : str
        One of ``window_adaptation_{diag,dense,low_rank}_imm``.
    ctx : dict
        Substitution context.  Required keys:
        - ``warmup_name``: echoed in the comment header.
        - ``target_acceptance_rate``: echoed in the comment header.
        - ``n_warmup``: number of warmup steps.
        - ``tuning_seed``: RNG seed.
        - ``warmup_algorithm``: ``blackjax.nuts`` / ``blackjax.mhmc`` etc.
        - ``warmup_extra_kwargs``: extra kwargs string (", k=v" form or "").
        - ``window_adaptation_fn``: ``blackjax.window_adaptation`` or
          ``blackjax.window_adaptation_low_rank``.
        - ``window_adaptation_extra_kwargs``: extra kwargs before target
          (e.g. ``"is_mass_matrix_diagonal=False,"`` or ``"max_rank=N,"``).
        - ``num_chains``: number of chains (used for multichain path).
        - ``warmup_progress_bar``: bool. Wraps the warmup run call in
          ``with blackjax.progress_bar():``.
    multichain : bool
        If True, emit multichain vmap path.  If False, emit single-chain path
        (selected only via the independent ``warmup_num_chains=[1]`` knob —
        avoids vmap-of-while_loop for expensive-logprob models).
    """
    lines: list[str] = []
    a = lines.append

    warmup_algorithm = ctx["warmup_algorithm"]
    target_acceptance_rate = ctx["target_acceptance_rate"]
    n_warmup = ctx["n_warmup"]
    tuning_seed = ctx["tuning_seed"]
    warmup_extra_kwargs = ctx.get("warmup_extra_kwargs", "")
    wa_fn = ctx["window_adaptation_fn"]
    wa_extra = ctx["window_adaptation_extra_kwargs"]
    warmup_progress_bar = ctx["warmup_progress_bar"]

    if multichain:
        a(
            f"# === WARMUP: {warmup_name} multi-chain (target_acceptance_rate={target_acceptance_rate}, n_warmup={n_warmup}) ==="
        )
        a(
            "# Multi-chain warmup: jax.vmap over window_adaptation.run so each chain gets its"
        )
        a("# own adapted (step_size, inverse_mass_matrix).")
        a("# This matches the runner's per-chain warmup behavior.")
        a(f"_warmup = {wa_fn}(")
        a(f"    {warmup_algorithm},")
        a("    logdensity_fn,")
        if wa_extra:
            a(f"    {wa_extra}")
        a(f"    target_acceptance_rate={target_acceptance_rate}{warmup_extra_kwargs},")
        a(")")
        a(f"_warmup_keys = jax.random.split(jax.random.key({tuning_seed}), num_chains)")
        if ctx.get("init_position_is_prebatched", False):
            a("# Initial positions are already batched at generation time.")
            a("_init_positions = init_position")
        else:
            a("# Replicate init_position to (num_chains, ...) for vmap.")
            a("_init_positions = jax.tree.map(")
            a(
                "    lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape), init_position"
            )
            a(")")
        a("")
        a("")
        a("@jax.vmap")
        a("def _run_one_warmup(k, x0):")
        a(f"    (state, params), _ = _warmup.run(k, x0, {n_warmup})")
        a("    return state, params")
        a("")
        a("")
        if warmup_progress_bar:
            a('with blackjax.progress_bar(label="warmup"):')
            a(
                "    _batched_states, _adapted_params = _run_one_warmup("
                "_warmup_keys, _init_positions)"
            )
        else:
            a(
                "_batched_states, _adapted_params = _run_one_warmup(_warmup_keys, _init_positions)"
            )
        a("# _warmup_is_perchain=True: adapted params have a leading num_chains axis.")
        a("_warmup_is_perchain = True")
        a("_state_post_warmup = _batched_states")
    else:
        a(
            f"# === WARMUP: {warmup_name} (target_acceptance_rate={target_acceptance_rate}, n_warmup={n_warmup}) ==="
        )
        a("# Single-chain warmup (warmup_num_chains=[1]): run once and broadcast the")
        a("# state to (num_chains,) so that scan(vmap(kernel)) in the inference loop")
        a(
            "# maps over chains sharing the same adapted (step_size, inverse_mass_matrix)."
        )
        a(f"_warmup = {wa_fn}(")
        a(f"    {warmup_algorithm},")
        a("    logdensity_fn,")
        if wa_extra:
            a(f"    {wa_extra}")
        a(f"    target_acceptance_rate={target_acceptance_rate}{warmup_extra_kwargs},")
        a(")")
        a(f"_warmup_key = jax.random.fold_in(jax.random.key({tuning_seed}), 0)")
        if warmup_progress_bar:
            a('with blackjax.progress_bar(label="warmup"):')
            a(
                f"    (state, _adapted_params), _ = _warmup.run(_warmup_key, init_position, {n_warmup})"
            )
        else:
            a(
                f"(state, _adapted_params), _ = _warmup.run(_warmup_key, init_position, {n_warmup})"
            )
        a("# Broadcast state to (num_chains,) for scan(vmap(kernel)).")
        a("_state_post_warmup = jax.tree.map(")
        a("    lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),")
        a("    state,")
        a(")")
        a(
            "# _warmup_is_perchain=False: adapted params are scalar / un-batched (shared across chains)."
        )
        a("_warmup_is_perchain = False")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pathfinder + multipathfinder
# ---------------------------------------------------------------------------


def _emit_pathfinder(ctx: dict[str, Any], *, multi: bool) -> str:
    """Emit pathfinder or multipathfinder warmup.

    Parameters
    ----------
    ctx : dict
        Required keys: target_acceptance_rate, tuning_seed, n_warmup,
        num_chains. For multi=True also: wp_n_paths, wp_num_samples_per_path.
    multi : bool
        If True emit multipathfinder; if False emit single-path pathfinder.
    """
    lines: list[str] = []
    a = lines.append

    target_acceptance_rate = ctx["target_acceptance_rate"]
    tuning_seed = ctx["tuning_seed"]
    n_warmup = ctx["n_warmup"]

    a("import jax.numpy as jnp")
    a("")

    if multi:
        n_paths = ctx["wp_n_paths"]
        num_samples_per_path = ctx["wp_num_samples_per_path"]
        a(
            f"# === WARMUP: multipathfinder (n_paths={n_paths}, target_acceptance_rate={target_acceptance_rate}, n_warmup={n_warmup}) ==="
        )
        a("# Multi-path Pathfinder + dual-averaging step size adaptation.")
        a(
            "# Derives a shared dense (d,d) IMM from the PSIS-weighted L-BFGS mixture covariance."
        )
        a(
            "# pathfinder_adaptation returns: step_size scalar (num_chains=1) or (num_chains,),"
        )
        a(
            "# inverse_mass_matrix always (d,d) — shared across chains, no broadcast needed."
        )
        a(f"_n_paths = {n_paths}")
        a("")
        a("_mpf_adapt = blackjax.pathfinder_adaptation(")
        a("    blackjax.nuts,")
        a("    logdensity_fn,")
        a("    num_chains=num_chains,")
        a("    n_paths=_n_paths,")
        a(f"    num_samples_per_path={num_samples_per_path},")
        a('    imm_estimator="lbfgs_psis_mixture",')
        a("    initial_step_size=1.0,")
        a(f"    target_acceptance_rate={target_acceptance_rate},")
        a(")")
        a(
            f"_mpf_results, _ = _mpf_adapt.run(jax.random.key({tuning_seed}), init_position, {n_warmup})"
        )
        a("_state_post_warmup = _mpf_results.state")
        a("_adapted_params = {")
        a("    # jnp.mean: handles scalar (num_chains=1) and (num_chains,) uniformly.")
        a('    "step_size": jnp.mean(_mpf_results.parameters["step_size"]),  # scalar')
        a(
            '    "inverse_mass_matrix": _mpf_results.parameters["inverse_mass_matrix"],  # (d, d)'
        )
        a('    "_multipathfinder_psis_pareto_k": _mpf_results.parameters.get(')
        a('        "_pathfinder_psis_pareto_k"')
        a("    ),")
        a("}")
    else:
        a(
            f"# === WARMUP: pathfinder (single-path, target_acceptance_rate={target_acceptance_rate}, n_warmup={n_warmup}) ==="
        )
        a("# Single-path Pathfinder + dual-averaging step size adaptation.")
        a(
            "# Derives a dense (d,d) IMM from the L-BFGS inverse Hessian; shared across chains."
        )
        a(
            "# pathfinder_adaptation returns: step_size scalar (num_chains=1) or (num_chains,),"
        )
        a(
            "# inverse_mass_matrix always (d,d) — shared across chains, no broadcast needed."
        )
        a("_pf_adapt = blackjax.pathfinder_adaptation(")
        a("    blackjax.nuts,")
        a("    logdensity_fn,")
        a("    num_chains=num_chains,")
        a(
            "    n_paths=1,  # explicit 1 → PATH A (num_chains=1) or PATH B (num_chains>1)"
        )
        a("    initial_step_size=1.0,")
        a(f"    target_acceptance_rate={target_acceptance_rate},")
        a(")")
        a(
            f"_pf_results, _ = _pf_adapt.run(jax.random.key({tuning_seed}), init_position, {n_warmup})"
        )
        a("_state_post_warmup = _pf_results.state")
        a("_adapted_params = {")
        a("    # jnp.mean: handles scalar (num_chains=1) and (num_chains,) uniformly.")
        a('    "step_size": jnp.mean(_pf_results.parameters["step_size"]),  # scalar')
        a(
            '    "inverse_mass_matrix": _pf_results.parameters["inverse_mass_matrix"],  # (d, d)'
        )
        a("}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# multipathfinder_window_adaptation (composition warmup)
# ---------------------------------------------------------------------------


def _emit_multipathfinder_window_adaptation(ctx: dict[str, Any]) -> str:
    """Emit multipathfinder_window_adaptation warmup.

    Stage 1: multipathfinder (derive dense IMM via PSIS-weighted L-BFGS mixture covariance
    + resample init positions).
    Stage 2: single-chain window_adaptation seeded with multipathfinder IMM.

    Parameters
    ----------
    ctx : dict
        Required keys: target_acceptance_rate, tuning_seed, n_warmup, num_chains,
        wp_n_paths, wp_num_samples_per_path, wp_imm_shrinkage_to_previous,
        warmup_algorithm, warmup_extra_kwargs, warmup_progress_bar.
    """
    lines: list[str] = []
    a = lines.append

    target_acceptance_rate = ctx["target_acceptance_rate"]
    tuning_seed = ctx["tuning_seed"]
    n_warmup = ctx["n_warmup"]
    n_paths = ctx["wp_n_paths"]
    num_samples_per_path = ctx["wp_num_samples_per_path"]
    imm_shrinkage = ctx["wp_imm_shrinkage_to_previous"]
    warmup_algorithm = ctx["warmup_algorithm"]
    warmup_progress_bar = ctx["warmup_progress_bar"]

    a("# === WARMUP: multipathfinder_window_adaptation")
    a(f"#   n_paths={n_paths}, num_samples_per_path={num_samples_per_path},")
    a(f"#   imm_shrinkage_to_previous={imm_shrinkage},")
    a(f"#   target_acceptance_rate={target_acceptance_rate}, n_warmup={n_warmup} ===")
    a("")
    a("# Stage 1: multipathfinder — derive dense (d,d) IMM via PSIS-weighted")
    a(
        "# L-BFGS mixture covariance (law of total variance) and resample init positions."
    )
    a("import jax.numpy as jnp")
    a("from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1")
    a("from blackjax.vi.multipathfinder import psis_weights")
    a("from jax.flatten_util import ravel_pytree")
    a("")
    a(f"_n_paths = {n_paths}")
    a(f"_num_samples_per_path = {num_samples_per_path}")
    a("")
    a(
        f"_pf_key, _resample_key, _adapt_key = jax.random.split(jax.random.key({tuning_seed}), 3)"
    )
    a("")
    a("# Replicate init_position to (n_paths, ...) for the multi-path fit.")
    a("_init_positions_mpf = jax.tree.map(")
    a("    lambda x: jnp.broadcast_to(x[None], (_n_paths,) + x.shape), init_position")
    a(")")
    a("")
    a("_mpf = blackjax.multipathfinder(logdensity_fn)")
    a(
        "_mpf_state, _ = _mpf.init(_pf_key, _init_positions_mpf, num_samples=_num_samples_per_path)"
    )
    a("")
    a("_log_weights, _pareto_k = psis_weights(_mpf_state)")
    a("")
    a("# Compute PSIS-weighted mixture covariance (law of total variance).")
    a("# NOTE: we flatten positions via ravel_pytree because PathfinderState.position")
    a("# stores the pytree-structured form, not a flat (n_paths, d) array.")
    a("_n_paths_actual = _log_weights.shape[0] // _num_samples_per_path")
    a("_log_w_per_path = _log_weights.reshape(_n_paths_actual, _num_samples_per_path)")
    a("_log_w_path_norm = jax.scipy.special.logsumexp(_log_w_per_path, axis=1)")
    a("_log_w_path_norm -= jax.scipy.special.logsumexp(_log_w_path_norm)")
    a("_w = jnp.exp(_log_w_path_norm)  # (n_paths,)")
    a(
        "_mu_per_path = jax.vmap(lambda x: ravel_pytree(x)[0])(_mpf_state.path_states.position)  # (n_paths, d)"
    )
    a("_sigmas = jax.vmap(lbfgs_inverse_hessian_formula_1)(")
    a("    _mpf_state.path_states.alpha,")
    a("    _mpf_state.path_states.beta,")
    a("    _mpf_state.path_states.gamma,")
    a(")  # (n_paths, d, d)")
    a('_mu_mix = jnp.einsum("i,id->d", _w, _mu_per_path)')
    a('_sigma_within = jnp.einsum("i,ijk->jk", _w, _sigmas)')
    a("_delta = _mu_per_path - _mu_mix[None, :]")
    a('_sigma_between = jnp.einsum("i,ij,ik->jk", _w, _delta, _delta)')
    a("_imm_dense = _sigma_within + _sigma_between  # (d, d)")
    a("")
    a("# PSIS-resample num_chains init positions (used later to broadcast the state).")
    a("_samples_flat = jax.tree.map(")
    a("    lambda x: x.reshape(-1, *x.shape[2:]), _mpf_state.samples")
    a(")")
    a("_probs = jnp.exp(_log_weights)")
    a("_init_indices = jax.random.choice(")
    a(
        "    _resample_key, _log_weights.shape[0], shape=(num_chains,), replace=True, p=_probs"
    )
    a(")")
    a("_init_positions_psis = jax.tree.map(lambda x: x[_init_indices], _samples_flat)")
    a(
        "# _init_positions_psis: shape (num_chains, ...) — PSIS-resampled starting points."
    )
    a("# Use the first resampled position as the single-chain warmup starting point.")
    a("_init_position_pf = jax.tree.map(lambda x: x[0], _init_positions_psis)")
    a("")
    a("# Stage 2: single-chain window_adaptation seeded with multipathfinder IMM.")
    a(
        "# imm_shrinkage_to_previous keeps the multipathfinder IMM influential across windows."
    )
    a("_warmup = blackjax.window_adaptation(")
    a(f"    {warmup_algorithm},")
    a("    logdensity_fn,")
    a("    is_mass_matrix_diagonal=False,")
    a("    initial_inverse_mass_matrix=_imm_dense,")
    a(f"    imm_shrinkage_to_previous={imm_shrinkage},")
    a(f"    target_acceptance_rate={target_acceptance_rate},")
    a("    initial_step_size=1.0,")
    a(")")
    a("")
    a("_warmup_key = jax.random.fold_in(_adapt_key, 0)")
    if warmup_progress_bar:
        a('with blackjax.progress_bar(label="warmup"):')
        a(
            f"    (state, _adapted_params), _ = _warmup.run(_warmup_key, _init_position_pf, {n_warmup})"
        )
    else:
        a(
            f"(state, _adapted_params), _ = _warmup.run(_warmup_key, _init_position_pf, {n_warmup})"
        )
    a(
        "# Broadcast state to (num_chains,) using PSIS-resampled positions as chain start points."
    )
    a(
        "# Each chain starts at a different PSIS-resampled position but shares adapted params."
    )
    a("_state_post_warmup = jax.tree.map(")
    a("    lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),")
    a("    state,")
    a(")")
    a('# _adapted_params["step_size"]: scalar (shared across chains)')
    a('# _adapted_params["inverse_mass_matrix"]: (d, d)')
    a("# _pareto_k: scalar PSIS Pareto-k diagnostic")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VI warmup (meanfield_vi + fullrank_vi unified)
# ---------------------------------------------------------------------------


def _emit_vi_warmup(ctx: dict[str, Any]) -> str:
    """Emit VI warmup (unified meanfield/fullrank).

    This implements the two-phase VI warmup:
    Phase 1: VI optimisation (Adam steps), extracts IMM + draws init positions.
    Phase 2: incremental step_size dual-averaging (VI IMM frozen).

    Parameters
    ----------
    ctx : dict
        Required keys: warmup_name, wp_num_optimization_steps, tuning_seed,
        target_acceptance_rate, n_warmup, num_chains, warmup_algorithm,
        warmup_extra_kwargs, vi_prefix, vi_module, vi_imm_description,
        vi_imm_extraction_block, vi_adapted_imm_expr.
    """
    lines: list[str] = []
    a = lines.append

    warmup_name = ctx["warmup_name"]
    num_opt_steps = ctx["wp_num_optimization_steps"]
    tuning_seed = ctx["tuning_seed"]
    target_acceptance_rate = ctx["target_acceptance_rate"]
    n_warmup = ctx["n_warmup"]
    warmup_algorithm = ctx["warmup_algorithm"]
    warmup_extra_kwargs = ctx.get("warmup_extra_kwargs", "")
    vp = ctx["vi_prefix"]
    vi_module = ctx["vi_module"]
    vi_imm_extraction_block = ctx["vi_imm_extraction_block"]
    vi_adapted_imm_expr = ctx["vi_adapted_imm_expr"]

    a(
        f"# === WARMUP: {warmup_name} (VI-based IMM + init-positions + adapted step_size) ==="
    )
    a(f"# Phase 1: single VI optimisation ({num_opt_steps} Adam steps),")
    a(
        "#           shared across all chains.  Extracts IMM and draws num_chains init positions."
    )
    a(f"# Phase 2: step_size-only dual-averaging ({n_warmup} steps from chain-0")
    a("#           VI-drawn position, VI IMM FROZEN).  Incremental Nesterov DA:")
    a("#           step_size updates each step; MCMC runs at the current step_size.")
    a(
        "#           The frozen-IMM constraint is load-bearing: mass matrix is not touched"
    )
    a("#           after VI fit.")
    a("# Compatible: nuts, hmc, mala, rwm, barker.  NOT compatible with mclmc.")
    a(f"import optax as {vp}_optax")
    a(f"import {vi_module} as {vp}_vi")
    a(
        f"from blackjax.adaptation.step_size import dual_averaging_adaptation as {vp}_da_adapt"
    )
    a(f"from jax.flatten_util import ravel_pytree as {vp}_ravel")
    a("")
    a(f"{vp}_optimizer = {vp}_optax.adam(1e-2)")
    a(f"{vp}_num_opt_steps = {num_opt_steps}")
    a("")
    a(f"{vp}_flat_init, {vp}_unravel = {vp}_ravel(init_position)")
    a(f"_d = int({vp}_flat_init.shape[0])")
    a("")
    a("# ── Phase 1: VI optimisation ─────────────────────────────────────────────────")
    a(f"{vp}_vi_key, {vp}_sample_key, {vp}_sa_key = jax.random.split(")
    a(f"    jax.random.key({tuning_seed}), 3")
    a(")")
    a(f"{vp}_vi_init = {vp}_vi.init(init_position, {vp}_optimizer)")
    a("")
    a("")
    a(f"def {vp}_vi_one_step(carry, step_key):")
    a(f"    new_state, info = {vp}_vi.step(")
    a(f"        step_key, carry, logdensity_fn, {vp}_optimizer, 5")
    a("    )")
    a("    return new_state, info")
    a("")
    a("")
    a(f"{vp}_vi_keys = jax.random.split({vp}_vi_key, {vp}_num_opt_steps)")
    a(f"{vp}_final_vi_state, _ = jax.lax.scan(")
    a(f"    {vp}_vi_one_step, {vp}_vi_init, {vp}_vi_keys")
    a(")")
    a("")
    a(f"# Extract IMM from fitted VI state ({ctx['vi_imm_description']}).")
    # Inline the VI IMM extraction block (pre-resolved by _emit_script.py)
    a(vi_imm_extraction_block)
    a("")
    a("# Draw num_chains init positions from the fitted distribution.")
    a(f"{vp}_chain_keys = jax.random.split({vp}_sample_key, num_chains)")
    a("")
    a("")
    a("@jax.vmap")
    a(f"def {vp}_draw_one(key):")
    a(f"    samples = {vp}_vi.sample(key, {vp}_final_vi_state, num_samples=1)")
    a("    pos = jax.tree.map(lambda x: x[0], samples)")
    a(f"    flat_pos, _ = {vp}_ravel(pos)")
    a("    return flat_pos")
    a("")
    a("")
    a(f"{vp}_flat_positions = {vp}_draw_one({vp}_chain_keys)  # (num_chains, d)")
    a(f"{vp}_init_positions = jax.vmap({vp}_unravel)({vp}_flat_positions)")
    a("")
    a(
        "# ── Phase 2: incremental step_size dual-averaging (VI IMM frozen) ─────────────"
    )
    a(f"{vp}_da_init_fn, {vp}_da_update_fn, {vp}_da_final_fn = {vp}_da_adapt(")
    a(f"    target={target_acceptance_rate}")
    a(")")
    a(f"{vp}_da_s0 = {vp}_da_init_fn(1.0)")
    a("")
    a(f"{vp}_sa_init_state = {warmup_algorithm}(")
    a("    logdensity_fn,")
    a("    step_size=1.0,")
    a(f"    inverse_mass_matrix={vp}_imm{warmup_extra_kwargs},")
    a(f").init(jax.tree.map(lambda x: x[0], {vp}_init_positions))")
    a("")
    a("")
    a(f"def {vp}_sa_one_step(carry, step_key):")
    a('    """One step of frozen-IMM incremental step_size dual-averaging."""')
    a("    mcmc_state, da_state = carry")
    a("    current_ss = jnp.exp(da_state.log_step_size)")
    a(f"    new_mcmc_state, mcmc_info = {warmup_algorithm}(")
    a("        logdensity_fn,")
    a("        step_size=current_ss,")
    a(f"        inverse_mass_matrix={vp}_imm{warmup_extra_kwargs},")
    a("    ).step(step_key, mcmc_state)")
    a("    _accept = jnp.asarray(")
    a("        getattr(")
    a("            mcmc_info,")
    a('            "acceptance_rate",')
    a('            getattr(mcmc_info, "is_accepted", jnp.asarray(0.5)),')
    a("        )")
    a("    )")
    a(f"    new_da_state = {vp}_da_update_fn(da_state, jnp.mean(_accept))")
    a("    return (new_mcmc_state, new_da_state), None")
    a("")
    a("")
    a(f"{vp}_sa_keys = jax.random.split({vp}_sa_key, {n_warmup})")
    a(f"({vp}_sa_final_mcmc, {vp}_sa_final_da), _ = jax.lax.scan(")
    a(f"    {vp}_sa_one_step,")
    a(f"    ({vp}_sa_init_state, {vp}_da_s0),")
    a(f"    {vp}_sa_keys,")
    a(")")
    a(f"{vp}_adapted_step_size = {vp}_da_final_fn({vp}_sa_final_da)")
    a("")
    a(
        "# ── Build downstream kernel states (VI positions + adapted step_size + VI IMM) ─"
    )
    a("")
    a("")
    a("@jax.vmap")
    a(f"def {vp}_init_one(pos):")
    a(f"    return {warmup_algorithm}(")
    a("        logdensity_fn,")
    a(f"        step_size={vp}_adapted_step_size,")
    a(f"        inverse_mass_matrix={vp}_imm{warmup_extra_kwargs},")
    a("    ).init(pos)")
    a("")
    a("")
    a(f"_state_post_warmup = {vp}_init_one({vp}_init_positions)")
    a("")
    a("_adapted_params = {")
    a(f'    "step_size": jnp.full((num_chains,), {vp}_adapted_step_size),')
    a(f'    "inverse_mass_matrix": {vi_adapted_imm_expr},')
    a("}")
    a("_warmup_is_perchain = True")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Laplace multi-phase warmup
# ---------------------------------------------------------------------------


def _emit_laplace_multiphase_warmup(ctx: dict[str, Any]) -> str:
    """Emit laplace multi-phase warmup.

    Phase 1: window_adaptation with diagonal IMM (traversal).
    Phase 2: window_adaptation with dense IMM (Welford capture).

    Each phase uses a separate LaplaceMarginal factory (different maxiter).
    Single-chain warmup; final state broadcast to (num_chains,).

    NOTE: The laplace_preamble (emitted before this section) sets up:
      - When warmup_algorithm is blackjax.nuts: scalar adapter ``_warmup_logdensity_fn``
        wrapping ``_laplace_warmup[0]`` (nuts calls has_aux=False).
      - When warmup_algorithm is blackjax.laplace_hmc: ``logdensity_fn = _laplace_warmup[0]``
        directly (laplace_hmc.init calls has_aux=True, needs aux-returning marginal).
    This warmup section then overrides ``logdensity_fn`` for Phase 2 using the
    same conditional logic applied to ``_laplace_warmup[1]``.

    Parameters
    ----------
    ctx : dict
        Required keys: num_warmup_phases, tuning_seed, warmup_algorithm,
        warmup_extra_kwargs, num_chains, warmup_progress_bar,
        wp0_name, wp0_target, wp0_n_warmup, wp0_extra_kwargs,
        wp1_name, wp1_target, wp1_n_warmup, wp1_extra_kwargs, wp1_maxiter.
    """
    lines: list[str] = []
    a = lines.append

    num_phases = ctx["num_warmup_phases"]
    tuning_seed = ctx["tuning_seed"]
    warmup_algorithm = ctx["warmup_algorithm"]
    warmup_progress_bar = ctx["warmup_progress_bar"]

    # Phase 0 slots
    wp0_name = ctx["wp0_name"]
    wp0_target = ctx["wp0_target"]
    wp0_n_warmup = ctx["wp0_n_warmup"]
    wp0_extra_kwargs = ctx.get("wp0_extra_kwargs", "")

    # Phase 1 slots
    wp1_name = ctx["wp1_name"]
    wp1_target = ctx["wp1_target"]
    wp1_n_warmup = ctx["wp1_n_warmup"]
    wp1_extra_kwargs = ctx.get("wp1_extra_kwargs", "")
    wp1_maxiter = ctx.get("wp1_maxiter", 30)

    a(f"# === WARMUP: laplace multi-phase ({num_phases} phases) ===")
    a(
        f"# Phase 1: {wp0_name} (n_warmup={wp0_n_warmup}, target={wp0_target}{wp0_extra_kwargs})"
    )
    a(
        f"# Phase 2: {wp1_name} (n_warmup={wp1_n_warmup}, target={wp1_target}{wp1_extra_kwargs})"
    )
    a(f"# Inner kernel for both phases: {warmup_algorithm}")
    a("#")
    a("# Each phase uses a separate LaplaceMarginal factory (different maxiter).")
    a("# The logdensity_fn override ensures each phase's window_adaptation sees the")
    a("# correct LaplaceMarginal with the right L-BFGS iteration budget.")
    a("#")
    a("# Single-chain warmup: both phases run on one chain; the final state is")
    a("# broadcast to (num_chains,) for scan(vmap(kernel)) in the inference loop.")
    a("")
    a("# ── Phase 1: traversal (diagonal IMM) ────────────────────────────────────────")
    a(
        "# Phase 1 logdensity_fn: scalar wrapper _warmup_logdensity_fn (already set by laplace_preamble)"
    )
    a("_warmup_p1 = blackjax.window_adaptation(")
    a(f"    {warmup_algorithm},")
    a("    logdensity_fn,")
    a(f"    target_acceptance_rate={wp0_target}{wp0_extra_kwargs},")
    a(")")
    a(f"_warmup_key_p1 = jax.random.fold_in(jax.random.key({tuning_seed}), 0)")
    if warmup_progress_bar:
        a('with blackjax.progress_bar(label="warmup phase 1"):')
        a("    (state_phase1, _adapted_params_phase1), _ = _warmup_p1.run(")
        a(f"        _warmup_key_p1, init_position, {wp0_n_warmup}")
        a("    )")
    else:
        a("(state_phase1, _adapted_params_phase1), _ = _warmup_p1.run(")
        a(f"    _warmup_key_p1, init_position, {wp0_n_warmup}")
        a(")")
    a("# After Phase 1 (single chain):")
    a("#   state_phase1   — LaplaceHMCState with phi + theta_star")
    a('#   _adapted_params_phase1["step_size"]             scalar — from dual-avg')
    a('#   _adapted_params_phase1["inverse_mass_matrix"]   shape (phi_dim,) [diag]')
    a("")
    a("# ── Phase 2: Welford dense IMM capture ───────────────────────────────────────")
    a("# Switch to Phase 2 LaplaceMarginal (higher maxiter for accurate Hessians).")

    # Mirror the preamble logic: nuts warmup needs scalar wrapper; laplace_hmc
    # warmup needs the aux-returning marginal directly.
    _warmup_alg = ctx.get("warmup_algorithm", "blackjax.nuts")
    if _warmup_alg == "blackjax.nuts":
        a("# Scalar adapter for NUTS warmup (has_aux=False path).")
        a("")
        a("")
        a("def _warmup_logdensity_fn(phi):  # noqa: F811")
        a("    return _laplace_warmup[1](phi)[0]")
        a("")
        a("")
        a("logdensity_fn = _warmup_logdensity_fn")
    else:
        a("# laplace_hmc inner kernel needs aux-returning marginal directly.")
        a("logdensity_fn = _laplace_warmup[1]")
    a("# Seed Phase 2 dual-averaging at Phase 1's adapted step_size (0-d JAX scalar).")
    a(
        "# Keep as a JAX array — float() would trigger the buffer protocol on a still-live"
    )
    a("# device buffer and can deadlock under vmap contention (2026-05-28 lesson).")
    a("# window_adaptation accepts 0-d JAX arrays for initial_step_size.")
    a('_initial_step_size_p2 = _adapted_params_phase1["step_size"]')
    a("_warmup_p2 = blackjax.window_adaptation(")
    a(f"    {warmup_algorithm},")
    a("    logdensity_fn,")
    a("    is_mass_matrix_diagonal=False,")
    a(f"    target_acceptance_rate={wp1_target}{wp1_extra_kwargs},")
    a("    initial_step_size=_initial_step_size_p2,")
    a(")")
    a("# Use key offset from Phase 1 to avoid correlation.")
    a(f"_warmup_key_p2 = jax.random.fold_in(jax.random.key({tuning_seed + 1}), 0)")
    a(
        "# Phase 2 starts from Phase 1 end-position (single chain phi; warm-started theta_star)."
    )
    a("_init_position_p2 = state_phase1.position")
    if warmup_progress_bar:
        a('with blackjax.progress_bar(label="warmup phase 2"):')
        a("    (state_post_warmup_single, _adapted_params), _ = _warmup_p2.run(")
        a(f"        _warmup_key_p2, _init_position_p2, {wp1_n_warmup}")
        a("    )")
    else:
        a("(state_post_warmup_single, _adapted_params), _ = _warmup_p2.run(")
        a(f"    _warmup_key_p2, _init_position_p2, {wp1_n_warmup}")
        a(")")
    a("# Broadcast final state to (num_chains,) for scan(vmap(kernel)).")
    a("_state_post_warmup = jax.tree.map(")
    a("    lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),")
    a("    state_post_warmup_single,")
    a(")")
    a("# After Phase 2 (final adapted params — used by inference loop):")
    a(
        f"#   _state_post_warmup  — LaplaceHMCState with accurate theta_star (maxiter={wp1_maxiter})"
    )
    a("#                         broadcast to (num_chains,) leading axis")
    a('#   _adapted_params["step_size"]             scalar (shared across chains)')
    a('#   _adapted_params["inverse_mass_matrix"]   shape (phi_dim, phi_dim) [dense]')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCLMC tuning (diagonal preconditioning)
# ---------------------------------------------------------------------------


def _emit_mclmc_tuning(ctx: dict[str, Any]) -> str:
    """Emit mclmc_tuning (diagonal MCLMC adaptation) warmup section.

    Runs ``blackjax.mclmc_find_L_and_step_size`` over ``num_chains`` chains
    via ``jax.vmap``.  Returns per-chain L / step_size / diagonal IMM.

    Parameters
    ----------
    ctx : dict
        Required keys: n_warmup, tuning_seed, num_chains.
    """
    lines: list[str] = []
    a = lines.append

    n_warmup = ctx["n_warmup"]
    tuning_seed = ctx["tuning_seed"]
    num_chains = ctx["num_chains"]

    a("# === WARMUP: mclmc_tuning (diagonal MCLMC adaptation) ===")
    a("# blackjax.mclmc_find_L_and_step_size vmapped over num_chains chains.")
    a("# Returns per-chain: L (num_chains,), step_size (num_chains,),")
    a("# inverse_mass_matrix (num_chains, d).")
    a(
        f"_mclmc_warmup_keys = jax.random.split(jax.random.key({tuning_seed}), 2 * {num_chains})"
    )
    a(f"_mclmc_init_keys = _mclmc_warmup_keys[:{num_chains}]")
    a(f"_mclmc_tune_keys = _mclmc_warmup_keys[{num_chains}:]")
    a("_mclmc_init_positions = jax.tree.map(")
    a(
        f"    lambda x: jnp.broadcast_to(x[None], ({num_chains},) + x.shape), init_position"
    )
    a(")")
    a("")
    a("")
    a("@jax.vmap")
    a("def _mclmc_init_one(k, x0):")
    a("    return blackjax.mcmc.mclmc.init(x0, logdensity_fn, k)")
    a("")
    a("")
    a("_mclmc_init_states = _mclmc_init_one(_mclmc_init_keys, _mclmc_init_positions)")
    a("_mclmc_kernel = blackjax.mclmc.build_kernel()")
    a("")
    a("")
    a("@jax.vmap")
    a("def _mclmc_tune_one(k, state):")
    a("    s, adap, total = blackjax.mclmc_find_L_and_step_size(")
    a("        _mclmc_kernel,")
    a(f"        num_steps={n_warmup},")
    a("        state=state,")
    a("        rng_key=k,")
    a("        logdensity_fn=logdensity_fn,")
    a("        diagonal_preconditioning=True,")
    a("    )")
    a("    return s, adap")
    a("")
    a("")
    a(
        "_mclmc_states, _mclmc_adap = _mclmc_tune_one(_mclmc_tune_keys, _mclmc_init_states)"
    )
    a("_adapted_params = {")
    a('    "L": _mclmc_adap.L,')
    a('    "step_size": _mclmc_adap.step_size,')
    a('    "inverse_mass_matrix": _mclmc_adap.inverse_mass_matrix,')
    a("}")
    a("_state_post_warmup = _mclmc_states")
    a("_warmup_is_perchain = True")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCLMC LRD tuning (Low-Rank + Diagonal preconditioning)
# ---------------------------------------------------------------------------


def _emit_mclmc_lrd_tuning(ctx: dict[str, Any]) -> str:
    """Emit mclmc_lrd_tuning (LRD MCLMC adaptation) warmup section.

    Pipeline (all inlined — D8 compliant, zero tuningfork imports):
    1. Single-chain NUTS pilot (pilot_n_warmup + pilot_n_samples steps).
    2. Rank-k_rank SVD extraction (inline extract_lrd_from_samples logic).
    3. Inline make_lrd_kernel closure + vmapped mclmc_find_L_and_step_size.

    D8 note: run_pilot_nuts, extract_lrd_from_samples, and make_lrd_kernel
    are inlined directly (only jax / blackjax imports, no tuningfork).

    Parameters
    ----------
    ctx : dict
        Required keys: n_warmup, tuning_seed, num_chains.
        Optional (from wp_* spread): wp_k_rank, wp_pilot_n_warmup,
        wp_pilot_n_samples.
    """
    lines: list[str] = []
    a = lines.append

    n_warmup = ctx["n_warmup"]
    tuning_seed = ctx["tuning_seed"]
    num_chains = ctx["num_chains"]
    k_rank = ctx.get("wp_k_rank", ctx.get("bm_k_rank", 10))
    pilot_n_warmup = ctx.get("wp_pilot_n_warmup", 1000)
    pilot_n_samples = ctx.get("wp_pilot_n_samples", 1000)

    a("# === WARMUP: mclmc_lrd_tuning (Low-Rank + Diagonal MCLMC adaptation) ===")
    a(
        "# Pipeline: NUTS pilot → rank-k SVD → LRD IMM → vmapped mclmc_find_L_and_step_size."
    )
    a("# The upstream isokinetic_mclachlan integrator dispatches natively on")
    a("# LowRankInverseMassMatrix (blackjax PR #936) — no logdensity_fn wrapping.")
    a("# D8: run_pilot_nuts / extract_lrd_from_samples / make_lrd_kernel inlined")
    a("# (only jax + blackjax imports — zero tuningfork inference imports).")
    a("import blackjax.mcmc.mclmc")
    a("from blackjax.mcmc.metrics import LowRankInverseMassMatrix as _LRD")
    a("from jax.flatten_util import ravel_pytree as _lrd_ravel")
    a("")
    a(f"_lrd_tuning_seed = {tuning_seed}")
    a(f"_lrd_k_rank = {k_rank}")
    a(f"_lrd_pilot_n_warmup = {pilot_n_warmup}")
    a(f"_lrd_pilot_n_samples = {pilot_n_samples}")
    a(f"_lrd_n_warmup = {n_warmup}")
    a(f"_lrd_num_chains = {num_chains}")
    a("")
    a(
        "_lrd_pilot_key, _lrd_init_key = jax.random.split("
        "jax.random.key(_lrd_tuning_seed), 2)"
    )
    a("")
    a("# ── Phase 1: NUTS pilot (inline run_pilot_nuts) ──────────────────────────")
    a("# Single-chain window_adaptation(nuts) warmup → collect pilot positions.")
    a("_lrd_warmup_key, _lrd_sampling_key = jax.random.split(_lrd_pilot_key)")
    a("_lrd_nuts_warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)")
    a("(_lrd_nuts_state, _lrd_nuts_params), _ = _lrd_nuts_warmup.run(")
    a("    _lrd_warmup_key, init_position, _lrd_pilot_n_warmup")
    a(")")
    a("_lrd_step_size = (")
    a('    _lrd_nuts_params["step_size"]')
    a("    if isinstance(_lrd_nuts_params, dict)")
    a('    else getattr(_lrd_nuts_params, "step_size")')
    a(")")
    a("_lrd_pilot_imm = (")
    a('    _lrd_nuts_params["inverse_mass_matrix"]')
    a("    if isinstance(_lrd_nuts_params, dict)")
    a('    else getattr(_lrd_nuts_params, "inverse_mass_matrix")')
    a(")")
    a("_lrd_nuts_kernel = blackjax.nuts(")
    a("    logdensity_fn, step_size=_lrd_step_size, inverse_mass_matrix=_lrd_pilot_imm")
    a(")")
    a("")
    a("")
    a("def _lrd_body_fn(state, key):")
    a("    state, info = _lrd_nuts_kernel.step(key, state)")
    a("    return state, state.position")
    a("")
    a("")
    a("_, _lrd_pilot_positions = jax.lax.scan(")
    a("    _lrd_body_fn,")
    a("    _lrd_nuts_state,")
    a("    jax.random.split(_lrd_sampling_key, _lrd_pilot_n_samples),")
    a(")")
    a("")
    a("# ── Phase 2: LRD extraction (inline extract_lrd_from_samples) ────────────")
    a("# SVD of standardised pilot samples → (sigma, U, lam) LRD components.")
    a(
        "_lrd_flat_positions = jax.vmap(lambda p: _lrd_ravel(p)[0])(_lrd_pilot_positions)"
    )
    a("_lrd_mean = jnp.mean(_lrd_flat_positions, axis=0)")
    a("_lrd_sigma = jnp.std(_lrd_flat_positions, axis=0)")
    a("_lrd_sigma = jnp.where(_lrd_sigma == 0.0, 1.0, _lrd_sigma)")
    a(
        "_lrd_flat_std = (_lrd_flat_positions - _lrd_mean[None, :]) / _lrd_sigma[None, :]"
    )
    a("_, _lrd_S, _lrd_Vt = jnp.linalg.svd(_lrd_flat_std, full_matrices=False)")
    a("_lrd_V = _lrd_Vt.T")
    a("_lrd_N = _lrd_flat_std.shape[0]")
    a("_lrd_lam_all = (_lrd_S ** 2) / _lrd_N")
    a("# Clamp k_rank to the number of available SVD modes: svd(full_matrices=False)")
    a("# on a (pilot_n_samples, d) matrix yields min(pilot_n_samples, d) singular")
    a("# values, so slicing [:k_rank] silently truncates when k_rank exceeds that.")
    a("_lrd_k_rank = min(_lrd_k_rank, _lrd_lam_all.shape[0])")
    a("_lrd_sort_idx = jnp.argsort(jnp.abs(_lrd_lam_all - 1.0))[::-1]")
    a("_lrd_top_idx = _lrd_sort_idx[:_lrd_k_rank]")
    a("_lrd_lam = _lrd_lam_all[_lrd_top_idx]")
    a("_lrd_U = _lrd_V[:, _lrd_top_idx]")
    a("_lrd_imm = _LRD(sigma=_lrd_sigma, U=_lrd_U, lam=_lrd_lam)")
    a("")
    a("# ── Phase 2b: build LRD kernel (inline make_lrd_kernel) ──────────────────")
    a("# Closure over _lrd_imm — always routes through LRD geometry regardless")
    a("# of the diagonal placeholder that mclmc_find_L_and_step_size passes.")
    a("_lrd_base_kernel = blackjax.mclmc.build_kernel()")
    a("")
    a("")
    a(
        "def _lrd_kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size):"
    )
    a("    # Override warmup placeholder IMM with the bound LRD mass matrix.")
    a(
        "    return _lrd_base_kernel(rng_key, state, logdensity_fn, _lrd_imm, L, step_size)"
    )
    a("")
    a("")
    a("# ── Phase 3: vmapped mclmc_find_L_and_step_size ──────────────────────────")
    a("_lrd_all_keys = jax.random.split(_lrd_init_key, 2 * _lrd_num_chains)")
    a("_lrd_chain_init_keys = _lrd_all_keys[:_lrd_num_chains]")
    a("_lrd_chain_tune_keys = _lrd_all_keys[_lrd_num_chains:]")
    a("_lrd_init_positions = jax.tree.map(")
    a(
        "    lambda x: jnp.broadcast_to(x[None], (_lrd_num_chains,) + x.shape), init_position"
    )
    a(")")
    a("")
    a("")
    a("@jax.vmap")
    a("def _lrd_init_one(k, x0):")
    a("    return blackjax.mcmc.mclmc.init(x0, logdensity_fn, k)")
    a("")
    a("")
    a("_lrd_init_states = _lrd_init_one(_lrd_chain_init_keys, _lrd_init_positions)")
    a("")
    a("")
    a("@jax.vmap")
    a("def _lrd_tune_one(k, state):")
    a("    s, adap, _ = blackjax.mclmc_find_L_and_step_size(")
    a("        _lrd_kernel,")
    a("        num_steps=_lrd_n_warmup,")
    a("        state=state,")
    a("        rng_key=k,")
    a("        logdensity_fn=logdensity_fn,")
    a("        diagonal_preconditioning=False,")
    a("    )")
    a("    return s, adap")
    a("")
    a("")
    a("_lrd_states, _lrd_adap = _lrd_tune_one(_lrd_chain_tune_keys, _lrd_init_states)")
    a("")
    a("# Broadcast the single shared LRD IMM to (num_chains, ...) for per-chain vmap.")
    a(
        "_lrd_sigma_b = jnp.broadcast_to(_lrd_sigma[None], (_lrd_num_chains,) + _lrd_sigma.shape)"
    )
    a("_lrd_U_b = jnp.broadcast_to(_lrd_U[None], (_lrd_num_chains,) + _lrd_U.shape)")
    a(
        "_lrd_lam_b = jnp.broadcast_to(_lrd_lam[None], (_lrd_num_chains,) + _lrd_lam.shape)"
    )
    a("_lrd_imm_batched = _LRD(sigma=_lrd_sigma_b, U=_lrd_U_b, lam=_lrd_lam_b)")
    a("")
    a("_adapted_params = {")
    a('    "L": _lrd_adap.L,')
    a('    "step_size": _lrd_adap.step_size,')
    a('    "inverse_mass_matrix": _lrd_imm_batched,')
    a("}")
    a("_state_post_warmup = _lrd_states")
    a("_warmup_is_perchain = True")

    return "\n".join(lines)
