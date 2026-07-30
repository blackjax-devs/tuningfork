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
"""Descriptor-driven Python emit-function for sampler sections.

Replaces the 15 ``.tmpl`` files in
``_templates/samplers/{nuts,hmc,mhmc,rmhmc,dynamic_hmc,dmhmc,ghmc,
laplace_hmc,laplace_mhmc,laplace_dhmc,laplace_dmhmc,
mala,barker,rwm,vi_sampler}.py.tmpl``
(695 LOC total) with a single Python entry point.

All routing is resolved at generation time (P1 straight-line principle).
No dispatch on ``base_method_name`` string equality — every fork comes
from descriptors (``per_chain_param_keys``, ``reinit_state``,
``extra_required_kwargs``) or family-level structural differences.

D8 compliant: emitted strings contain no ``import tuningfork``.

Family groupings
----------------
- **HMC family** (nuts, hmc, mhmc, rmhmc, dynamic_hmc, dmhmc, ghmc):
  ``kernel_builder = _build_kernel``; factory called with step_size + IMM.
  dynamic_hmc/dmhmc/ghmc additionally emit ``_state_reinit``.
- **Laplace family** (laplace_hmc, laplace_mhmc, laplace_dhmc, laplace_dmhmc):
  factory takes ``log_joint_fn`` + ``theta_init`` instead of ``logdensity_fn``.
  laplace_dhmc/laplace_dmhmc additionally emit ``_state_reinit``.
- **MALA** (mala): step_size only; IMM arg present for protocol compat.
- **Barker** (barker): step_size + IMM; thin wrapper.
- **RWM** (rwm): ``sigma`` HP (not step_size); isotropic Gaussian proposal.
- **VI** (meanfield_vi, fullrank_vi): IID samples from fitted distribution;
  all VI-specific slots resolved from ``ctx`` (populated by ``_emit_script.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tuningfork.base_method._base import BaseMethod

# ---------------------------------------------------------------------------
# Internal family predicates — derived purely from descriptors
# ---------------------------------------------------------------------------

_LAPLACE_EXTRA_KWARGS = frozenset({"log_joint_fn", "theta_init"})
_VI_NAMES = frozenset({"meanfield_vi", "fullrank_vi"})


def _is_laplace(base_method: BaseMethod) -> bool:
    """True when the method requires log_joint_fn/theta_init extra kwargs."""
    return _LAPLACE_EXTRA_KWARGS.issubset(set(base_method.extra_required_kwargs))


def _is_vi(base_method: BaseMethod) -> bool:
    """True when the method name is a VI sampler."""
    return base_method.name in _VI_NAMES


def _needs_imm(base_method: BaseMethod) -> bool:
    """True when step_size + inverse_mass_matrix are per-chain adapted params."""
    return "inverse_mass_matrix" in base_method.per_chain_param_keys


def _is_gradient_free(base_method: BaseMethod) -> bool:
    """True when per_chain_param_keys is empty (no warmup-adapted params)."""
    return len(base_method.per_chain_param_keys) == 0


def _is_numeric_tree(value: Any) -> bool:
    """Return whether ``value`` is an inline finite numeric scalar/tree."""
    import math
    import numbers

    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_is_numeric_tree(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def emit_sampler(base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit the sampler section for a recipe reproduction script.

    Replaces the per-method ``.py.tmpl`` template files with a single
    descriptor-driven Python function.

    Parameters
    ----------
    base_method : BaseMethod
        Registry entry for the sampler.  Descriptors consumed:
        - ``base_method.name`` — method identifier string.
        - ``base_method.per_chain_param_keys`` — drives which adapted params
          the kernel factory call needs.
        - ``base_method.reinit_state`` — drives ``_state_reinit`` emission.
        - ``base_method.extra_required_kwargs`` — drives laplace / VI dispatch.
    ctx : dict
        Substitution context from ``emit_script()``.  Required keys depend on
        the method family; see per-family helpers below.

    Returns
    -------
    str
        Python source for the sampler block (D8 compliant — no tuningfork
        inference imports).
    """
    if _is_vi(base_method):
        body = _emit_vi_sampler(base_method, ctx)
    elif _is_laplace(base_method):
        body = _emit_laplace_sampler(base_method, ctx)
    else:
        # Gradient-based non-Laplace families — dispatch on structural differences.
        name = base_method.name
        if name == "rwm":
            body = _emit_rwm(base_method, ctx)
        elif name == "mala":
            body = _emit_mala(base_method, ctx)
        elif name == "mclmc":
            body = _emit_mclmc(base_method, ctx)
        else:
            # HMC family: nuts, hmc, mhmc, rmhmc, dynamic_hmc, dmhmc, ghmc, barker
            body = _emit_hmc_family(base_method, ctx)
    # Ensure trailing newline so _strip_no_warmup_try_block can match the last
    # indented line of the try/except block (the regex requires \n after each line).
    if not body.endswith("\n"):
        body += "\n"
    return body


# ---------------------------------------------------------------------------
# HMC family: nuts, hmc, mhmc, rmhmc, dynamic_hmc, dmhmc, ghmc, barker
# ---------------------------------------------------------------------------

# Which HMC-family methods have a fixed num_integration_steps HP in the recipe.
# (dynamic_hmc, dmhmc, ghmc auto-tune trajectory length; nuts auto-doubles.)
_HMC_WITH_NIS = frozenset({"hmc", "mhmc", "rmhmc"})

# Methods that use the special blackjax.rmhmc IMM→mass_matrix inversion.
_RMHMC_NAMES = frozenset({"rmhmc"})

# GHMC extra args: alpha + delta (scalar, read from _adapted_params with fallback).
_GHMC_NAMES = frozenset({"ghmc"})


def _emit_hmc_family(base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for the HMC family.

    Covers: nuts, hmc, mhmc, rmhmc, dynamic_hmc, dmhmc, ghmc, barker.

    All read step_size + inverse_mass_matrix from warmup adaptation.
    dynamic_hmc / dmhmc / ghmc additionally emit _state_reinit because they
    require a different state type than NUTSState (which warmup produces).
    """
    name = base_method.name
    # Use structural frozenset for the emit-time _state_reinit decision.
    # ghmc has reinit_state=False in the registry descriptor (the runner uses
    # MEADS warmup which produces GHMCState directly) but the emitted script
    # DOES need _state_reinit because window_adaptation warmup produces NUTSState,
    # not GHMCState.  This mirrors _STATE_REINIT_SAMPLERS in _emit_script.py.
    _HMC_EMIT_REINIT = frozenset({"dynamic_hmc", "dmhmc", "ghmc"})
    needs_reinit = name in _HMC_EMIT_REINIT
    has_nis = name in _HMC_WITH_NIS
    is_rmhmc = name in _RMHMC_NAMES
    is_ghmc = name in _GHMC_NAMES

    lines: list[str] = []
    a = lines.append

    # Section header comment.
    a(f"# === SAMPLER: {name} (step_size and IMM adapted per-chain from warmup) ===")

    # n_params block: needed by methods that require a default IMM at no_warmup
    # init time (all HMC-family except nuts which doesn't use it at no_warmup init).
    # Emit for all because it's cheap and avoids conditional complexity.
    a("from jax.flatten_util import ravel_pytree as _ravel_pytree")
    a("")
    a("_flat_init, _ = _ravel_pytree(init_position)")
    a("_n_params = int(_flat_init.shape[0])")
    a("")

    # Default step_size + IMM (used for no_warmup path init).
    _bm_step_size = ctx.get("bm_step_size", 1.0)
    _bm_imm = ctx.get("bm_inverse_mass_matrix")
    _is_baked_replay = bool(ctx.get("is_baked_replay", False))
    if _is_baked_replay and (
        not _is_numeric_tree(_bm_step_size) or not _is_numeric_tree(_bm_imm)
    ):
        raise ValueError(
            "No-warmup replay requires numeric inline step_size and "
            "inverse_mass_matrix; "
            "refusing to use a sidecar sentinel or invent sampler tuning."
        )
    a(f"_default_step_size = {_bm_step_size!r}")
    if not _is_numeric_tree(_bm_imm):
        a("_default_imm = jnp.ones(_n_params)")
    else:
        a(f"_default_imm = jnp.asarray({_bm_imm!r})")

    # num_integration_steps: recipe-pinned HP for hmc/mhmc/rmhmc.
    if has_nis:
        _nis = ctx.get("bm_num_integration_steps", 10)
        a("# num_integration_steps is a recipe-pinned HP (NOT adapted by warmup).")
        a(f"_num_steps = {_nis!r}")

    # GHMC extra: alpha + delta (scalar, not per-chain).
    if is_ghmc:
        _alpha = ctx.get("bm_alpha", 0.9)
        _delta = ctx.get("bm_delta", 0.1)
        a("# alpha and delta are scalar (not per-chain) recipe hyperparameters.")
        a(f'_alpha = float(_adapted_params.get("alpha", {_alpha!r}))')
        a(f'_delta = float(_adapted_params.get("delta", {_delta!r}))')

    # rmhmc: emit IMM→mass_matrix helper.
    if is_rmhmc:
        a("")
        a("")
        a("def _imm_to_mass_matrix(inverse_mass_matrix):")
        a('    """Convert inverse_mass_matrix to mass_matrix for blackjax.rmhmc."""')
        a("    if inverse_mass_matrix.ndim == 1:")
        a("        return 1.0 / inverse_mass_matrix")
        a("    return jnp.linalg.inv(inverse_mass_matrix)")

    a("")
    a("")

    # _build_kernel factory.
    a("def _build_kernel(step_size, inverse_mass_matrix):")
    if name == "nuts":
        _mnd = ctx.get("max_num_doublings", 10)
        a("    return blackjax.nuts(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a(f"        max_num_doublings={_mnd!r},")
        a("    ).step")
    elif name == "hmc":
        a("    return blackjax.hmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        num_integration_steps=_num_steps,")
        a("    ).step")
    elif name == "mhmc":
        a("    return blackjax.mhmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        num_integration_steps=_num_steps,")
        a("    ).step")
    elif name == "rmhmc":
        a("    return blackjax.rmhmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        mass_matrix=_imm_to_mass_matrix(inverse_mass_matrix),")
        a("        num_integration_steps=_num_steps,")
        a("    ).step")
    elif name == "dynamic_hmc":
        a("    return blackjax.dynamic_hmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        integration_steps_fn=_integration_steps_fn,")
        if ctx.get("chees_adapted", False):
            a("        next_random_arg_fn=_next_random_arg_fn,")
            a("        integration_steps_params=_integration_steps_params,")
        a("    ).step")
    elif name == "dmhmc":
        a("    return blackjax.dmhmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        integration_steps_fn=_integration_steps_fn,")
        a("    ).step")
    elif name == "ghmc":
        a(
            "    # inverse_mass_matrix from warmup serves as momentum_inverse_scale for GHMC."
        )
        a("    return blackjax.ghmc(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        momentum_inverse_scale=inverse_mass_matrix,")
        a("        alpha=_alpha,")
        a("        delta=_delta,")
        a("    ).step")
    elif name == "barker":
        a("    return blackjax.barker(")
        a("        logdensity_fn,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("    ).step")

    a("")
    a("")
    a("# kernel_builder is the protocol expected by the inference loop when")
    a("# _adapted_params is multi-chain: each chain builds its own kernel.")
    a("kernel_builder = _build_kernel")

    # Explicit state initializer used by pre-batched no-warmup replay.  The
    # kernel builder returns a step callable, not a SamplingAlgorithm, so state
    # construction must call the concrete BlackJAX factory directly.
    if name in {"dynamic_hmc", "dmhmc"}:
        a("")
        a("def _state_init(position, rng_key):")
        a(f"    return blackjax.{name}(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        integration_steps_fn=_integration_steps_fn,")
        if name == "dynamic_hmc" and ctx.get("chees_adapted", False):
            a("        next_random_arg_fn=_next_random_arg_fn,")
            a("        integration_steps_params=_integration_steps_params,")
        a("    ).init(position, rng_key)")
    elif name == "ghmc":
        a("")
        a("def _state_init(position, rng_key=None):")
        a("    return blackjax.ghmc(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        momentum_inverse_scale=_default_imm,")
        a("        alpha=_alpha,")
        a("        delta=_delta,")
        a("    ).init(position, rng_key)")
    elif name == "rmhmc":
        a("")
        a("def _state_init(position, rng_key=None):")
        a("    return blackjax.rmhmc(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        mass_matrix=_imm_to_mass_matrix(_default_imm),")
        a("        num_integration_steps=_num_steps,")
        a("    ).init(position)")
    else:
        a("")
        a("def _state_init(position, rng_key=None):")
        a(f"    return blackjax.{name}(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        if has_nis:
            a("        num_integration_steps=_num_steps,")
        if name == "nuts":
            a(f"        max_num_doublings={ctx.get('max_num_doublings', 10)!r},")
        a("    ).init(position)")

    # _state_reinit for dynamic_hmc / dmhmc / ghmc.
    if needs_reinit:
        a("")
        a("")
        if name == "dynamic_hmc":
            a("def _state_reinit(step_size, inverse_mass_matrix, position, rng_key):")
            a('    """Re-init per-chain DynamicHMCState from (position, rng_key)."""')
            a("    return blackjax.dynamic_hmc(")
            a("        logdensity_fn,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        integration_steps_fn=_integration_steps_fn,")
            a("    ).init(position, rng_key)")
        elif name == "dmhmc":
            a("def _state_reinit(step_size, inverse_mass_matrix, position, rng_key):")
            a('    """Re-init per-chain DynamicHMCState from (position, rng_key)."""')
            a("    return blackjax.dmhmc(")
            a("        logdensity_fn,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        integration_steps_fn=_integration_steps_fn,")
            a("    ).init(position, rng_key)")
        elif name == "ghmc":
            a("def _state_reinit(step_size, inverse_mass_matrix, position, rng_key):")
            a('    """Re-init per-chain GHMCState from (position, rng_key)."""')
            a("    return blackjax.ghmc(")
            a("        logdensity_fn,")
            a("        step_size=step_size,")
            a("        momentum_inverse_scale=inverse_mass_matrix,")
            a("        alpha=_alpha,")
            a("        delta=_delta,")
            a("    ).init(position, rng_key)")

    # no_warmup init block (T1.3: only emitted for no_warmup recipes).
    # The caller (_emit_script.py) strips this block for non-no_warmup paths.
    # We still emit it so the no_warmup path works correctly.
    a("")
    a("# Bind _state_post_warmup: warmup template sets it when adaptation runs.")
    a("# For no_warmup, initialise from init_position here.")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    _warmup_init_is_single_chain = True")
    if needs_reinit:
        # dynamic_hmc/dmhmc/ghmc: placeholder state (reinit handles per-chain init).
        if name in ("dynamic_hmc", "dmhmc"):
            a(
                "    # no_warmup path: initialise a placeholder NUTSState from init_position."
            )
            a("    # _state_reinit will replace it per-chain inside the vmap.")
            a("    _state_post_warmup = blackjax.nuts(")
            a("        logdensity_fn,")
            a("        step_size=_default_step_size,")
            a("        inverse_mass_matrix=_default_imm,")
            a("    ).init(init_position)")
        elif name == "ghmc":
            a(
                "    # no_warmup path: initialise a placeholder NUTSState from init_position."
            )
            a("    # _state_reinit will replace it per-chain inside the vmap.")
            a("    _state_post_warmup = blackjax.nuts(")
            a("        logdensity_fn,")
            a("        step_size=_default_step_size,")
            a("        inverse_mass_matrix=_default_imm,")
            a("    ).init(init_position)")
    elif name == "nuts":
        # nuts: no IMM-dependent init (uses default).
        _mnd = ctx.get("max_num_doublings", 10)
        a("    _state_post_warmup = blackjax.nuts(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a(f"        max_num_doublings={_mnd!r},")
        a("    ).init(init_position)")
    elif name == "hmc":
        a("    _state_post_warmup = blackjax.hmc(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        num_integration_steps=_num_steps,")
        a("    ).init(init_position)")
    elif name == "mhmc":
        a("    _state_post_warmup = blackjax.mhmc(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        num_integration_steps=_num_steps,")
        a("    ).init(init_position)")
    elif name == "rmhmc":
        a("    _state_post_warmup = blackjax.rmhmc(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        mass_matrix=_imm_to_mass_matrix(_default_imm),")
        a("        num_integration_steps=_num_steps,")
        a("    ).init(init_position)")
    elif name == "barker":
        a("    _state_post_warmup = blackjax.barker(")
        a("        logdensity_fn,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("    ).init(init_position)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MALA
# ---------------------------------------------------------------------------


def _emit_mala(_base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for MALA (step_size only; no IMM)."""
    lines: list[str] = []
    a = lines.append

    _bm_step_size = ctx.get("bm_step_size", 0.1)

    a(
        "# === SAMPLER: mala (Metropolis-Adjusted Langevin; step_size only, no mass matrix) ==="
    )
    a(f"_default_step_size = {_bm_step_size!r}")
    a("")
    a("")
    a("def _build_kernel(step_size, inverse_mass_matrix):  # noqa: ARG001")
    a("    # MALA does not use inverse_mass_matrix; IMM arg present for protocol")
    a("    # compatibility with the kernel_builder interface in the inference loop.")
    a("    return blackjax.mala(logdensity_fn, step_size=step_size).step")
    a("")
    a("")
    a("# kernel_builder is the protocol expected by the inference loop.")
    a("kernel_builder = _build_kernel")
    a("")
    a("# Bind _state_post_warmup: warmup template sets it when adaptation runs;")
    a("# for no_warmup we initialize from init_position here.")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    _warmup_init_is_single_chain = True")
    a(
        "    _state_post_warmup = blackjax.mala("
        " logdensity_fn, step_size=_default_step_size"
        " ).init(init_position)"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RWM
# ---------------------------------------------------------------------------


def _emit_rwm(_base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for RWM (Random-Walk Metropolis).

    RWM uses sigma (not step_size) and builds an isotropic Gaussian proposal
    via ravel_pytree.  The kernel_builder accepts (step_size, inverse_mass_matrix)
    for protocol compat — internally it uses step_size as the sigma/scale.
    """
    lines: list[str] = []
    a = lines.append

    # bm_sigma is the recipe HP for RWM (stored as "sigma" in base_method_params).
    _sigma = ctx.get("bm_sigma", 0.1)

    a(
        "# === SAMPLER: rwm (Random-Walk Metropolis with isotropic Gaussian proposal) ==="
    )
    a("# Wraps blackjax.rmh with an isotropic Gaussian proposal in flat parameter")
    a("# space, un-ravelled back to the pytree structure.")
    a("from jax.flatten_util import ravel_pytree as _ravel_pytree")
    a("")
    a("_flat_init, _ = _ravel_pytree(init_position)")
    a(f"_default_step_size = {_sigma!r}")
    a("")
    a("")
    a("def _build_kernel(step_size, inverse_mass_matrix):  # noqa: ARG001")
    a("    # RWM uses an isotropic Gaussian proposal scaled by step_size; IMM arg is")
    a("    # present for protocol compatibility with the inference loop.")
    a("    def _proposal(rng_key, position):")
    a("        flat, unravel = _ravel_pytree(position)")
    a("        noise = jax.random.normal(rng_key, shape=flat.shape) * step_size")
    a("        return unravel(flat + noise)")
    a("")
    a("    return blackjax.rmh(logdensity_fn, proposal_generator=_proposal).step")
    a("")
    a("")
    a("# kernel_builder is the protocol expected by the inference loop.")
    a("kernel_builder = _build_kernel")
    a("")
    a("")
    a("def _default_rwm_proposal(rng_key, position):")
    a("    flat, unravel = _ravel_pytree(position)")
    a(
        "    return unravel(flat + jax.random.normal(rng_key, shape=flat.shape) * _default_step_size)"
    )
    a("")
    a("")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    _warmup_init_is_single_chain = True")
    a(
        "    _state_post_warmup = blackjax.rmh("
        " logdensity_fn, proposal_generator=_default_rwm_proposal"
        " ).init(init_position)"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Laplace family: laplace_hmc, laplace_mhmc, laplace_dhmc, laplace_dmhmc
# ---------------------------------------------------------------------------

# Laplace methods that need num_integration_steps (fixed trajectory length).
_LAPLACE_WITH_NIS = frozenset({"laplace_hmc", "laplace_mhmc"})

# Laplace methods that need _state_reinit (dynamic trajectory length + rng_arg).
_LAPLACE_WITH_REINIT = frozenset({"laplace_dhmc", "laplace_dmhmc"})


def _emit_laplace_sampler(base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for the Laplace family.

    log_joint_fn and theta_init come from the laplace_preamble section above.
    D8: no logdensity_fn reference — laplace_* uses log_joint_fn directly.
    """
    name = base_method.name
    # needs_reinit governs _state_reinit emission:
    # - laplace_dhmc/laplace_dmhmc: always need reinit (dynamic trajectory + rng arg).
    # - laplace_hmc/laplace_mhmc: need reinit when warmup used NUTS substitute
    #   (NUTS produces HMCState; laplace_hmc requires LaplaceHMCState with theta_star).
    #   ctx["_laplace_needs_warmup_state_reinit"] is set True by _emit_script.py when
    #   _warmup_sampler == "nuts" for a laplace_hmc/laplace_mhmc recipe.
    #   When warmup uses laplace_hmc as inner kernel (multi-phase path), the warmup
    #   already produces LaplaceHMCState — no reinit needed.
    _warmup_reinit_for_hmc_mhmc = ctx.get("_laplace_needs_warmup_state_reinit", False)
    needs_reinit = name in _LAPLACE_WITH_REINIT or (
        name in {"laplace_hmc", "laplace_mhmc"} and _warmup_reinit_for_hmc_mhmc
    )
    has_nis = name in _LAPLACE_WITH_NIS

    lines: list[str] = []
    a = lines.append

    a(f"# === SAMPLER: {name} (Laplace-marginalised sampler on phi-space) ===")
    a("# log_joint_fn and theta_init come from the laplace_preamble section above.")
    a("# D8: no logdensity_fn reference -- laplace_* uses log_joint_fn directly.")
    a("from jax.flatten_util import ravel_pytree as _ravel_pytree")
    a("")
    a("_flat_phi_init, _ = _ravel_pytree(init_position)")
    a("_n_phi = int(_flat_phi_init.shape[0])")
    a("")

    _bm_step_size = ctx.get("bm_step_size", 1.0)
    _opt_kwargs = ctx.get("bm_optimizer_kwargs_expr", "{}")

    a(f"_optimizer_kwargs = {_opt_kwargs}")
    a(f"_default_step_size = {_bm_step_size!r}")
    a("_default_imm = jnp.ones(_n_phi)")

    if has_nis:
        _nis = ctx.get("bm_num_integration_steps", 10)
        a(f"_num_steps = {_nis!r}")

    a("")
    a("")
    a("def _build_kernel(step_size, inverse_mass_matrix):")

    if name == "laplace_hmc":
        a("    return blackjax.laplace_hmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        num_integration_steps=_num_steps,")
        a("        **_optimizer_kwargs,")
        a("    ).step")
    elif name == "laplace_mhmc":
        a("    return blackjax.laplace_mhmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        num_integration_steps=_num_steps,")
        a("        **_optimizer_kwargs,")
        a("    ).step")
    elif name == "laplace_dhmc":
        a("    return blackjax.laplace_dhmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        **_optimizer_kwargs,")
        a("    ).step")
    elif name == "laplace_dmhmc":
        a("    return blackjax.laplace_dmhmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=step_size,")
        a("        inverse_mass_matrix=inverse_mass_matrix,")
        a("        **_optimizer_kwargs,")
        a("    ).step")

    a("")
    a("")
    a("# kernel_builder is the protocol expected by the inference loop.")
    a("kernel_builder = _build_kernel")

    if needs_reinit:
        a("")
        a("")
        a("def _state_reinit(step_size, inverse_mass_matrix, position, rng_key):")
        if name == "laplace_hmc":
            a('    """Re-init per-chain LaplaceHMCState from warmup position.')
            a("    Warmup used blackjax.nuts (WARMUP_SUBSTITUTE path), which produces")
            a("    HMCState.  laplace_hmc requires LaplaceHMCState (with theta_star).")
            a(
                "    rng_key accepted for interface compatibility but ignored (laplace_hmc.init"
            )
            a('    does not consume a PRNG key)."""')
            a("    return blackjax.laplace_hmc(")
            a("        log_joint_fn,")
            a("        theta_init,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        num_integration_steps=_num_steps,")
            a("        **_optimizer_kwargs,")
            a("    ).init(position)")
        elif name == "laplace_mhmc":
            a('    """Re-init per-chain LaplaceHMCState from warmup position.')
            a("    Warmup used blackjax.nuts (WARMUP_SUBSTITUTE path), which produces")
            a("    HMCState.  laplace_mhmc requires LaplaceHMCState (with theta_star).")
            a(
                "    rng_key accepted for interface compatibility but ignored (laplace_mhmc.init"
            )
            a('    does not consume a PRNG key)."""')
            a("    return blackjax.laplace_mhmc(")
            a("        log_joint_fn,")
            a("        theta_init,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        num_integration_steps=_num_steps,")
            a("        **_optimizer_kwargs,")
            a("    ).init(position)")
        elif name == "laplace_dhmc":
            a(
                '    """Re-init per-chain LaplaceDynamicHMCState from (position, rng_key)."""'
            )
            a("    return blackjax.laplace_dhmc(")
            a("        log_joint_fn,")
            a("        theta_init,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        **_optimizer_kwargs,")
            a("    ).init(position, rng_key)")
        elif name == "laplace_dmhmc":
            a(
                '    """Re-init per-chain LaplaceDynamicHMCState from (position, rng_key)."""'
            )
            a("    return blackjax.laplace_dmhmc(")
            a("        log_joint_fn,")
            a("        theta_init,")
            a("        step_size=step_size,")
            a("        inverse_mass_matrix=inverse_mass_matrix,")
            a("        **_optimizer_kwargs,")
            a("    ).init(position, rng_key)")

    # no_warmup init block.
    a("")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    # no_warmup path: placeholder state; _state_reinit handles per-chain init.")
    a("    _warmup_init_is_single_chain = True")
    if name in _LAPLACE_WITH_REINIT:
        # laplace_dhmc/dmhmc: placeholder via laplace_hmc (which produces LaplaceHMCState).
        a("    _state_post_warmup = blackjax.laplace_hmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        num_integration_steps=10,")
        a("        **_optimizer_kwargs,")
        a("    ).init(init_position)")
    elif name == "laplace_hmc":
        a("    _state_post_warmup = blackjax.laplace_hmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        num_integration_steps=_num_steps,")
        a("        **_optimizer_kwargs,")
        a("    ).init(init_position)")
    elif name == "laplace_mhmc":
        a("    _state_post_warmup = blackjax.laplace_mhmc(")
        a("        log_joint_fn,")
        a("        theta_init,")
        a("        step_size=_default_step_size,")
        a("        inverse_mass_matrix=_default_imm,")
        a("        num_integration_steps=_num_steps,")
        a("        **_optimizer_kwargs,")
        a("    ).init(init_position)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VI family: meanfield_vi, fullrank_vi
# ---------------------------------------------------------------------------


def _emit_vi_sampler(_base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for VI samplers (meanfield_vi, fullrank_vi).

    All VI-specific slots (vi_prefix, vi_module, vi_state_name, vi_info_name)
    are pre-populated in ctx by _emit_script.py.  We reference them directly
    here rather than via $slot substitution.
    """
    lines: list[str] = []
    a = lines.append

    vp = ctx["vi_prefix"]
    vi_module = ctx["vi_module"]
    vi_state_name = ctx["vi_state_name"]
    vi_info_name = ctx["vi_info_name"]
    base_method_name = ctx["base_method_name"]
    _num_opt_steps = ctx.get("bm_num_optimization_steps", 100)
    _tuning_seed = ctx["tuning_seed"]

    a(
        f"# === SAMPLER: {base_method_name} (full VI optimisation; IID samples from fitted distribution) ==="
    )
    a("# Variational inference in sampler mode.")
    a(f"# The VI optimisation loop (Adam, {_num_opt_steps} steps) runs during")
    a(
        "# init via jax.lax.scan; each .step draws one IID sample from the fitted distribution."
    )
    a("# Does NOT use step_size or inverse_mass_matrix (VI has no MH acceptance step).")
    a(
        "# step_size / IMM args present only for protocol compat with kernel_builder interface."
    )
    a(f"import collections as {vp}_collections")
    a(f"import optax as _optax_{vp[1:]}")
    a(f"import {vi_module} as {vp}_module")
    a(f"from jax.flatten_util import ravel_pytree as {vp}_ravel")
    a("")
    a("# Inline state definition (D8: no tuningfork import in inference choreography).")
    a(
        f"{vi_state_name} = {vp}_collections.namedtuple("
        f'    "{vi_state_name}", ["position", "vi_state"]'
        ")"
    )
    a("")
    a(f"{vp}_num_opt_steps = {_num_opt_steps!r}")
    a(f"{vp}_optimizer = _optax_{vp[1:]}.adam(1e-2)")
    a("")
    a("")
    a("def _build_kernel(step_size, inverse_mass_matrix):  # noqa: ARG001")
    a("    # VI does not use step_size or inverse_mass_matrix; both args present for")
    a(
        "    # protocol compatibility with the kernel_builder interface in the inference loop."
    )
    a("    # Each call draws one IID sample from the fitted variational distribution.")
    a("    def _vi_step(rng_key, state):")
    a(f"        samples = {vp}_module.sample(rng_key, state.vi_state, num_samples=1)")
    a("        new_position = jax.tree.map(lambda x: x[0], samples)")
    a("        return (")
    a(f"            {vi_state_name}(position=new_position, vi_state=state.vi_state),")
    a(f"            {vp}_module.{vi_info_name}(elbo=jnp.asarray(0.0)),")
    a("        )")
    a("")
    a("    return _vi_step")
    a("")
    a("")
    a("kernel_builder = _build_kernel")
    a("")
    a("")
    a("# Bind _state_post_warmup for the no_warmup path: run the full VI optimisation")
    a("# loop here and store the final VI state (position = variational mean).")
    a("# For a VI-warmup recipe this block is unreachable -- warmup template sets it.")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    _warmup_init_is_single_chain = True")
    a(f"    {vp}_init_state = {vp}_module.init(init_position, {vp}_optimizer)")
    a("")
    a(f"    def {vp}_one_step(carry, step_key):")
    a(f"        new_state, info = {vp}_module.step(")
    a(f"            step_key, carry, logdensity_fn, {vp}_optimizer, 5")
    a("        )")
    a("        return new_state, info")
    a("")
    a(
        f"    {vp}_keys = jax.random.split(jax.random.key({_tuning_seed!r}), {vp}_num_opt_steps)"
    )
    a(
        f"    {vp}_final_vi_state, _ = jax.lax.scan("
        f" {vp}_one_step, {vp}_init_state, {vp}_keys"
        " )"
    )
    a("    # Use the variational mean as the initial position.")
    a(f"    {vp}_mu_flat, {vp}_unravel = {vp}_ravel({vp}_final_vi_state.mu)")
    a(f"    {vp}_init_pos = {vp}_unravel({vp}_mu_flat)")
    a(
        f"    _state_post_warmup = {vi_state_name}("
        f"position={vp}_init_pos, vi_state={vp}_final_vi_state"
        ")"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCLMC sampler
# ---------------------------------------------------------------------------


def _emit_mclmc(base_method: BaseMethod, ctx: dict[str, Any]) -> str:
    """Emit sampler block for MCLMC (Microcanonical Langevin Monte Carlo).

    Defines ``kernel_builder(step_size, imm, L)`` that instantiates
    ``blackjax.mclmc`` with the per-chain L alongside step_size and imm.
    The third positional argument L is required; the inference loop emitter
    passes it when ``has_per_chain_L=True`` (mclmc_tuning / mclmc_lrd_tuning).

    Produces no ``_state_reinit`` — ``MCLMCState`` from warmup is directly
    usable for sampling.

    Notes
    -----
    - MCLMCInfo does NOT carry ``num_integration_steps``; ``acceptance_rate``
      and ``is_divergent`` are absent.  The postamble handles this via the
      resolver in ``_build_info_diagnostics_block`` (mclmc is not in
      ``_SAMPLERS_WITH_IS_DIVERGENT``).
    - ``blackjax.mclmc.init`` requires an ``rng_key`` (to generate the initial
      unit-vector momentum).  The warmup template handles init; the
      ``kernel_builder`` here is for the *sampling* phase only.
    """
    lines: list[str] = []
    a = lines.append

    a("# === SAMPLER: mclmc ===")
    a("# kernel_builder(step_size, imm, L) wraps blackjax.mclmc.")
    a("# L is the third arg (per-chain trajectory length from mclmc_tuning warmup).")
    a(
        "# MCLMCInfo._fields = ('logdensity', 'kinetic_change', 'energy_change', 'nonans')"
    )
    a("# No is_divergent / acceptance_rate / num_integration_steps fields.")
    a(
        "import blackjax.mcmc.mclmc  # noqa: F401 (needed for init; sampler uses blackjax.mclmc)"
    )
    a("")
    a("")
    a("def kernel_builder(step_size, imm, L=None):")
    a('    """Build an mclmc step function for the given (step_size, imm, L)."""')
    a("    return blackjax.mclmc(")
    a("        logdensity_fn,")
    a("        step_size=step_size,")
    a("        inverse_mass_matrix=imm,")
    a("        L=L if L is not None else 1.0,")
    a("    ).step")
    a("")
    a("")
    a("try:")
    a("    _state_post_warmup")
    a("except NameError:")
    a("    # no_warmup path: init from init_position with a fixed key.")
    a("    _warmup_init_is_single_chain = True")
    a("    _no_warmup_init_key = jax.random.key(0)")
    a("    _state_post_warmup = blackjax.mclmc(")
    a("        logdensity_fn,")
    a("        step_size=1.0,")
    a("        inverse_mass_matrix=1.0,")
    a("        L=1.0,")
    a("    ).init(init_position, _no_warmup_init_key)")

    return "\n".join(lines)
