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
"""Emit a recipe-reproduction script from a Recipe.

This is the entry point for recipe portability (Principle F): given a Recipe,
emit a Python script that reproduces the recipe's inference with the wiring
code visible inline.

Templates live in ``_templates/`` and use string.Template ($slot) substitution
because Python code contains curly braces that conflict with str.format.

Design decisions
----------------
- **D8 STRICT (clarified 2026-05-17 post R3.5-MVP)**: the **inference
  choreography** (warmup + sampler + inference loop) has zero ``import
  tuningfork`` — it's auditable in one file and shows the exact BlackJAX
  call shape.  The **model** is imported via ``from tuningfork.model import
  MODELS`` (canonical NumPyro code lives upstream; not duplicated here).
  This avoids template-drift risk on the largest, most-stable code surface
  while preserving the design-smell forcing function on the actual wiring
  layer (per Principle A — heavy sampler/warmup template = upstream BlackJAX
  design issue).
- **D9**: pure function — returns a string; no side effects.  The caller
  writes to whatever path they want.
- **D10**: hand-written templates + round-trip CI gate in
  ``tests/recipes/test_emit_script.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe

# The x64 config line is injected into the preamble template for models that
# require float64 (e.g., gp_regression — Cholesky NaN at float32).  Float32
# models must NOT get this line; the slot is left empty for them.
_X64_CONFIG_LINE = 'jax.config.update("jax_enable_x64", True)  # required by this model'
_X64_CONFIG_LINE_EMPTY = ""

# Warning text emitted when progress_bar=True: injected at the TOP of the generated
# preamble (before model build) AND issued at emit_script() call time.  Defined once
# here so both use sites share exactly the same text (no drift).
_PROGRESS_BAR_WARNING_TEXT = (
    "progress_bar=True forces SINGLE chain warmup and sampling — multi-chain runs "
    "under jax.vmap, which is incompatible with the progress bar's io_callback "
    "(see blackjax issue #927). For the full num_chains-chain run, set progress_bar=False."
)

# The warning block injected into the preamble template when progress_bar=True.
# Uses `warnings` (already imported by the preamble) and UserWarning for visibility.
_PROGRESS_BAR_WARNING_BLOCK = (
    "warnings.warn(\n"
    '    "progress_bar=True forces SINGLE chain warmup and sampling -- multi-chain runs "\n'
    '    "under jax.vmap, which is incompatible with the progress bar\'s io_callback "\n'
    '    "(see blackjax issue #927). For the full num_chains-chain run, set progress_bar=False.",\n'
    "    stacklevel=1,\n"
    ")"
)
_PROGRESS_BAR_WARNING_BLOCK_EMPTY = ""

# Timing block inserted between warmup_body and sampler_body.
# jax.block_until_ready forces async dispatch to complete before stamping
# _warmup_t1, giving an honest warmup wall time (not just dispatch time).
# The try/except NameError guard handles the no_warmup path, where
# _state_post_warmup is initialised inside the sampler template (not warmup);
# in that case _warmup_wall records only the trivial setup time (<1 ms).
_WARMUP_TIMING_BLOCK = (
    "# --- warmup timing fence ---\n"
    "try:\n"
    "    jax.block_until_ready(_state_post_warmup)\n"
    "except NameError:\n"
    "    pass\n"
    "_warmup_wall = _recipe_time.perf_counter() - _warmup_t0\n"
    "_warmup_t1 = _recipe_time.perf_counter()\n"
)


__all__ = ["emit_script"]

_TEMPLATES_DIR = Path(__file__).parent / "_templates"


def _load_template(relpath: str) -> Template:
    """Load a .py.tmpl file as a string.Template."""
    return Template((_TEMPLATES_DIR / relpath).read_text())


def _recipe_hash(recipe: Recipe) -> str:
    """SHA-1 of the canonical recipe JSON; first 12 chars."""
    payload = json.dumps(
        {
            "model_name": recipe.model_name,
            "base_method_name": recipe.base_method_name,
            "warmup_name": recipe.warmup_name,
            "effort": recipe.effort.value,
            "base_method_params": recipe.base_method_params,
            "warmup_params": recipe.warmup_params,
            "tuning_seed": recipe.tuning_seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def emit_script(
    recipe: Recipe,
    *,
    num_samples: int | None = None,
    sampler_seed: int | None = None,
    num_chains: int | None = None,
    num_warmup: int | list[int] | None = None,
    progress_bar: bool | None = None,
    warmup_num_chains: list[int] | None = None,
) -> str:
    """Assemble a recipe-reproduction Python script.

    Per locked decision D8 (STRICT inference, 2026-05-17 clarification),
    the emitted script's **inference choreography** (warmup + sampler +
    inference loop) has zero ``import tuningfork`` and is auditable inline.
    The **model definition** is imported via ``from tuningfork.model import
    MODELS`` — canonical NumPyro code lives upstream, not duplicated as
    a per-model template.

    Parameters
    ----------
    recipe : Recipe
        The recipe to emit. Loaded via :func:`tuningfork.catalog.load_recipe`.
    num_samples : int, optional
        Number of post-warmup samples to draw in the emitted inference loop.
        When ``None`` (default), reads ``recipe.calibration_budget["n_samples"]``
        (the validated config) and falls back to 1000 if not set.
        Pass an explicit integer to override (e.g. for a longer production run).
    sampler_seed : int, optional
        RNG seed for the post-warmup sampling. Defaults to
        ``recipe.tuning_seed + 1`` so the emitted script is deterministic
        given the recipe.
    num_chains : int, optional
        Number of chains for the vmap-scan inference loop. When ``None``,
        derived from the recipe: ``recipe.warmup_params.get("num_chains",
        recipe.calibration_budget.get("num_chains", 1))``. Falls back to 1
        for legacy groundtruth recipes that pre-date the ``num_chains`` field.
    num_warmup : int or list[int] or None, optional
        Override the warmup step count(s) in the emitted script.

        - ``None`` (default): use the recipe's stored warmup step counts
          (backward-compatible).
        - ``int``: applies to single-phase warmups.  Sets ``$n_warmup`` in the
          warmup template.
        - ``list[int]``: applies to multi-phase warmups (e.g.
          ``laplace_multiphase``).  One entry per phase; the list length MUST
          equal the number of warmup phases in ``recipe.warmups``; otherwise
          a ``ValueError`` is raised.  Maps onto ``$wp0_n_warmup``,
          ``$wp1_n_warmup``, … in the template.
        - An ``int`` passed for a multi-phase warmup is treated as a 1-element
          list; if there is more than one phase a ``ValueError`` is raised.

        Typical use: fast dry-run without editing the recipe.::

            emit_script(recipe, num_warmup=[100, 10], num_samples=100)
    progress_bar : bool or None, optional
        When not ``None``, overrides BOTH the warmup ``progress_bar=`` arguments
        AND the sampling ``_SAMPLING_PROGRESS_BAR`` constant in the emitted
        script.  When ``None`` (default), defaults are used: warmup
        ``progress_bar=True`` and sampling ``_SAMPLING_PROGRESS_BAR = True``.
    warmup_num_chains : list[int] or None, optional
        Runtime override for ``recipe.warmup_num_chains``.  Affects which warmup
        template variant is selected:

        - ``None`` (default): falls back to ``recipe.warmup_num_chains``, which
          is itself ``None`` for legacy recipes (uses the multichain template when
          ``progress_bar=False``).
        - ``[1]`` or all-ones list: forces the single-chain warmup template
          (``window_adaptation_*.py.tmpl``), regardless of ``progress_bar``.
          Recommended for expensive-logprob models to avoid vmap-of-while_loop.
        - ``[W]`` with ``W == num_chains``: same as ``None`` — uses the multichain
          template when ``progress_bar=False``.
        - ``[W]`` with ``W != num_chains``: uses the multichain template (vmap
          over W chains); the reduce+broadcast is handled by the emitted script's
          runner, not by template selection.

        Only the first entry is used for single-phase warmups; for multi-phase
        warmups (``laplace_multiphase``), the template is already single-chain
        by design and this argument has no effect.

    Returns
    -------
    str
        The full Python script content. The function is pure — no side effects.
        The caller writes the returned string to whatever path they want
        (per locked decision D9).

    Raises
    ------
    FileNotFoundError
        If a required template is missing for the given
        ``(model_name, warmup_name, base_method_name)`` combo.
    KeyError
        If the recipe's ``warmup_params`` or ``base_method_params`` lack a
        required slot for the template.
    ValueError
        If ``num_warmup`` is a list whose length does not match the number of
        warmup phases in ``recipe.warmups``.
    """
    if sampler_seed is None:
        sampler_seed = recipe.tuning_seed + 1

    # Resolve num_samples: prefer calibration_budget (validated config), then 1000.
    if num_samples is None:
        num_samples = int(recipe.calibration_budget.get("n_samples") or 1000)

    if num_chains is None:
        num_chains = recipe.warmup_params.get(
            "num_chains",
            recipe.calibration_budget.get("num_chains", 1),
        )

    # Resolve progress_bar overrides.
    # When None: defaults (warmup True, sampling True via _SAMPLING_PROGRESS_BAR).
    _warmup_pb = True if progress_bar is None else bool(progress_bar)
    _sampling_pb = True if progress_bar is None else bool(progress_bar)

    # Call-time warning: issued immediately when progress_bar=True (explicit)
    # so that callers using emit_script() in a notebook or script see the advisory
    # before the generated file is written, not only at script-execution time.
    # Not issued when progress_bar=None (the backward-compatible default) to avoid
    # breaking existing callers that never set this argument explicitly.
    if progress_bar is True:
        import warnings as _warnings

        _warnings.warn(_PROGRESS_BAR_WARNING_TEXT, stacklevel=2)

    # x64 requirement: look up the model in the registry and check requires_x64.
    # This mirrors the runner logic at _recipe_runner.py:551-554.
    # The x64 line must appear BEFORE any JAX computation (build_logdensity_fn,
    # jax.random.key, etc.), so it is injected into the preamble immediately
    # after ``import jax``.
    from tuningfork.model import MODELS as _MODELS

    _posterior_meta = _MODELS[recipe.model_name]
    _x64_config_line = (
        _X64_CONFIG_LINE if _posterior_meta.requires_x64 else _X64_CONFIG_LINE_EMPTY
    )

    # Normalise warmup_params key spelling: groundtruth recipes use
    # "target_acceptance" (legacy key from certify_reference.py);
    # newer recipe-generation code uses "target_acceptance_rate".
    target_acceptance_rate = recipe.warmup_params.get(
        "target_acceptance_rate",
        recipe.warmup_params.get("target_acceptance", 0.8),
    )

    # Substitution context — every $slot the templates reference must be here.
    #
    # Prefix convention (Option A — programmatic spread, R3.5b):
    #   bm_<key>  — from recipe.base_method_params  (e.g. $bm_step_size, $bm_num_integration_steps)
    #   wp_<key>  — from recipe.warmup_params        (e.g. $wp_n_warmup, $wp_target_acceptance_rate)
    #
    # These prefixed slots are in addition to the hand-unrolled top-level slots
    # (which remain for backward compatibility with existing templates).
    # New templates should prefer the prefixed $bm_* / $wp_* form so the context
    # auto-expands when new hyperparameter fields are added to recipes.
    # Resolve n_warmup for single-phase warmups.
    # num_warmup override takes precedence; recipe value is the fallback.
    _n_warmup_recipe = recipe.warmup_params.get("n_warmup", 1000)
    if num_warmup is None:
        _n_warmup_resolved = _n_warmup_recipe
    elif isinstance(num_warmup, int):
        _n_warmup_resolved = num_warmup
    elif isinstance(num_warmup, list):
        # list for single-phase: must be length 1 (or raise)
        n_phases = len(recipe.warmups) if recipe.warmups else 1
        if n_phases == 1:
            if len(num_warmup) != 1:
                raise ValueError(
                    f"num_warmup list length {len(num_warmup)} does not match "
                    f"single-phase warmup (expected 1 entry). "
                    f"For single-phase warmups, pass an int or a 1-element list."
                )
            _n_warmup_resolved = num_warmup[0]
        else:
            # Multi-phase: validate length matches phases; $n_warmup uses first entry
            # (for template templates that still reference it), per-phase uses $wpN_n_warmup.
            if len(num_warmup) != n_phases:
                raise ValueError(
                    f"num_warmup list length {len(num_warmup)} does not match "
                    f"number of warmup phases {n_phases} in recipe.warmups. "
                    f"Provide exactly {n_phases} entries (one per phase)."
                )
            _n_warmup_resolved = num_warmup[0]  # fallback for $n_warmup slot
    else:
        raise TypeError(
            f"num_warmup must be int, list[int], or None; got {type(num_warmup).__name__}"
        )

    # Progress-bar warning block for preamble: emitted at the TOP of the generated
    # file when progress_bar=True so users see the advisory before any computation.
    _pb_warning_block = (
        _PROGRESS_BAR_WARNING_BLOCK if _warmup_pb else _PROGRESS_BAR_WARNING_BLOCK_EMPTY
    )

    ctx = {
        "recipe_id": (
            f"{recipe.model_name}/{recipe.effort.value}"
            f"__{recipe.base_method_name}__{recipe.warmup_name}"
        ),
        "x64_config_line": _x64_config_line,
        "progress_bar_warning_block": _pb_warning_block,
        "model_name": recipe.model_name,
        "base_method_name": recipe.base_method_name,
        "warmup_name": recipe.warmup_name,
        "effort": recipe.effort.value,
        "recipe_hash": _recipe_hash(recipe),
        "verdict": recipe.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN"),
        "tuning_seed": recipe.tuning_seed,
        "sampler_seed": sampler_seed,
        "num_samples": num_samples,
        "num_chains": num_chains,
        # warmup_params unrolled (legacy top-level slots — backward compat)
        "target_acceptance_rate": target_acceptance_rate,
        "n_warmup": _n_warmup_resolved,
        # base_method_params unrolled (legacy top-level slots — backward compat)
        "max_num_doublings": recipe.base_method_params.get("max_num_doublings", 10),
        # progress_bar overrides
        "warmup_progress_bar": _warmup_pb,
        "sampling_progress_bar": _sampling_pb,
    }
    # Programmatic spread: bm_<key> from base_method_params, wp_<key> from warmup_params.
    # Values are JSON-serialised scalar types (int/float/list); templates that need
    # them reference $bm_step_size, $bm_num_integration_steps, $wp_n_warmup, etc.
    ctx.update({f"bm_{k}": v for k, v in recipe.base_method_params.items()})
    ctx.update({f"wp_{k}": v for k, v in recipe.warmup_params.items()})

    # The warmup template needs to call the right blackjax algorithm. The recipe-
    # runner uses `resolve_warmup_algorithm` which substitutes `blackjax.nuts` for
    # the warmup-substitute family (laplace_*, dynamic_hmc, dmhmc — methods whose
    # interface doesn't compose with `blackjax.window_adaptation` directly:
    # laplace_* needs `log_joint_fn` + `theta_init`; dynamic_hmc / dmhmc need
    # `random_generator_arg` at warmup step). For all other samplers we use the
    # sampler's own factory (= `blackjax.<base_method_name>`). Reproduce that
    # selection here so the emitted script faithfully reproduces the runner's
    # warmup protocol.
    #
    # Schema extension (warmup_inner_kernel): `recipe.warmup_inner_kernel` overrides the implicit selection when
    # it is set AND differs from the implicit default.  This is the exact mirror of
    # `resolve_warmup_inner_kernel` in `_warmup_to_sampler_transform.py` so that
    # the emitted script is bit-faithful to what the runner did.
    from tuningfork.warmup._laplace_adapter import (
        LAPLACE_METHOD_NAMES,
        WARMUP_SUBSTITUTE_METHOD_NAMES,
    )

    _implicit_warmup_default = (
        "nuts"
        if recipe.base_method_name in WARMUP_SUBSTITUTE_METHOD_NAMES
        else recipe.base_method_name
    )
    if (
        recipe.warmup_inner_kernel is not None
        and recipe.warmup_inner_kernel != _implicit_warmup_default
    ):
        # Explicit override: the runner used a non-default inner kernel for
        # warmup (e.g. nuts driving an hmc recipe). Use it directly.
        _warmup_sampler = recipe.warmup_inner_kernel
    else:
        _warmup_sampler = _implicit_warmup_default
    ctx["warmup_algorithm"] = f"blackjax.{_warmup_sampler}"

    # The warmup template also needs to pass any kernel-construction kwargs
    # that the chosen blackjax algorithm requires beyond `logdensity_fn`,
    # `step_size`, and `inverse_mass_matrix` (which come from adaptation
    # itself). The recipe-runner injects these via
    # `default_value_for_space` on the base_method's HP space; the
    # substitute path (uses NUTS) needs no extra kwargs. Reproduce both
    # branches here so e.g. an mhmc warmup gets its required
    # `num_integration_steps` kwarg (without it, `blackjax.mhmc` raises
    # TypeError at warmup time).
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.calibration.tune import default_value_for_space

    _warmup_extra: dict[str, object]
    if (
        _warmup_sampler == "nuts"
        or recipe.base_method_name in WARMUP_SUBSTITUTE_METHOD_NAMES
    ):
        # NUTS (explicit or via substitute-family) picks its own trajectory
        # length and needs no extra kernel kwargs at warmup time.
        # Exception: laplace_* warmup inner kernels DO need num_integration_steps
        # because blackjax.laplace_hmc / laplace_mhmc require it. For single-phase
        # recipes, pull from warmup_params if present; for multi-phase, each phase
        # supplies its own via wp*_extra_kwargs (computed below).
        if (
            recipe.base_method_name in LAPLACE_METHOD_NAMES
            and recipe.warmup_inner_kernel
            in (
                "laplace_hmc",
                "laplace_mhmc",
            )
        ):
            _nis = recipe.warmup_params.get("num_integration_steps")
            _warmup_extra = {"num_integration_steps": _nis} if _nis is not None else {}
        else:
            _warmup_extra = {}
    else:
        _bm = BASE_METHODS[recipe.base_method_name]
        _warmup_extra = {}
        for _space in _bm.default_hp_space:
            if _space.name not in ("step_size", "inverse_mass_matrix"):
                _warmup_extra[_space.name] = default_value_for_space(_space)

    # Render as ", k1=v1, k2=v2" so the template can inject it after the
    # base kwargs without re-thinking comma placement. Empty for nuts
    # (which only adapts step_size + IMM; needs no extra kernel kwargs).
    _warmup_extra_str = "".join(f", {k}={v!r}" for k, v in _warmup_extra.items())
    ctx["warmup_extra_kwargs"] = _warmup_extra_str

    # ── Laplace-* recipe handling ────────────────────────────────────────────
    # For laplace_* samplers, the emitted script needs a laplace_preamble section
    # (phi/theta split, log_joint_fn, LaplaceMarginal factories) inserted between
    # the standard preamble and the warmup body.  This is D8 compliant: the
    # laplace_preamble only imports from blackjax (laplace_marginal_factory),
    # not from tuningfork.
    _is_laplace = recipe.base_method_name in LAPLACE_METHOD_NAMES
    _is_multiphase_warmup = len(recipe.warmups) > 1

    if _is_laplace:
        # Import the phi/theta split table from the recipe runner.
        from tuningfork.recipes._recipe_runner import _LAPLACE_PHI_THETA_SPLITS

        if recipe.model_name not in _LAPLACE_PHI_THETA_SPLITS:
            raise ValueError(
                f"laplace_* recipe requested for model {recipe.model_name!r} but no "
                "phi/theta split is registered in _LAPLACE_PHI_THETA_SPLITS. "
                "Add an entry before calling emit_script for this model."
            )
        phi_sites, theta_sites = _LAPLACE_PHI_THETA_SPLITS[recipe.model_name]
        ctx["phi_sites_repr"] = repr(phi_sites)
        ctx["theta_sites_repr"] = repr(theta_sites)

        # Build the LaplaceMarginal factory expression for each warmup phase.
        # Each phase may have a different maxiter (from phase.params.maxiter).
        # For single-phase recipes: warmup_params["maxiter"] or bm_maxiter fallback.
        # For multi-phase recipes: extract per-phase maxiter from recipe.warmups.
        def _laplace_factory_expr(maxiter: int) -> str:
            return f"_lmf(log_joint_fn, theta_init, maxiter={maxiter})"

        if _is_multiphase_warmup:
            _factory_exprs = []
            for _phase in recipe.warmups:
                _phase_maxiter = _phase["params"].get(
                    "maxiter", recipe.base_method_params.get("maxiter", 30)
                )
                _factory_exprs.append(_laplace_factory_expr(_phase_maxiter))
            ctx["laplace_factories_expr"] = ", ".join(_factory_exprs)
            ctx["num_warmup_phases"] = len(recipe.warmups)
        else:
            # Single-phase: use warmup_params["maxiter"] or bm_maxiter fallback.
            _single_maxiter = recipe.warmup_params.get(
                "maxiter", recipe.base_method_params.get("maxiter", 30)
            )
            ctx["laplace_factories_expr"] = _laplace_factory_expr(_single_maxiter)
            ctx["num_warmup_phases"] = 1

    # ── Multi-phase warmup slot population ───────────────────────────────────
    # For multi-phase laplace warmup (len(recipe.warmups) > 1), populate per-phase
    # template slots: $wp0_*, $wp1_*, etc.  These are consumed by the
    # laplace_multiphase_warmup.py.tmpl template.
    if _is_laplace and _is_multiphase_warmup:
        # Build per-phase num_warmup overrides: None → use recipe value per phase.
        _per_phase_n_warmup: list[int | None]
        if num_warmup is None:
            _per_phase_n_warmup = [None] * len(recipe.warmups)
        elif isinstance(num_warmup, list):
            _per_phase_n_warmup = list(num_warmup)
        else:
            # int passed for multi-phase: already caught above (single-phase only)
            _per_phase_n_warmup = [None] * len(recipe.warmups)

        for _i, _phase in enumerate(recipe.warmups):
            _phase_params = _phase["params"]
            _prefix = f"wp{_i}_"
            ctx[f"{_prefix}name"] = _phase["name"]
            _phase_target = _phase_params.get(
                "target_acceptance", _phase_params.get("target_acceptance_rate", 0.8)
            )
            ctx[f"{_prefix}target"] = _phase_target
            # num_warmup override for this phase (None → use recipe value).
            _phase_nw_override = _per_phase_n_warmup[_i]
            ctx[f"{_prefix}n_warmup"] = (
                _phase_nw_override
                if _phase_nw_override is not None
                else _phase_params.get("n_warmup", 1000)
            )
            ctx[f"{_prefix}maxiter"] = _phase_params.get(
                "maxiter", recipe.base_method_params.get("maxiter", 30)
            )
            # Per-phase warmup extra kwargs for the laplace inner kernel.
            # blackjax.laplace_hmc requires num_integration_steps at warmup time.
            _phase_nis = _phase_params.get("num_integration_steps")
            if _phase_nis is not None and _warmup_sampler in (
                "laplace_hmc",
                "laplace_mhmc",
            ):
                ctx[f"{_prefix}extra_kwargs"] = (
                    f", num_integration_steps={_phase_nis!r}"
                )
            else:
                ctx[f"{_prefix}extra_kwargs"] = ""

    # Use safe_substitute so templates with optional $bm_*/wp_* slots that are
    # absent from the recipe (e.g. $bm_num_integration_steps in a nuts recipe)
    # leave the slot as a literal dollar-prefixed string rather than raising
    # KeyError.  Each template is responsible for using only the slots that
    # actually exist for its algorithm family.
    preamble = _load_template("preamble.py.tmpl").safe_substitute(ctx)

    # Warmup variants that support a multi-chain (vmap) path when progress_bar=False.
    # The single-chain templates (existing) include a warnings.warn() block that fires
    # when progress_bar=True.  The multi-chain (_multichain) templates run
    # jax.vmap(warmup.run) over num_chains so each chain gets its own adapted params;
    # they require progress_bar=False because io_callback is unsupported in vmap.
    _MULTICHAIN_WARMUP_VARIANTS = frozenset(
        {
            "window_adaptation_diag_imm",
            "window_adaptation_dense_imm",
            "window_adaptation_low_rank_imm",
        }
    )

    # Resolve effective warmup_num_chains: call-time override wins over recipe-stamped.
    # None → fall back to recipe.warmup_num_chains (may also be None for legacy recipes).
    _wnc_emit: list[int] | None = (
        warmup_num_chains if warmup_num_chains is not None else recipe.warmup_num_chains
    )
    # For single-phase warmups, the first (and only) entry drives template selection:
    # W == 1 → force single-chain template (ignore progress_bar for template selection).
    # W == num_chains → use existing multichain/single-chain logic.
    # W != num_chains but W > 1 → use multichain template (vmap over W; reduce+broadcast
    # is the runner's concern, not the template's).
    _warmup_W0 = _wnc_emit[0] if _wnc_emit is not None else None

    # Build the warmup body: multi-phase laplace uses a dedicated template;
    # single-phase uses the per-warmup template, branching on progress_bar for
    # warmup variants that support multi-chain execution.
    if _is_laplace and _is_multiphase_warmup:
        # Multi-phase laplace: dedicated template (already single-chain + broadcast
        # by design); warmup_num_chains doesn't affect template selection here.
        warmup_body = _load_template(
            "warmups/laplace_multiphase_warmup.py.tmpl"
        ).safe_substitute(ctx)
    elif _warmup_W0 == 1 and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS:
        # warmup_num_chains[0]=1 → force single-chain template regardless of
        # progress_bar. Mirrors the laplace_multiphase_warmup pattern: one warmup
        # chain, broadcast to num_chains for sampling.
        warmup_body = _load_template(
            f"warmups/{recipe.warmup_name}.py.tmpl"
        ).safe_substitute(ctx)
    elif not _warmup_pb and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS:
        # progress_bar=False → multi-chain warmup via jax.vmap(warmup.run).
        # Per-chain adapted params; _warmup_is_perchain=True set in the template.
        warmup_body = _load_template(
            f"warmups/{recipe.warmup_name}_multichain.py.tmpl"
        ).safe_substitute(ctx)
    else:
        # progress_bar=True (default) or non-vmap warmup variant →
        # single-chain warmup with broadcast + warnings.warn() in template.
        warmup_body = _load_template(
            f"warmups/{recipe.warmup_name}.py.tmpl"
        ).safe_substitute(ctx)

    sampler_body = _load_template(
        f"samplers/{recipe.base_method_name}.py.tmpl"
    ).safe_substitute(ctx)

    # Select inference loop template based on progress_bar:
    # - True  → single-chain (no jax.vmap; progress bar safe; warns about single-chain)
    # - False → multi-chain (jax.vmap over kernel step; no progress bar)
    # Legacy inference_loop.py.tmpl is the multi-chain path (backward compat).
    if _sampling_pb:
        inference_loop = _load_template(
            "inference_loop_singlechain.py.tmpl"
        ).safe_substitute(ctx)
    else:
        inference_loop = _load_template("inference_loop.py.tmpl").safe_substitute(ctx)

    postamble = _load_template("postamble.py.tmpl").safe_substitute(ctx)

    # Assembly order:
    # - Standard:  [preamble, warmup_body, sampler_body, inference_loop, postamble]
    # - Laplace:   [preamble, laplace_preamble, warmup_body, sampler_body, ...]
    #
    # The laplace_preamble is inserted after the standard preamble to:
    # 1. Split init_position into phi_init and theta_init
    # 2. Build log_joint_fn (wrapping the joint logdensity_fn from preamble)
    # 3. Build LaplaceMarginal factories for each warmup phase
    # 4. Override init_position and logdensity_fn for the warmup templates
    # Model definition is imported from tuningfork.model in the preamble;
    # no separate model template assembled here (post R3.5-MVP clarification).
    if _is_laplace:
        laplace_preamble = _load_template("laplace_preamble.py.tmpl").safe_substitute(
            ctx
        )
        return "\n\n".join(
            [
                preamble,
                laplace_preamble,
                warmup_body,
                _WARMUP_TIMING_BLOCK,
                sampler_body,
                inference_loop,
                postamble,
            ]
        )
    return "\n\n".join(
        [
            preamble,
            warmup_body,
            _WARMUP_TIMING_BLOCK,
            sampler_body,
            inference_loop,
            postamble,
        ]
    )
