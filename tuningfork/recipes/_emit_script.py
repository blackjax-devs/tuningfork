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
    ctx = {
        "recipe_id": (
            f"{recipe.model_name}/{recipe.effort.value}"
            f"__{recipe.base_method_name}__{recipe.warmup_name}"
        ),
        "x64_config_line": _x64_config_line,
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
        "n_warmup": recipe.warmup_params.get("n_warmup", 1000),
        # base_method_params unrolled (legacy top-level slots — backward compat)
        "max_num_doublings": recipe.base_method_params.get("max_num_doublings", 10),
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
        for _i, _phase in enumerate(recipe.warmups):
            _phase_params = _phase["params"]
            _prefix = f"wp{_i}_"
            ctx[f"{_prefix}name"] = _phase["name"]
            _phase_target = _phase_params.get(
                "target_acceptance", _phase_params.get("target_acceptance_rate", 0.8)
            )
            ctx[f"{_prefix}target"] = _phase_target
            ctx[f"{_prefix}n_warmup"] = _phase_params.get("n_warmup", 1000)
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

    # Build the warmup body: multi-phase laplace uses a dedicated template;
    # single-phase uses the per-warmup template (existing path).
    if _is_laplace and _is_multiphase_warmup:
        warmup_body = _load_template(
            "warmups/laplace_multiphase_warmup.py.tmpl"
        ).safe_substitute(ctx)
    else:
        warmup_body = _load_template(
            f"warmups/{recipe.warmup_name}.py.tmpl"
        ).safe_substitute(ctx)

    sampler_body = _load_template(
        f"samplers/{recipe.base_method_name}.py.tmpl"
    ).safe_substitute(ctx)
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
