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

from tuningfork.recipes._emit import (
    emit_laplace_preamble,
    emit_postamble,
    emit_preamble,
)

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
# T1.4: resolved at emit time — no_warmup path omits block_until_ready
# (no _state_post_warmup exists before the sampler template sets it).
# Non-no_warmup path emits block_until_ready directly (no try/except).
_WARMUP_TIMING_BLOCK_WARMUP = (
    "# --- warmup timing fence ---\n"
    "jax.block_until_ready(_state_post_warmup)\n"
    "_warmup_wall = _recipe_time.perf_counter() - _warmup_t0\n"
    "_warmup_t1 = _recipe_time.perf_counter()\n"
)
_WARMUP_TIMING_BLOCK_NO_WARMUP = (
    "# --- warmup timing fence (no_warmup: state set by sampler template below) ---\n"
    "_warmup_wall = _recipe_time.perf_counter() - _warmup_t0\n"
    "_warmup_t1 = _recipe_time.perf_counter()\n"
)

# Sentinel set of samplers that define _state_reinit (require state type change
# after warmup). Resolved at emit time — no try/except NameError needed.
_STATE_REINIT_SAMPLERS = frozenset(
    {"dynamic_hmc", "dmhmc", "ghmc", "laplace_dhmc", "laplace_dmhmc"}
)

# T1.5: Info-field sets per sampler — resolved at emit time.
# is_divergent: HMC-family only.
_SAMPLERS_WITH_IS_DIVERGENT = frozenset(
    {
        "nuts",
        "hmc",
        "mhmc",
        "dmhmc",
        "dynamic_hmc",
        "ghmc",
        "rmhmc",
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
    }
)
# acceptance_rate: all MCMC except VI; VI only has elbo.
_VI_SAMPLER_NAMES = frozenset({"meanfield_vi", "fullrank_vi"})
# is_accepted (in addition to acceptance_rate): HMC + MH-family except pure NUTS.
# NUTS has acceptance_rate but not is_accepted.
_SAMPLERS_WITH_IS_ACCEPTED = frozenset(
    {
        "hmc",
        "mhmc",
        "dmhmc",
        "dynamic_hmc",
        "ghmc",
        "rmhmc",
        "mala",
        "barker",
        "rwm",
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
    }
)
# Per-step stats to persist: vary by sampler.
# All HMC-family have is_divergent + energy; NUTS also has num_integration_steps.
# acceptance_rate and is_accepted vary; VI has none of these.
_SAMPLERS_WITH_NIS_STAT = frozenset(
    {"nuts", "hmc", "mhmc", "dmhmc", "dynamic_hmc", "ghmc", "rmhmc"}
)


def _build_info_diagnostics_block(sampler_name: str) -> str:
    """T1.5: build resolved info-diagnostics block for the postamble.

    Replaces the hasattr(_infos, ...) probes with straight-line code per
    sampler family.
    """
    lines = [
        '_acceptance = float("nan")',
        "_n_div = 0",
    ]
    if sampler_name in _SAMPLERS_WITH_IS_DIVERGENT:
        lines.append("_n_div = int(jnp.sum(_infos.is_divergent))")
    if sampler_name not in _VI_SAMPLER_NAMES:
        lines.append("_acceptance = float(jnp.mean(_infos.acceptance_rate))")
    return "\n".join(lines)


def _build_draws_ss_block(sampler_name: str) -> str:
    """T1.5: build resolved per-step sample-stats block for the draws persistence.

    Replaces the hasattr(_infos, _ss_field) loop with explicit field access
    per sampler family.
    """
    if sampler_name in _VI_SAMPLER_NAMES:
        # VI samplers have no MCMC diagnostics to persist.
        return "    # VI sampler: no per-step MCMC stats (only elbo in info)."

    fields: list[str] = []
    if sampler_name in _SAMPLERS_WITH_IS_DIVERGENT:
        fields.append("is_divergent")
        fields.append("energy")
    if sampler_name in _SAMPLERS_WITH_NIS_STAT:
        fields.append("num_integration_steps")
    fields.append("acceptance_rate")
    if sampler_name in _SAMPLERS_WITH_IS_ACCEPTED:
        fields.append("is_accepted")

    lines = []
    for field in fields:
        lines.append(f'    _draws_dict["_ss_{field}"] = np.asarray(_infos.{field})')
    return (
        "\n".join(lines) if lines else "    pass  # no per-step stats for this sampler"
    )


def _strip_no_warmup_try_block(sampler_body: str) -> str:
    """T1.3: strip the per-sampler ``try: _state_post_warmup / except NameError:``
    block from the emitted sampler body for non-no_warmup recipes.

    All 15 non-nuts sampler templates contain a block of the form::

        try:
            _state_post_warmup
        except NameError:
            _warmup_init_is_single_chain = True   # (optional)
            _state_post_warmup = ...              # init from init_position

    For non-no_warmup recipes this entire block is dead code (the warmup
    template already set ``_state_post_warmup`` before the sampler section
    runs). Strip it here so the emitted script is straight-line.

    The try-block sentinel is ``try:\\n    _state_post_warmup``, which is
    unique — no other try/except probes ``_state_post_warmup`` directly.

    Returns the sampler body with the dead block removed.
    """
    import re

    # Match: optional leading blank line, then
    # "try:\n    _state_post_warmup\nexcept NameError:\n" +
    # all indented continuation lines.
    pattern = re.compile(
        r"\ntry:\n    _state_post_warmup\nexcept NameError:\n"
        r"(?:    [^\n]*\n)*"  # one or more indented lines
    )
    return pattern.sub("\n", sampler_body)


def _build_inference_loop(
    *,
    num_samples: int,
    sampler_seed: int,
    tuning_seed: int,
    num_chains: int,
    sampling_pb: bool,
    warmup_is_perchain: bool,
    warmup_init_is_single_chain: bool,
    needs_state_reinit: bool,
) -> str:
    """Build straight-line inference loop code with all branches resolved at emit time.

    T1.1: replaces inference_loop.py.tmpl + inference_loop_singlechain.py.tmpl.
    No try/except NameError probes — every flag is known at generation time.

    Parameters
    ----------
    sampling_pb : bool
        If True → single-chain loop (progress_bar=True, io_callback safe).
        If False → multi-chain loop (scan + vmap, no progress bar).
    warmup_is_perchain : bool
        Warmup ran per-chain (jax.vmap). Adapted params are (num_chains, ...).
    warmup_init_is_single_chain : bool
        no_warmup path. State initialised by sampler template; needs broadcast.
    needs_state_reinit : bool
        Sampler (dynamic_hmc / dmhmc / ghmc / laplace_dhmc / laplace_dmhmc)
        needs a different state type — per-chain re-init required.
    """
    lines: list[str] = []
    a = lines.append

    a(f"_NUM_SAMPLES = {num_samples}")

    if sampling_pb:
        # ── Single-chain path ────────────────────────────────────────────────
        a(
            "_SAMPLING_PROGRESS_BAR = True"
            "  # single-chain (progress bar safe); set False for multi-chain"
        )
        a("# Single-chain sampling (progress_bar=True).")
        a(
            "# progress_bar uses io_callback inside the scan body.  io_callback is"
            " not"
        )
        a(
            "# supported inside jax.vmap, so multi-chain sampling cannot use a"
            " progress bar."
        )
        a(
            "# We sample ONE chain then re-add a leading axis of 1 so downstream"
            " consumers"
        )
        a("# see shape (1, num_samples, ...) regardless of num_chains.")
        a("")

        # Step-size / IMM resolution
        if warmup_init_is_single_chain:
            # no_warmup: broadcast state, set defaults
            a(
                "_state_post_warmup = jax.tree.map("
                "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
                " _state_post_warmup)"
            )
            a('_shared_step_size = float(_adapted_params.get("step_size", 1.0))')
            a("from jax.flatten_util import ravel_pytree as _il_ravel")
            a("_il_flat, _ = _il_ravel(init_position)")
            a("_n_dims = int(_il_flat.shape[0])")
            a("_shared_imm = jnp.ones(_n_dims)")
        elif warmup_is_perchain:
            # Per-chain warmup — use per-chain params, pick chain-0 for single-chain run
            a('_shared_step_size = _adapted_params["step_size"][0]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"][0]')
        else:
            # Single-chain warmup → scalar params
            a('_shared_step_size = _adapted_params["step_size"]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"]')

        a("")
        a(
            "from blackjax.util import run_inference_algorithm as _run_inference_algorithm"
        )
        a("from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm")
        a("")
        a("# Extract single-chain state (chain 0).")
        a("_single_chain_state = jax.tree.map(lambda x: x[0], _state_post_warmup)")

        if needs_state_reinit:
            a("")
            a("# Re-init per sampler state type (dynamic_hmc / dmhmc / ghmc).")
            a(f"_single_reinit_key = jax.random.key({tuning_seed + 998})")
            a(
                "_single_chain_state = _state_reinit("
                "_shared_step_size, _shared_imm,"
                " _single_chain_state.position, _single_reinit_key)"
            )

        a("")
        a("_single_chain_step = kernel_builder(_shared_step_size, _shared_imm)")
        a(
            "_sc_alg = _SamplingAlgorithm(lambda *args, **kwargs: None,"
            " _single_chain_step)"
        )
        a("")
        a(
            "_sc_final_state, (_sc_states_hist, _sc_infos_hist) ="
            " _run_inference_algorithm("
        )
        a(f"    jax.random.key({sampler_seed}),")
        a("    _sc_alg,")
        a("    num_steps=_NUM_SAMPLES,")
        a("    initial_state=_single_chain_state,")
        a("    progress_bar=True,")
        a(")")
        a("")
        a(
            "# Re-add the leading chain axis (size 1) for downstream shape"
            " consistency."
        )
        a("_samples = jax.tree.map(lambda x: x[None], _sc_states_hist)")
        a("_infos = jax.tree.map(lambda x: x[None], _sc_infos_hist)")

    else:
        # ── Multi-chain path ─────────────────────────────────────────────────
        a(
            "_SAMPLING_PROGRESS_BAR = False"
            "  # multi-chain scan+vmap; set True for single-chain with progress bar"
        )
        a("")

        # Step-size / IMM resolution
        if warmup_init_is_single_chain:
            # no_warmup: broadcast state, set defaults
            a(
                "# no_warmup: broadcast init_position-derived state to (num_chains, ...)."
            )
            a(
                "_state_post_warmup = jax.tree.map("
                "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
                " _state_post_warmup)"
            )
            a('_shared_step_size = float(_adapted_params.get("step_size", 1.0))')
            a("from jax.flatten_util import ravel_pytree as _il_ravel")
            a("_il_flat, _ = _il_ravel(init_position)")
            a("_n_dims = int(_il_flat.shape[0])")
            a("_shared_imm = jnp.ones(_n_dims)")
        elif not warmup_is_perchain:
            # Single-chain warmup → scalar shared params
            a("# Single-chain warmup: adapted params are scalar / un-batched.")
            a('_shared_step_size = _adapted_params["step_size"]')
            a('_shared_imm = _adapted_params["inverse_mass_matrix"]')

        if needs_state_reinit:
            a("")
            a(
                "# Re-init per-chain state (dynamic_hmc / dmhmc / ghmc: different state"
                " type than warmup)."
            )
            a(
                f"_reinit_keys = jax.random.split(jax.random.key({tuning_seed + 999}),"
                f" num_chains)"
            )
            if warmup_is_perchain:
                a('_batched_step_size = _adapted_params["step_size"]')
                a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
                a(
                    "_state_post_warmup = jax.vmap("
                    "lambda s, k, ss, imm: _state_reinit(ss, imm, s.position, k)"
                    ")(_state_post_warmup, _reinit_keys, _batched_step_size,"
                    " _batched_imm)"
                )
            else:
                a(
                    "_state_post_warmup = jax.vmap("
                    "lambda s, k: _state_reinit(_shared_step_size, _shared_imm,"
                    " s.position, k)"
                    ")(_state_post_warmup, _reinit_keys)"
                )

        a("")
        a("# Build the vmapped step function.")
        a(
            "from blackjax.util import run_inference_algorithm as _run_inference_algorithm"
        )
        a("from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm")
        a("")

        if warmup_is_perchain:
            a("# Per-chain warmup: each chain gets its own (step_size, imm).")
            a('_batched_step_size = _adapted_params["step_size"]')
            a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
            a("")
            a("def _step_one_chain(state, key, step_size, imm):")
            a("    return kernel_builder(step_size, imm)(key, state)")
            a("")
            a("def _vmapped_step(rng_key, states):")
            a(f"    keys = jax.random.split(rng_key, {num_chains})")
            a(
                "    return jax.vmap(_step_one_chain)(states, keys, _batched_step_size,"
                " _batched_imm)"
            )
        else:
            a("# Single-chain warmup: shared step_size + IMM across all chains.")
            a("_kernel_step = kernel_builder(_shared_step_size, _shared_imm)")
            a("")
            a("def _vmapped_step(rng_key, states):")
            a(f"    keys = jax.random.split(rng_key, {num_chains})")
            a("    return jax.vmap(_kernel_step)(keys, states)")

        a("")
        a("# run_inference_algorithm: (num_steps, num_chains, ...) output.")
        a("_alg = _SamplingAlgorithm(lambda *args, **kwargs: None, _vmapped_step)")
        a("_final_state, (_states_hist, _infos_hist) = _run_inference_algorithm(")
        a(f"    jax.random.key({sampler_seed}),")
        a("    _alg,")
        a("    num_steps=_NUM_SAMPLES,")
        a("    initial_state=_state_post_warmup,")
        a("    progress_bar=False,")
        a(")")
        a("")
        a("# Swap axes: (num_steps, num_chains, ...) -> (num_chains, num_steps, ...).")
        a("_samples = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _states_hist)")
        a("_infos = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), _infos_hist)")

    return "\n".join(lines)


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
    # T1.5: resolve postamble info-diagnostics and draws-stats blocks at emit time.
    ctx["info_diagnostics_block"] = _build_info_diagnostics_block(
        recipe.base_method_name
    )
    ctx["draws_ss_block"] = _build_draws_ss_block(recipe.base_method_name)

    # T1.7: VI warmup + sampler slots (unified vi_warmup.py.tmpl + vi_sampler.py.tmpl).
    _VI_WARMUP_NAMES = frozenset({"meanfield_vi", "fullrank_vi"})
    _VI_SAMPLER_NAMES_TMPL = frozenset({"meanfield_vi", "fullrank_vi"})
    if (
        recipe.warmup_name in _VI_WARMUP_NAMES
        or recipe.base_method_name in _VI_SAMPLER_NAMES_TMPL
    ):
        if "meanfield" in recipe.warmup_name or "meanfield" in recipe.base_method_name:
            _vp = "_mf"
            ctx["vi_prefix"] = _vp
            ctx["vi_module"] = "blackjax.vi.meanfield_vi"
            ctx["vi_imm_description"] = (
                "diagonal IMM: exp(2*rho), rho encodes log-scale"
            )
            # Pre-resolve vi_prefix in extraction block (no nested template substitution).
            ctx["vi_imm_extraction_block"] = (
                f"{_vp}_rho_flat, _ = {_vp}_ravel({_vp}_final_vi_state.rho)\n"
                f"{_vp}_imm = jnp.exp(2.0 * {_vp}_rho_flat)  # shape (d,)"
            )
            ctx["vi_adapted_imm_expr"] = (
                f"jnp.broadcast_to(\n"
                f"        {_vp}_imm[None, :], (num_chains, _d)\n"
                f"    )"
            )
            ctx["vi_state_name"] = "_MFVISamplerState"
            ctx["vi_info_name"] = "MFVIInfo"
        else:  # fullrank
            _vp = "_fr"
            ctx["vi_prefix"] = _vp
            ctx["vi_module"] = "blackjax.vi.fullrank_vi"
            ctx["vi_imm_description"] = "dense IMM: L@L.T (Cholesky)"
            # Cholesky extraction — pre-resolve vi_prefix.
            ctx["vi_imm_extraction_block"] = (
                f"def {_vp}_unflatten_cholesky(chol_params, dim):\n"
                f"    tril = jnp.zeros((dim, dim))\n"
                f"    tril = tril.at[jnp.tril_indices(dim, k=-1)].set(chol_params[dim:])\n"
                f"    diag = jnp.exp(chol_params[:dim])\n"
                f"    return tril + jnp.diag(diag)\n"
                f"\n"
                f"\n"
                f"{_vp}_chol = {_vp}_unflatten_cholesky({_vp}_final_vi_state.chol_params, _d)\n"
                f"{_vp}_imm = {_vp}_chol @ {_vp}_chol.T  # shape (d, d)"
            )
            ctx["vi_adapted_imm_expr"] = (
                f"jnp.broadcast_to(\n"
                f"        {_vp}_imm[None, :, :], (num_chains, _d, _d)\n"
                f"    )"
            )
            ctx["vi_state_name"] = "_FRVISamplerState"
            ctx["vi_info_name"] = "FRVIInfo"

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
        # All _LAPLACE_OPTIMIZER_KWARG_NAMES keys from phase/recipe params are included.
        from tuningfork.recipes._recipe_runner import _extract_laplace_optimizer_kwargs

        def _laplace_factory_expr(opt_kwargs: dict) -> str:
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in opt_kwargs.items())
            sep = ", " if kwargs_str else ""
            return f"_lmf(log_joint_fn, theta_init{sep}{kwargs_str})"

        if _is_multiphase_warmup:
            _factory_exprs = []
            for _phase in recipe.warmups:
                _phase_opt_kwargs = _extract_laplace_optimizer_kwargs(
                    _phase["params"], recipe.base_method_params
                )
                _factory_exprs.append(_laplace_factory_expr(_phase_opt_kwargs))
            ctx["laplace_factories_expr"] = ", ".join(_factory_exprs)
            ctx["num_warmup_phases"] = len(recipe.warmups)
        else:
            # Single-phase: extract optimizer kwargs from warmup_params then bm fallback.
            _single_opt_kwargs = _extract_laplace_optimizer_kwargs(
                recipe.warmup_params, recipe.base_method_params
            )
            ctx["laplace_factories_expr"] = _laplace_factory_expr(_single_opt_kwargs)
            ctx["num_warmup_phases"] = 1

        # Build a Python dict literal of optimizer kwargs for the sampler template.
        # Templates use $bm_optimizer_kwargs_expr to get all optimizer kwargs as a
        # dict they can spread: **_optimizer_kwargs.
        _bm_opt_kwargs = _extract_laplace_optimizer_kwargs(recipe.base_method_params)
        ctx["bm_optimizer_kwargs_expr"] = repr(_bm_opt_kwargs)

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
            # Full optimizer kwargs for per-phase notes (informational only in comment).
            _ph_opt = _extract_laplace_optimizer_kwargs(
                _phase_params, recipe.base_method_params
            )
            ctx[f"{_prefix}optimizer_kwargs"] = _ph_opt
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
    preamble = emit_preamble(ctx)

    # Warmup variants that support a multi-chain (vmap) path when progress_bar=False.
    # T1.6: window_adaptation variants unified into 2 templates (singlechain +
    # multichain), parameterised by $window_adaptation_fn and
    # $window_adaptation_extra_kwargs.
    _MULTICHAIN_WARMUP_VARIANTS = frozenset(
        {
            "window_adaptation_diag_imm",
            "window_adaptation_dense_imm",
            "window_adaptation_low_rank_imm",
        }
    )
    if recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS:
        if recipe.warmup_name == "window_adaptation_diag_imm":
            ctx["window_adaptation_fn"] = "blackjax.window_adaptation"
            ctx["window_adaptation_extra_kwargs"] = ""
        elif recipe.warmup_name == "window_adaptation_dense_imm":
            ctx["window_adaptation_fn"] = "blackjax.window_adaptation"
            ctx["window_adaptation_extra_kwargs"] = "is_mass_matrix_diagonal=False,"
        else:  # window_adaptation_low_rank_imm
            ctx["window_adaptation_fn"] = "blackjax.window_adaptation_low_rank"
            # T0.2 guard: max_rank must be present for low_rank recipes.
            # Resolved to its actual value here rather than leaving as $wp_max_rank
            # (a nested slot inside a slot value is never re-substituted).
            _max_rank = recipe.warmup_params.get("max_rank")
            if _max_rank is None:
                raise ValueError(
                    "window_adaptation_low_rank_imm recipe is missing 'max_rank' in "
                    "warmup_params. Add max_rank=<int> to the recipe's warmup_params "
                    "before calling emit_script."
                )
            ctx["window_adaptation_extra_kwargs"] = f"max_rank={_max_rank},"

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
    # window_adaptation variants use the unified templates (T1.6);
    # other warmups use their own templates.
    if _is_laplace and _is_multiphase_warmup:
        # Multi-phase laplace: dedicated template (already single-chain + broadcast
        # by design); warmup_num_chains doesn't affect template selection here.
        warmup_body = _load_template(
            "warmups/laplace_multiphase_warmup.py.tmpl"
        ).safe_substitute(ctx)
    elif (
        not _is_laplace
        and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS
        and not (_warmup_W0 == 1)
        and not _warmup_pb
    ):
        # progress_bar=False (and warmup_num_chains != 1) → multi-chain warmup.
        # T1.6: use unified multichain template.
        warmup_body = _load_template(
            "warmups/window_adaptation_multichain.py.tmpl"
        ).safe_substitute(ctx)
    elif recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS:
        # progress_bar=True or warmup_num_chains=1 → single-chain warmup.
        # T1.6: use unified singlechain template.
        warmup_body = _load_template(
            "warmups/window_adaptation.py.tmpl"
        ).safe_substitute(ctx)
    elif recipe.warmup_name in _VI_WARMUP_NAMES:
        # T1.7: unified VI warmup template.
        warmup_body = _load_template("warmups/vi_warmup.py.tmpl").safe_substitute(ctx)
    else:
        # Non-window_adaptation, non-VI warmup (no_warmup, pathfinder, etc.).
        warmup_body = _load_template(
            f"warmups/{recipe.warmup_name}.py.tmpl"
        ).safe_substitute(ctx)

    if recipe.base_method_name in _VI_SAMPLER_NAMES_TMPL:
        # T1.7: unified VI sampler template.
        sampler_body = _load_template("samplers/vi_sampler.py.tmpl").safe_substitute(
            ctx
        )
    else:
        sampler_body = _load_template(
            f"samplers/{recipe.base_method_name}.py.tmpl"
        ).safe_substitute(ctx)

    # T1.3: strip the try/except NameError _state_post_warmup block from
    # sampler templates for non-no_warmup recipes (dead code for those paths).
    # For no_warmup, the block is the initialization path and must be kept.
    if recipe.warmup_name != "no_warmup":
        sampler_body = _strip_no_warmup_try_block(sampler_body)

    # T1.1: resolve the 3 inference-loop sentinels at emit time.
    # All three flags are statically determined from (warmup_name, warmup_template_path,
    # base_method_name) — no runtime try/except NameError probes needed.
    #
    # _warmup_init_is_single_chain: True iff warmup_name == "no_warmup"
    _resolved_warmup_init_is_single_chain = recipe.warmup_name == "no_warmup"

    # _warmup_is_perchain: True iff the selected warmup template sets it True.
    # Specifically: multichain window_adaptation templates + VI warmups.
    _VI_WARMUP_NAMES = frozenset({"meanfield_vi", "fullrank_vi"})
    _uses_multichain_warmup_tmpl = (
        not _is_laplace
        and not _is_multiphase_warmup
        and not _resolved_warmup_init_is_single_chain
        and not (_warmup_W0 == 1 and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS)
        and (not _warmup_pb)
        and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS
    )
    _resolved_warmup_is_perchain = (
        _uses_multichain_warmup_tmpl or recipe.warmup_name in _VI_WARMUP_NAMES
    )

    # _needs_state_reinit: True iff sampler template defines _state_reinit.
    _resolved_needs_state_reinit = recipe.base_method_name in _STATE_REINIT_SAMPLERS

    # T1.4: emit timing block without try/except — no_warmup path omits
    # block_until_ready (state not yet set at this point in assembly).
    _timing_block = (
        _WARMUP_TIMING_BLOCK_NO_WARMUP
        if _resolved_warmup_init_is_single_chain
        else _WARMUP_TIMING_BLOCK_WARMUP
    )

    # T1.1: build straight-line inference loop (no try/except NameError probes).
    inference_loop = _build_inference_loop(
        num_samples=num_samples,
        sampler_seed=sampler_seed,
        tuning_seed=recipe.tuning_seed,
        num_chains=num_chains,
        sampling_pb=_sampling_pb,
        warmup_is_perchain=_resolved_warmup_is_perchain,
        warmup_init_is_single_chain=_resolved_warmup_init_is_single_chain,
        needs_state_reinit=_resolved_needs_state_reinit,
    )

    postamble = emit_postamble(ctx)

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
        laplace_preamble = emit_laplace_preamble(ctx)
        return "\n\n".join(
            [
                preamble,
                laplace_preamble,
                warmup_body,
                _timing_block,
                sampler_body,
                inference_loop,
                postamble,
            ]
        )
    return "\n\n".join(
        [
            preamble,
            warmup_body,
            _timing_block,
            sampler_body,
            inference_loop,
            postamble,
        ]
    )
