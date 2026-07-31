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
  Opt-in tap instrumentation is imported from
  ``tuningfork.diagnostics._tap`` so its compatibility and artifact policy
  stays single-sourced.
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

from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from tuningfork._version import __version__
from tuningfork.recipes._emit import (
    EMITTABLE_WARMUP_NAMES,
    emit_diagnostics,
    emit_diagnostics_close,
    emit_init_strategy,
    emit_laplace_preamble,
    emit_postamble,
    emit_preamble,
    emit_sampler,
    emit_step_policy,
    emit_warmup,
)
from tuningfork.recipes._execution_manifest import ExecutionManifest
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._execution_telemetry import TELEMETRY_SCHEMA
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan
from tuningfork.recipes._sample_stats import SAMPLE_STAT_PREFIX, sample_stat_fields

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe

# The x64 config line is injected into the preamble template for models that
# require float64 (e.g., gp_regression — Cholesky NaN at float32).  Float32
# models must NOT get this line; the slot is left empty for them.
_X64_CONFIG_LINE = 'jax.config.update("jax_enable_x64", True)  # required by this model'
_X64_CONFIG_LINE_EMPTY = ""

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
    {
        "dynamic_hmc",
        "dmhmc",
        "ghmc",
        "adjusted_mclmc_dynamic",
        "laplace_dhmc",
        "laplace_dmhmc",
    }
)

_REPLAY_HMC_SAMPLERS = frozenset(
    {
        "nuts",
        "hmc",
        "mhmc",
        "rmhmc",
        "dynamic_hmc",
        "dmhmc",
        "ghmc",
        "barker",
        "mclmc",
        "adjusted_mclmc",
        "adjusted_mclmc_dynamic",
    }
)

_UNSUPPORTED_PINNED_REPLAY_SAMPLERS = frozenset(
    {"laplace_hmc", "laplace_mhmc", "laplace_dhmc", "laplace_dmhmc"}
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
    fields = sample_stat_fields(sampler_name)
    if "is_divergent" in fields:
        lines.append("_n_div = int(jnp.sum(_infos.is_divergent))")
    if "acceptance_rate" in fields:
        lines.append("_acceptance = float(jnp.mean(_infos.acceptance_rate))")
    return "\n".join(lines)


def _build_draws_ss_block(sampler_name: str) -> str:
    """T1.5: build resolved per-step sample-stats block for the draws persistence.

    Replaces the hasattr(_infos, _ss_field) loop with explicit field access
    per sampler family.
    """
    fields = sample_stat_fields(sampler_name)
    if not fields:
        return "    # VI sampler: no per-step MCMC stats (only elbo in info)."

    lines = []
    for field in fields:
        lines.append(
            f'    _draws_dict["{SAMPLE_STAT_PREFIX}{field}"] = '
            f"np.asarray(_infos.{field})"
        )
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
    reinit_seed: int,
    num_chains: int,
    use_progress_bar: bool,
    warmup_is_perchain: bool,
    warmup_init_is_single_chain: bool,
    warmup_init_is_prebatched: bool = False,
    needs_state_reinit: bool,
    has_per_chain_L: bool = False,
    no_warmup_step_size_expr: str = 'float(_adapted_params.get("step_size", 1.0))',
    no_warmup_imm_expr: str = "jnp.ones(_n_dims)",
    no_warmup_L_expr: str = "1.0",
) -> str:
    """Build straight-line inference loop code with all branches resolved at emit time.

    T1.1: replaces inference_loop.py.tmpl + inference_loop_singlechain.py.tmpl.
    No try/except NameError probes — every flag is known at generation time.

    Stage 2 (blackjax #964): topology is now unconditionally multi-chain
    (scan + vmap) — the old single-chain sampling path existed solely to keep
    the legacy io_callback-based progress bar off jax.vmap (#927); blackjax's
    new progress_bar() context manager is vmap-safe, so that constraint no
    longer applies and the single-chain branch is gone.

    Parameters
    ----------
    has_per_chain_L : bool
        When True (mclmc_tuning / mclmc_lrd_tuning warmups), also extract and
        vmap over ``_adapted_params["L"]`` alongside step_size and imm.
        ``kernel_builder`` receives a third positional argument ``L``.
    use_progress_bar : bool
        If True, wrap the ``run_inference_algorithm`` call in
        ``with blackjax.progress_bar():``.
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
    if use_progress_bar:
        a(f"_SAMPLING_PROGRESS_BAR = {use_progress_bar}  # see the with-block below")
    else:
        a(f"_SAMPLING_PROGRESS_BAR = {use_progress_bar}  # set True for a progress bar")
    a("")

    # Step-size / IMM resolution
    if warmup_init_is_prebatched:
        a("# no_warmup: init strategy already supplied one position per chain.")
        a("from jax.flatten_util import ravel_pytree as _il_ravel")
        a("_il_flat, _ = _il_ravel(init_position)")
        a("_n_dims = int(_il_flat.shape[-1])")
        a(
            f"_batched_step_size = jnp.broadcast_to({no_warmup_step_size_expr}, (num_chains,))"
        )
        a(f"_shared_imm = {no_warmup_imm_expr}")
        a(
            "_batched_imm = jax.tree.map("
            "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
            " _shared_imm)"
        )
        if has_per_chain_L:
            a(f"_batched_L = jnp.broadcast_to({no_warmup_L_expr}, (num_chains,))")
        a(f"_init_keys = jax.random.split(jax.random.key({reinit_seed}), num_chains)")
        a("_state_post_warmup = jax.vmap(_state_init)(init_position, _init_keys)")
    elif warmup_init_is_single_chain:
        # no_warmup: broadcast state, set defaults
        a("# no_warmup: broadcast init_position-derived state to (num_chains, ...).")
        a(
            "_state_post_warmup = jax.tree.map("
            "lambda x: jnp.broadcast_to(x[None], (num_chains,) + x.shape),"
            " _state_post_warmup)"
        )
        a(f"_shared_step_size = {no_warmup_step_size_expr}")
        a("from jax.flatten_util import ravel_pytree as _il_ravel")
        a("_il_flat, _ = _il_ravel(init_position)")
        a("_n_dims = int(_il_flat.shape[0])")
        a(f"_shared_imm = {no_warmup_imm_expr}")
    elif not warmup_is_perchain:
        # Single-chain warmup → scalar shared params
        a("# Single-chain warmup: adapted params are scalar / un-batched.")
        a('_shared_step_size = _adapted_params["step_size"]')
        a('_shared_imm = _adapted_params["inverse_mass_matrix"]')

    if needs_state_reinit and not warmup_init_is_prebatched:
        a("")
        a(
            "# Re-init per-chain state (dynamic_hmc / dmhmc / ghmc: different state"
            " type than warmup)."
        )
        a(
            f"_reinit_keys = jax.random.split(jax.random.key({reinit_seed}),"
            f" num_chains)"
        )
        if warmup_is_perchain and not warmup_init_is_prebatched:
            a('_batched_step_size = _adapted_params["step_size"]')
            a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
            if has_per_chain_L:
                a('_batched_L = _adapted_params["L"]')
                a(
                    "_state_post_warmup = jax.vmap("
                    "lambda s, k, ss, imm, L: _state_reinit(ss, imm, s.position, k, L)"
                    ")(_state_post_warmup, _reinit_keys, _batched_step_size,"
                    " _batched_imm, _batched_L)"
                )
            else:
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
    a("from blackjax.util import run_inference_algorithm as _run_inference_algorithm")
    a("from blackjax.base import SamplingAlgorithm as _SamplingAlgorithm")
    a("")

    if warmup_is_perchain and has_per_chain_L:
        # MCLMC per-chain warmup: each chain gets its own (step_size, imm, L).
        a("# Per-chain warmup (mclmc): each chain gets its own (step_size, imm, L).")
        if not warmup_init_is_prebatched:
            a('_batched_step_size = _adapted_params["step_size"]')
            a('_batched_imm = _adapted_params["inverse_mass_matrix"]')
            a('_batched_L = _adapted_params["L"]')
        a("")
        a("def _step_one_chain(state, key, step_size, imm, L):")
        a("    return kernel_builder(step_size, imm, L)(key, state)")
        a("")
        a("def _vmapped_step(rng_key, states):")
        a(f"    keys = jax.random.split(rng_key, {num_chains})")
        a(
            "    return jax.vmap(_step_one_chain)("
            "states, keys, _batched_step_size, _batched_imm, _batched_L)"
        )
    elif warmup_is_perchain:
        a("# Per-chain warmup: each chain gets its own (step_size, imm).")
        if not warmup_init_is_prebatched:
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
    if use_progress_bar:
        a('with blackjax.progress_bar(label="sampling"):')
        a("    _final_state, (_states_hist, _infos_hist) = _run_inference_algorithm(")
        a(f"        jax.random.key({sampler_seed}),")
        a("        _alg,")
        a("        num_steps=_NUM_SAMPLES,")
        a("        initial_state=_state_post_warmup,")
        a("    )")
    else:
        a("_final_state, (_states_hist, _infos_hist) = _run_inference_algorithm(")
        a(f"    jax.random.key({sampler_seed}),")
        a("    _alg,")
        a("    num_steps=_NUM_SAMPLES,")
        a("    initial_state=_state_post_warmup,")
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


def emit_script(
    recipe: Recipe,
    *,
    tuning_seed: int | None = None,
    num_samples: int | None = None,
    sampler_seed: int | None = None,
    reinit_seed: int | None = None,
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
    tuning_seed : int, optional
        Runtime override for the recipe tuning seed. The effective seed drives
        warmup and defaults the sampler and reinitialization seeds to
        ``tuning_seed + 1`` and ``tuning_seed + 999`` respectively. The input
        recipe is never mutated.
    num_samples : int, optional
        Number of post-warmup samples to draw in the emitted inference loop.
        When ``None`` (default), reads ``recipe.calibration_budget["n_samples"]``
        (the validated config) and falls back to 1000 if not set.
        Pass an explicit integer to override (e.g. for a longer production run).
    sampler_seed : int, optional
        RNG seed for the post-warmup sampling. Defaults to
        ``effective_tuning_seed + 1`` so the emitted script is deterministic
        given the recipe and any runtime tuning-seed override.
    reinit_seed : int, optional
        RNG seed used when a sampler requires per-chain state reinitialization
        after warmup. Defaults to ``effective_tuning_seed + 999``.
    num_chains : int, optional
        Number of chains for the vmap-scan inference loop. When ``None``,
        derived from the recipe: ``recipe.warmup_params.get("num_chains",
        recipe.calibration_budget.get("num_chains", 4))``. Falls back to 1
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
        - An ``int`` passed for a multi-phase warmup is rejected because it
          cannot identify a count for every phase.

        Typical use: fast dry-run without editing the recipe.::

            emit_script(recipe, num_warmup=[100, 10], num_samples=100)
    progress_bar : bool or None, optional
        When ``True``, wraps the emitted warmup and sampling
        ``run``/``run_inference_algorithm`` calls in
        ``with blackjax.progress_bar():`` (blackjax #964 — vmap-safe, so this
        no longer affects TOPOLOGY at all: warmup/sampling stay multi-chain
        either way, see ``warmup_num_chains`` below for the one knob that
        does select single-chain).  Also sets the sampling
        ``_SAMPLING_PROGRESS_BAR`` constant in the emitted script to the same
        value (informational only — a plain local variable, not itself
        forwarded to blackjax).  When ``None`` (default) or ``False``, no
        progress bar is wrapped.
    warmup_num_chains : list[int] or None, optional
        Runtime override for ``recipe.warmup_num_chains``.  Affects which warmup
        template variant is selected:

        - ``None`` (default): falls back to ``recipe.warmup_num_chains``. If
          both are ``None``, every warmup phase uses the sampling chain count.
        - ``[1]`` with more than one sampling chain selects shared single-chain
          warmup. Recommended for expensive-logprob models to avoid
          vmap-of-while_loop; unrelated to ``progress_bar``.
        - ``[W]`` with ``W == num_chains``: same as ``None`` — uses the multichain
          template.
        Other single-phase window-adaptation topologies fail closed until code
        generation implements their reduce-and-broadcast choreography.
        Multi-phase generation currently supports only the two-phase Laplace
        diagonal-to-dense path with one warmup chain per phase.

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
    NotImplementedError
        If the requested warmup chain topology is not supported by code
        generation.
    """
    # Keep legacy baked recipes on the same canonical executable identity as
    # the execution-plan resolver (zero-step ``no_warmup`` stage).
    _budget = getattr(recipe, "calibration_budget", {}) or {}
    if (
        getattr(recipe, "warmup_name", None) == ""
        and isinstance(_budget, dict)
        and isinstance(_budget.get("baked_from"), dict)
    ):
        recipe = recipe.normalize_pinned_replay()
    plan = resolve_execution_plan(
        recipe,
        ExecutionOverrides(
            tuning_seed=tuning_seed,
            sampler_seed=sampler_seed,
            reinit_seed=reinit_seed,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            num_warmup=num_warmup,
            warmup_num_chains=warmup_num_chains,
        ),
    )
    manifest = ExecutionManifest.from_plan(plan, generator_version=__version__)
    config = plan.config
    config_values = config.as_dict()
    sampler_seed = config.sampler_seed
    reinit_seed = config.reinit_seed
    num_samples = config.num_samples
    num_chains = config.num_chains
    _warmup_pb = config.progress_bar
    _sampling_pb = config.progress_bar
    _warmup_counts = [stage.num_warmup for stage in config.warmup_stages]
    _warmup_chain_counts = [stage.num_chains for stage in config.warmup_stages]

    # x64 requirement is part of the resolved executable configuration.
    # The x64 line must appear BEFORE any JAX computation (build_logdensity_fn,
    # jax.random.key, etc.), so it is injected into the preamble immediately
    # after ``import jax``.
    _x64_config_line = (
        _X64_CONFIG_LINE if config.requires_x64 else _X64_CONFIG_LINE_EMPTY
    )

    # Normalise warmup_params key spelling: groundtruth recipes use
    # "target_acceptance" (legacy key from certify_reference.py);
    # newer recipe-generation code uses "target_acceptance_rate".
    target_acceptance_rate = recipe.warmup_params.get(
        "target_acceptance_rate",
        recipe.warmup_params.get(
            "target_acceptance",
            0.9 if recipe.warmup_name == "adjusted_mclmc_trajectory_tuning" else 0.8,
        ),
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
    _n_warmup_resolved = _warmup_counts[0]

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
        "plan_hash": manifest.plan_hash,
        "execution_manifest_json": manifest.to_json(),
        "verdict": recipe.gate_evidence.get("auto", {}).get("verdict", "NOT_RUN"),
        "tuning_seed": config.tuning_seed,
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
    _init_strategy = config_values["init_strategy"]
    _init_strategy_kind = (
        "prior_sample" if _init_strategy is None else _init_strategy.get("type")
    )
    ctx["init_position_is_prebatched"] = _init_strategy_kind in {
        "uniform_perchain",
        "zero_perchain",
        "reference_summary",
    }
    init_body = emit_init_strategy(_init_strategy, num_chains)
    step_policy_body = (
        emit_step_policy(config_values["step_policy"])
        if recipe.base_method_name in {"dynamic_hmc", "dmhmc"}
        and recipe.warmup_name != "chees"
        else None
    )
    # T1.5: resolve postamble info-diagnostics and draws-stats blocks at emit time.
    ctx["info_diagnostics_block"] = _build_info_diagnostics_block(
        recipe.base_method_name
    )
    ctx["draws_ss_block"] = _build_draws_ss_block(recipe.base_method_name)
    ctx["sample_stat_prefix"] = SAMPLE_STAT_PREFIX
    # Fixed trajectory length is recipe geometry (not adapted) for these samplers.
    if recipe.base_method_name in {
        "hmc",
        "mhmc",
        "rmhmc",
        "laplace_hmc",
        "laplace_mhmc",
    }:
        ctx["fixed_num_integration_steps"] = recipe.base_method_params.get(
            "num_integration_steps"
        )
    else:
        ctx["fixed_num_integration_steps"] = None
    ctx["telemetry_schema"] = TELEMETRY_SCHEMA
    ctx["telemetry_resolved_step_policy_expr"] = (
        "_resolved_step_policy" if step_policy_body is not None else "None"
    )
    _telemetry_geometry: dict[str, str] = {}
    _telemetry_geometry_reason: str | None = None
    _telemetry_geometry_source = "unavailable"
    if recipe.warmup_name not in {"", "no_warmup"}:
        _telemetry_geometry_source = "adapted"
        _telemetry_geometry = {
            # ``get`` is deliberate: old/upstream warmups may omit fields.  The
            # generated postamble turns any missing field into an explicitly
            # unavailable geometry rather than serialising JSON nulls.
            "step_size": "_adapted_params.get('step_size')",
            "inverse_mass_matrix": "_adapted_params.get('inverse_mass_matrix')",
        }
        if recipe.warmup_name in {
            "mclmc_tuning",
            "mclmc_lrd_tuning",
            "adjusted_mclmc_tuning",
            "adjusted_mclmc_trajectory_tuning",
        }:
            _telemetry_geometry["L"] = "_adapted_params.get('L')"
        if recipe.warmup_name == "meads":
            _telemetry_geometry.update(
                {
                    "alpha": "_adapted_params.get('alpha')",
                    "delta": "_adapted_params.get('delta')",
                }
            )
    elif recipe.base_method_name in _REPLAY_HMC_SAMPLERS:
        _telemetry_geometry_source = "pinned"
        _telemetry_geometry = {
            "step_size": "_default_step_size",
            "inverse_mass_matrix": "_default_imm",
        }
        if recipe.base_method_name == "mclmc":
            _telemetry_geometry["L"] = "_default_L"
    else:
        _telemetry_geometry_reason = (
            "sampler has no stable adapted geometry fields in generated protocol"
        )
    ctx["telemetry_geometry_expr"] = _telemetry_geometry
    ctx["telemetry_geometry_source"] = _telemetry_geometry_source
    ctx["telemetry_geometry_unavailable_reason"] = _telemetry_geometry_reason

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

    # A baked recipe has no adaptation phase: replay must use the exact pinned
    # sampler geometry, rather than silently falling back to library defaults.
    # Legacy ``no_warmup`` scaffolds remain allowed to use identity IMM.
    _is_no_warmup = recipe.warmup_name in {"", "no_warmup"}
    _is_baked_replay = recipe.warmup_name == "" or bool(
        isinstance(recipe.calibration_budget, dict)
        and recipe.calibration_budget.get("baked_from")
    )
    ctx["is_baked_replay"] = _is_baked_replay
    if (
        _is_baked_replay
        and recipe.base_method_name in _UNSUPPORTED_PINNED_REPLAY_SAMPLERS
    ):
        raise NotImplementedError(
            "Pinned Laplace replay requires a phi-space reference initializer "
            "and metric representation; code generation refuses to substitute "
            "an identity inverse mass matrix."
        )
    if _is_baked_replay and recipe.base_method_name in _REPLAY_HMC_SAMPLERS:
        _required_replay_params = ["step_size", "inverse_mass_matrix"]
        if recipe.base_method_name == "mclmc":
            _required_replay_params.append("L")
        if recipe.base_method_name in {"hmc", "mhmc", "rmhmc"}:
            _required_replay_params.append("num_integration_steps")
        _missing = [
            key
            for key in _required_replay_params
            if recipe.base_method_params.get(key) is None
        ]
        if _missing:
            raise ValueError(
                "No-warmup replay requires pinned "
                f"{', '.join(_missing)} for {recipe.base_method_name}; "
                "refusing to invent sampler tuning."
            )

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
    from tuningfork.base_method import BASE_METHODS, default_value_for_space

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
                _warmup_extra[_space.name] = recipe.base_method_params.get(
                    _space.name, default_value_for_space(_space)
                )

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
        from tuningfork.recipes._laplace_config import LAPLACE_PHI_THETA_SPLITS

        if recipe.model_name not in LAPLACE_PHI_THETA_SPLITS:
            raise ValueError(
                f"laplace_* recipe requested for model {recipe.model_name!r} but no "
                "phi/theta split is registered in _LAPLACE_PHI_THETA_SPLITS. "
                "Add an entry before calling emit_script for this model."
            )
        phi_sites, theta_sites = LAPLACE_PHI_THETA_SPLITS[recipe.model_name]
        ctx["phi_sites_repr"] = repr(phi_sites)
        ctx["theta_sites_repr"] = repr(theta_sites)

        # Build the LaplaceMarginal factory expression for each warmup phase.
        # All _LAPLACE_OPTIMIZER_KWARG_NAMES keys from phase/recipe params are included.
        from tuningfork.recipes._laplace_config import extract_laplace_optimizer_kwargs

        def _laplace_factory_expr(opt_kwargs: dict) -> str:
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in opt_kwargs.items())
            sep = ", " if kwargs_str else ""
            return f"_lmf(log_joint_fn, theta_init{sep}{kwargs_str})"

        if _is_multiphase_warmup:
            _factory_exprs = []
            for _phase in recipe.warmups:
                _phase_opt_kwargs = extract_laplace_optimizer_kwargs(
                    _phase["params"], recipe.base_method_params
                )
                _factory_exprs.append(_laplace_factory_expr(_phase_opt_kwargs))
            ctx["laplace_factories_expr"] = ", ".join(_factory_exprs)
            ctx["num_warmup_phases"] = len(recipe.warmups)
        else:
            # Single-phase: extract optimizer kwargs from warmup_params then bm fallback.
            _single_opt_kwargs = extract_laplace_optimizer_kwargs(
                recipe.warmup_params, recipe.base_method_params
            )
            ctx["laplace_factories_expr"] = _laplace_factory_expr(_single_opt_kwargs)
            ctx["num_warmup_phases"] = 1

        # Build a Python dict literal of optimizer kwargs for the sampler template.
        # Templates use $bm_optimizer_kwargs_expr to get all optimizer kwargs as a
        # dict they can spread: **_optimizer_kwargs.
        _bm_opt_kwargs = extract_laplace_optimizer_kwargs(recipe.base_method_params)
        ctx["bm_optimizer_kwargs_expr"] = repr(_bm_opt_kwargs)

    # ── Multi-phase warmup slot population ───────────────────────────────────
    # For multi-phase laplace warmup (len(recipe.warmups) > 1), populate per-phase
    # template slots: $wp0_*, $wp1_*, etc.  These are consumed by the
    # laplace_multiphase_warmup.py.tmpl template.
    if _is_laplace and _is_multiphase_warmup:
        # Build per-phase num_warmup overrides: None → use recipe value per phase.
        _per_phase_n_warmup: list[int | None]
        _per_phase_n_warmup = list(_warmup_counts)

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
            _ph_opt = extract_laplace_optimizer_kwargs(
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

    # Warmup variants that support a multi-chain (vmap) path (always used unless
    # warmup_num_chains=[1] forces single-chain — see _is_wa_multichain below).
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

    # The execution plan has already applied override precedence and canonicalized
    # omitted topology to the sampling chain count.
    _wnc_emit = _warmup_chain_counts
    # For single-phase warmups, the first (and only) entry drives template selection:
    # W == 1 < S selects shared single-chain warmup; W == S uses the per-chain
    # template, including the degenerate one-chain case.
    # Other values were rejected while resolving the executable plan.
    _warmup_W0 = _wnc_emit[0]
    _uses_shared_window_warmup = _warmup_W0 == 1 and num_chains > 1

    # Registry entry for the sampler — needed by both emit_warmup (A3) and
    # emit_sampler (A2). Resolved once here before warmup dispatch.
    _bm_entry = BASE_METHODS[recipe.base_method_name]

    # Build the warmup body with descriptor-selected Python emit functions.
    # The multichain flag for WA variants is resolved here and injected into ctx
    # so emit_warmup can dispatch on it without re-deriving the same logic.
    #
    # Stage 2 (blackjax #964): topology is now ALWAYS multichain except for the
    # independent warmup_num_chains=[1] knob (avoids vmap-of-while_loop for
    # expensive-logprob models — unrelated to progress bars). progress_bar no
    # longer forces single-chain: blackjax.progress_bar() is vmap-safe, so the
    # flag now only controls whether emitted calls are wrapped in
    # `with blackjax.progress_bar():` (see emit_warmup / _build_inference_loop).
    _is_wa_multichain = (
        not _is_multiphase_warmup
        and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS
        and not _uses_shared_window_warmup
    )
    ctx["_warmup_is_multichain"] = _is_wa_multichain

    # Laplace multi-phase recipes
    # dispatch to "laplace_multiphase_warmup" REGARDLESS of recipe.warmup_name.
    # The compatibility ``recipe.warmup_name`` may identify the first phase;
    # the ordered ``recipe.warmups`` list is authoritative whenever it contains
    # multiple phases, so dispatch to the explicit orchestration emitter.
    _effective_warmup_name = (
        "laplace_multiphase_warmup"
        if (_is_laplace and _is_multiphase_warmup)
        else ("no_warmup" if _is_no_warmup else recipe.warmup_name)
    )
    if _effective_warmup_name in EMITTABLE_WARMUP_NAMES:
        warmup_body = emit_warmup(_effective_warmup_name, _bm_entry, ctx)
    else:
        raise FileNotFoundError(
            f"No emit function for warmup {recipe.warmup_name!r}. "
            "The warmup registry and emitter support must be updated together."
        )

    # Pre-compute whether laplace_hmc/laplace_mhmc need warmup-state reinit.
    # Must be in ctx BEFORE emit_sampler is called (sampler reads it for
    # _state_reinit emission).  The same flag is also used by _resolved_needs_state_reinit
    # below (for the inference loop).
    # Condition: laplace recipe + NUTS warmup substitute → warmup produces HMCState,
    # which is incompatible with laplace_hmc.kernel (requires LaplaceHMCState.theta_star).
    ctx["_laplace_needs_warmup_state_reinit"] = (
        _is_laplace
        and recipe.base_method_name in {"laplace_hmc", "laplace_mhmc"}
        and _warmup_sampler == "nuts"
    )

    # A2: descriptor-driven emit for all 15 sampler template families.
    # For methods outside this set (mclmc, adjusted_mclmc, etc.), the
    # FileNotFoundError from _load_template propagates up, which causes
    # _try_emit_script to return None (skip) in the golden gate tests --
    # same behavior as before.
    _EMIT_SAMPLER_NAMES = frozenset(
        {
            "nuts",
            "hmc",
            "mhmc",
            "rmhmc",
            "dynamic_hmc",
            "dmhmc",
            "ghmc",
            "laplace_hmc",
            "laplace_mhmc",
            "laplace_dhmc",
            "laplace_dmhmc",
            "mala",
            "barker",
            "rwm",
            "meanfield_vi",
            "fullrank_vi",
            "mclmc",
            "adjusted_mclmc",
            "adjusted_mclmc_dynamic",
        }
    )
    if recipe.base_method_name in _EMIT_SAMPLER_NAMES:
        # A2: Python emit-function (descriptor-driven; no .tmpl file).
        ctx["chees_adapted"] = recipe.warmup_name == "chees"
        sampler_body = emit_sampler(_bm_entry, ctx)
    else:
        sampler_body = _load_template(
            f"samplers/{recipe.base_method_name}.py.tmpl"
        ).safe_substitute(ctx)
    # T1.3: strip the try/except NameError _state_post_warmup block from
    # sampler templates for non-no_warmup recipes (dead code for those paths).
    # For no_warmup, the block is the initialization path and must be kept.
    if not _is_no_warmup or ctx["init_position_is_prebatched"]:
        sampler_body = _strip_no_warmup_try_block(sampler_body)

    # T1.1: resolve the 3 inference-loop sentinels at emit time.
    # All three flags are statically determined from (warmup_name, warmup_template_path,
    # base_method_name) — no runtime try/except NameError probes needed.
    #
    # _warmup_init_is_single_chain: True iff warmup_name == "no_warmup"
    _resolved_warmup_init_is_single_chain = (
        _is_no_warmup and not ctx["init_position_is_prebatched"]
    )
    _resolved_warmup_init_is_prebatched = (
        _is_no_warmup and ctx["init_position_is_prebatched"]
    )

    # _warmup_is_perchain: True iff the selected warmup template sets it True.
    # Specifically: multichain window_adaptation templates + VI warmups +
    # mclmc_tuning / mclmc_lrd_tuning (always multi-chain per-chain warmup).
    _VI_WARMUP_NAMES = frozenset({"meanfield_vi", "fullrank_vi"})
    _MCLMC_WARMUP_NAMES = frozenset(
        {
            "mclmc_tuning",
            "mclmc_lrd_tuning",
            "adjusted_mclmc_tuning",
            "adjusted_mclmc_trajectory_tuning",
        }
    )
    # Stage 2 (blackjax #964): must stay in lockstep with _is_wa_multichain
    # above — this flag decides how the INFERENCE LOOP interprets the warmup
    # output's shape (per-chain vs shared/scalar), so it can never disagree
    # with what emit_warmup actually emitted. progress_bar no longer forces
    # single-chain, so it plays no role here either.
    _uses_multichain_warmup_tmpl = (
        not _is_multiphase_warmup
        and not _resolved_warmup_init_is_single_chain
        and not _uses_shared_window_warmup
        and recipe.warmup_name in _MULTICHAIN_WARMUP_VARIANTS
    )
    _resolved_warmup_is_perchain = (
        _uses_multichain_warmup_tmpl
        or recipe.warmup_name in _VI_WARMUP_NAMES
        or recipe.warmup_name in _MCLMC_WARMUP_NAMES
        or _resolved_warmup_init_is_prebatched
    )
    ctx["telemetry_geometry_scope"] = (
        None
        if not _telemetry_geometry
        else (
            "shared"
            if _is_no_warmup or not _resolved_warmup_is_perchain
            else "per_chain"
        )
    )
    # mclmc warmups also return per-chain L — the inference loop needs a
    # separate batched_L vmap axis alongside step_size and imm.
    _resolved_has_per_chain_L = recipe.warmup_name in _MCLMC_WARMUP_NAMES

    # _needs_state_reinit: True iff sampler template defines _state_reinit.
    # ctx["_laplace_needs_warmup_state_reinit"] was already set before emit_sampler.
    _resolved_needs_state_reinit = (
        recipe.base_method_name in _STATE_REINIT_SAMPLERS
        or ctx["_laplace_needs_warmup_state_reinit"]
    )
    if (recipe.base_method_name, recipe.warmup_name) in {
        ("dynamic_hmc", "chees"),
        ("ghmc", "meads"),
    }:
        _resolved_needs_state_reinit = False

    _no_warmup_step_expr = 'float(_adapted_params.get("step_size", 1.0))'
    _no_warmup_imm_expr = "jnp.ones(_n_dims)"
    _no_warmup_L_expr = "1.0"
    if recipe.base_method_name in _REPLAY_HMC_SAMPLERS:
        _no_warmup_step_expr = "_default_step_size"
        _no_warmup_imm_expr = "_default_imm"
    if recipe.base_method_name == "mclmc":
        _no_warmup_L_expr = "_default_L"

    # T1.4: emit timing block without try/except — no_warmup path omits
    # block_until_ready (state not yet set at this point in assembly).
    _timing_block = (
        _WARMUP_TIMING_BLOCK_NO_WARMUP if _is_no_warmup else _WARMUP_TIMING_BLOCK_WARMUP
    )

    # T1.1: build straight-line inference loop (no try/except NameError probes).
    inference_loop = _build_inference_loop(
        num_samples=num_samples,
        sampler_seed=sampler_seed,
        reinit_seed=reinit_seed,
        num_chains=num_chains,
        use_progress_bar=_sampling_pb,
        warmup_is_perchain=_resolved_warmup_is_perchain,
        warmup_init_is_single_chain=_resolved_warmup_init_is_single_chain,
        warmup_init_is_prebatched=_resolved_warmup_init_is_prebatched,
        needs_state_reinit=_resolved_needs_state_reinit,
        has_per_chain_L=_resolved_has_per_chain_L,
        no_warmup_step_size_expr=_no_warmup_step_expr,
        no_warmup_imm_expr=_no_warmup_imm_expr,
        no_warmup_L_expr=_no_warmup_L_expr,
    )

    diagnostics_body = emit_diagnostics(ctx)
    ctx["diagnostics_close_body"] = emit_diagnostics_close()
    postamble = emit_postamble(ctx)

    # Laplace setup first narrows the full model position to phi-space. The
    # configured initialization strategy must run after that transformation and
    # before warmup, matching the execution plan's position-space semantics.
    # Dynamic-HMC step policies are defined after warmup and before every
    # constructor that consumes them.
    sections = [preamble]
    if _is_laplace:
        sections.append(emit_laplace_preamble(ctx))
    sections.extend([init_body, diagnostics_body, warmup_body, _timing_block])
    if step_policy_body is not None:
        sections.append(step_policy_body)
    sections.extend([sampler_body, inference_loop, postamble])
    return "\n\n".join(sections)
