#!/usr/bin/env python3
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
"""Re-emit committed recipes under their own recorded configuration.

A re-emit is only interpretable if it reruns the SAME cell.  A committed recipe
does not store the emit call that produced it, so this module reconstructs that
call from the artifact and refuses to emit any cell it cannot reconstruct
faithfully.

That refusal is the point.  A skipped cell is a known gap.  A cell emitted under
a wrongly-reconstructed configuration is a silently confounded number that will
later be read as a scientific finding — the committed radon recipe is the worked
example: its headline was stamped from a cached run recording 600000 gradient
evaluations, so re-emitting it under the standard protocol moves the headline
9.4x for reasons that have nothing to do with the metric under study.

Run the verification pass over the whole corpus BEFORE emitting anything::

    uv run python tools/reemit_sweep.py --verify-only

Then emit, one cell per process, each under its own memory cap::

    uv run python tools/reemit_sweep.py --plan > cells.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "tuningfork" / "catalog"

# Base methods whose headline is not an autocorrelation ESS, or whose recipes
# carry no headline at all.  Out of scope for an estimator migration.
_VI_METHODS = frozenset({"meanfield_vi", "fullrank_vi"})

# Effort tiers this driver will emit.  emit_low_recipe_for_cell documents
# behaviour for other tiers as untested, and a HIGH recipe's configuration
# includes a hyperparameter search that the artifact does not record.
_EMITTABLE_EFFORTS = frozenset({"low", "medium"})

# The documented standard protocol, applied ONLY to the cells below.
_STANDARD_PROTOCOL = {"n_warmup": 1000, "n_samples": 1000, "num_chains": 4}

#: Cells emitted under the standard protocol rather than their own recorded one,
#: because what they recorded is itself defective.  Their headline was stamped
#: from a cached run instead of measured, so the artifact carries a gradient
#: budget that no stated protocol reproduces — re-measuring is a correction, not
#: a like-for-like replay.
#:
#: These are NOT reconstruction successes and must not be pooled with the rest:
#: their movement is dominated by the budget being fixed, so the delta report
#: reports them as a config correction rather than as an estimator effect.
#: Every entry is a deliberate, recorded decision; do not add one to make a cell
#: emit.
CONFIG_CORRECTION_CELLS: dict[str, str] = {
    "radon/low__nuts__window_adaptation_diag_imm.json": (
        "headline was stamped from cached chain statistics recording 600000 "
        "gradient evaluations, which the standard protocol does not reproduce "
        "(it yields 60704); the artifact records no sample budget"
    ),
    "lotka_volterra/low__nuts__window_adaptation_low_rank_imm.json": (
        "headline was stamped from cached chain statistics, so the artifact "
        "records no sample budget at all; emitted under the standard protocol"
    ),
}

#: Historical MCLMC-LRD artifacts whose bake record does not contain a
#: reconstructable sampling budget.  Their seed evidence records the measured
#: gradient budget, but not the n_samples intent needed to replay the generated
#: recipe call.  They remain committed evidence and are intentionally excluded
#: from verify-only rather than assigned an invented default.
HISTORICAL_NON_REEMITTABLE_CELLS: dict[str, str] = {
    "german_credit/low__mclmc_lrd__mclmc_lrd_tuning.json": (
        "historical MCLMC-LRD artifact has no recorded n_samples intent; "
        "generated call cannot be reconstructed without inventing a sample budget"
    ),
    "ill_cond_50/low__mclmc_lrd__mclmc_lrd_tuning.json": (
        "historical MCLMC-LRD artifact has no recorded n_samples intent; "
        "generated call cannot be reconstructed without inventing a sample budget"
    ),
}

#: Cells whose re-emit ran at a different float precision than the run it
#: reproduces, keyed by cell with the recorded ``jax_x64_enabled`` transition.
#:
#: x64 is a per-model attribute (``Posterior.requires_x64``, honoured at
#: generated certification lifecycle rather than a recorded parameter, so it is set by
#: the ambient environment for every model that does not demand it.  These cells
#: were committed with ``JAX_ENABLE_X64=1`` ambient — off-protocol, since none of
#: their models sets ``requires_x64`` — and the re-emit ran under the documented
#: float32 default.  Three artifact signals agree on the same set: the recorded
#: flag, the decimal precision of the adapted ``step_size`` (float64-exact ->
#: float32-exact), and the dtype of the ``.imm.npz`` sidecar (9 of them,
#: float64 -> float32).
#:
#: The re-emit is arguably the MORE correct run, so these are not re-emitted
#: again.  They are pinned because their movement has an unmodelled cause: they
#: are enriched in the residual tails (extreme movers 5/16 here vs 7/122
#: elsewhere, one-sided Fisher p = 0.005), so pooling them into the version-drift
#: aggregate attributes precision to dependencies.  ``estimator_delta_report``
#: reports them as their own category for that reason.
#:
#: Every entry is a deliberate, recorded acceptance.  Do NOT add one to silence a
#: fresh flip: an unpinned flip is a real finding — a run executed at a precision
#: its own artifact does not claim.
PRECISION_FLIP_CELLS: dict[str, str] = {
    "banana/medium__adjusted_mclmc_dynamic__adjusted_mclmc_tuning.json": (
        "jax_x64_enabled True -> False"
    ),
    "eight_schools_ncp/low__dmhmc__window_adaptation_dense_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "eight_schools_ncp/low__hmc__window_adaptation_low_rank_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "german_credit/low__dynamic_hmc__chees.json": "jax_x64_enabled True -> False",
    "german_credit/medium__hmc__window_adaptation_diag_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "irt_2pl/low__hmc__window_adaptation_diag_imm__inner_nuts.json": (
        "jax_x64_enabled True -> False"
    ),
    "irt_2pl/low__hmc__window_adaptation_low_rank_imm__inner_nuts.json": (
        "jax_x64_enabled True -> False"
    ),
    "irt_2pl/low__mhmc__window_adaptation_diag_imm__inner_nuts.json": (
        "jax_x64_enabled True -> False"
    ),
    "irt_2pl/low__nuts__window_adaptation_dense_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "mvn_10/low__mclmc__mclmc_tuning.json": "jax_x64_enabled True -> False",
    "mvn_10/medium__hmc__window_adaptation_dense_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "mvn_10/medium__hmc__window_adaptation_diag_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "radon/low__mhmc__window_adaptation_diag_imm__inner_nuts.json": (
        "jax_x64_enabled True -> False"
    ),
    # 2026-07-30 recert (tuningfork PR #257): baseline was captured on a CUDA
    # GPU host with x64 ambient-on; the re-emit ran on this project's aarch64
    # CPU host under the documented float32 default. radon does not set
    # requires_x64, so both runs are within protocol -- same shape as the 16
    # entries above, discovered later because #257 never saw the
    # verify_emitted_configs.py gate (merged 4 seconds after the PR that wired
    # it into CI, #256).
    "radon/medium__dynamic_hmc__chees.json": "jax_x64_enabled True -> False",
    "stoch_vol/low__mhmc__window_adaptation_diag_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    "stoch_vol/low__nuts__window_adaptation_diag_imm.json": (
        "jax_x64_enabled True -> False"
    ),
    # Also a config correction: the baseline recorded no machine_info at all, so
    # the precision it ran at is not merely different, it is unknown.
    "radon/low__nuts__window_adaptation_diag_imm.json": (
        "jax_x64_enabled absent -> False"
    ),
}

_NON_REPLAYABLE_RECIPE_PARAMETER_KEYS = frozenset({"k_rank", "ncp_variant"})


def recorded_x64(recipe: dict) -> bool | None:
    """The float precision an artifact records having run at, or ``None``."""
    machine_info = (recipe.get("calibration_budget") or {}).get("machine_info") or {}
    return machine_info.get("jax_x64_enabled")


@dataclass
class CellConfig:
    """A reconstructed emit call for one committed recipe."""

    recipe_path: Path
    model_name: str
    warmup_name: str
    sampler_name: str
    effort: str
    n_warmup: int
    n_samples: int
    num_chains: int
    seed: int
    target_acceptance: float | None = None
    sampler_kwargs_override: dict[str, Any] | None = None
    warmup_kwargs_override: dict[str, Any] | None = None
    step_policy: dict[str, Any] | None = None
    policy_tag: str | None = None
    warmup_inner_kernel: str | None = None
    init_strategy: dict[str, Any] | None = None
    variant_label: str | None = None
    config_correction: bool = False
    #: ``jax_x64_enabled`` as recorded by the artifact this config was read FROM.
    #: Not a settable input — x64 follows the model's ``requires_x64`` and the
    #: ambient environment — but recording it is what lets the fidelity check see
    #: a replay that ran at a different precision than the run it reproduces.
    recorded_x64: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.model_name}/{self.recipe_path.name}"


@dataclass
class Skip:
    """A cell this driver declines to emit, and why."""

    recipe_path: Path
    reason: str

    @property
    def key(self) -> str:
        return f"{self.recipe_path.parent.parent.name}/{self.recipe_path.name}"


def _policy_tag(name: str) -> str | None:
    """Recover the ``__policy_<slug>`` filename modifier, if present."""
    stem = name.removesuffix(".json")
    for part in stem.split("__")[3:]:
        if part.startswith("policy_"):
            return part
    return None


def _combined_filename_tag(
    sampler_name: str, warmup_inner_kernel: str | None, policy_tag: str | None
) -> str | None:
    """The filename modifier the runner would compose, in its own order.

    Mirrors the runner: an ``inner_<kernel>`` segment appears only when the
    explicit inner kernel differs from the implicit default for that sampler,
    then any policy segment.  Composed here rather than parsed off the filename
    so the round-trip check compares against what an emit would actually write.
    """
    from tuningfork.warmup._laplace_adapter import WARMUP_SUBSTITUTE_METHOD_NAMES

    implicit = (
        "nuts" if sampler_name in WARMUP_SUBSTITUTE_METHOD_NAMES else sampler_name
    )
    inner_tag = (
        f"inner_{warmup_inner_kernel}"
        if warmup_inner_kernel is not None and warmup_inner_kernel != implicit
        else None
    )
    parts = [t for t in (inner_tag, policy_tag) if t]
    return "__".join(parts) if parts else None


def _predicted_tuning_seed(seed: int) -> int:
    """What ``tuning_seed`` the runner would stamp for this master seed.

    The runner derives it as ``bits(split(key(seed), 3)[1])``.  The master seed
    itself never reaches the artifact, so matching this derived value is the only
    available check that the seed we are about to pass is the seed that produced
    the committed recipe.

    Measured to be invariant to ``jax_enable_x64`` and identical under jax 0.10.0,
    0.10.1 and 0.11.0, so a mismatch means a different seed rather than a
    library-version artefact.
    """
    import jax

    _, warmup_key, _ = jax.random.split(jax.random.key(seed), 3)
    return int(jax.random.bits(warmup_key, dtype="uint32"))


def _seed_candidates() -> list[int]:
    """Master seeds that could have produced a committed recipe, in priority order.

    Not guesswork — each entry corresponds to a real code path:

    - the documented default;
    - the value that default DERIVES, because the rerun path feeds a recipe's own
      ``tuning_seed`` back in as the master seed, so a re-emitted cell sits one
      generation down the chain and the artifact records only the terminal value;
    - historical recorded certification seeds present in committed evidence.

    A brute-force search over master seeds up to 4e7 confirmed these account for
    every ``tuning_seed`` in the committed corpus; anything else is skipped rather
    than guessed at.
    """
    from tuningfork.recipes._certification_runner import RECIPE_SEED

    return [RECIPE_SEED, _predicted_tuning_seed(RECIPE_SEED), 11111, 22222, 33333]


def _roundtrip_filename(recipe_path: Path, filename_tag: str | None) -> str:
    """The filename ``Recipe.save`` would choose for this recipe, given a tag.

    Uses the real save path against a throwaway directory instead of
    reimplementing its stem composition, so this check cannot drift away from the
    behaviour it is checking.
    """
    import tempfile

    from tuningfork.recipes import Recipe

    loaded = Recipe.load(recipe_path)
    with tempfile.TemporaryDirectory() as tmp:
        return loaded.save(Path(tmp), filename_tag=filename_tag).name


def _reconstruct_sampler_kwargs(
    recipe: dict[str, Any],
    base_method: Any,
    warmup_inner_kernel: str | None,
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """Separate pinned kernel kwargs into "was an override" and "was derived".

    ``base_method_params`` is the kernel kwarg dict the run actually used, which
    mixes three provenances: registry defaults, values the warmup derived at run
    time, and values the caller overrode.  Only the third may be replayed — a
    derived value re-derives itself, and passing it as an override would freeze a
    run-time quantity into an input.

    Returns ``(override_or_None, notes, blocking_reason_or_None)``.
    """
    from tuningfork.base_method import default_params_for

    pinned = dict(recipe.get("base_method_params") or {})
    defaults = default_params_for(base_method)
    notes: list[str] = []

    # Never replayable: adapted by warmup, non-serialisable, or consumer-only.
    for k in (
        "step_size",
        "inverse_mass_matrix",
        "L",
        *_NON_REPLAYABLE_RECIPE_PARAMETER_KEYS,
    ):
        pinned.pop(k, None)

    sampler_name = recipe.get("base_method_name")
    if sampler_name in ("dynamic_hmc", "dmhmc"):
        # The runner strips the integer trajectory length for these and injects a
        # callable built from step_policy, so a pinned value cannot be an override.
        pinned.pop("num_integration_steps", None)

    if warmup_inner_kernel is not None and "num_integration_steps" in pinned:
        # transform_warmup_state derives this from the warmup's own trajectory
        # lengths.  Replaying it would pin a measured quantity as a setting.
        notes.append("num_integration_steps is warmup-derived; re-derived on emit")
        pinned.pop("num_integration_steps")

    override = {k: v for k, v in pinned.items() if defaults.get(k) != v}
    # Everything the runner injects itself has already been removed above, so a
    # surviving key outside the registry default space can only have arrived as a
    # caller override — replayable, but worth naming since it is unusual.
    unknown = sorted(k for k in override if k not in defaults)
    if unknown:
        notes.append(f"replaying non-registry kernel kwargs {unknown}")
    return (override or None, notes, None)


#: Warmup arguments the emit call takes explicitly; everything else in a
#: recipe's warmup_params came from the warmup's declared hyperparameter space.
_EXPLICIT_WARMUP_ARGS = frozenset({"n_warmup", "num_chains", "target_acceptance"})
_DECLARED_REPLAY_WARMUP_ARGS = {
    "mclmc_lrd_tuning": frozenset(
        {"k_rank", "pilot_n_warmup", "pilot_n_samples", "inner_kernel"}
    )
}


def _reconstruct_warmup_kwargs(warmup: Any, warmup_params: dict) -> dict | None:
    """Recover the warmup hyperparameters a committed recipe recorded.

    Returns only the keys the warmup actually declares, so a stray annotation in
    an old artifact cannot be passed through to the runner as a kwarg.
    """
    declared = {s.name for s in getattr(warmup, "default_hp_space", ())}
    declared.update(_DECLARED_REPLAY_WARMUP_ARGS.get(warmup.name, ()))
    override = {
        k: v
        for k, v in warmup_params.items()
        if k not in _EXPLICIT_WARMUP_ARGS and k in declared
    }
    return override or None


def config_fidelity_violations(cfg: CellConfig, committed: dict) -> list[str]:
    """Every committed parameter this reconstruction would fail to reproduce.

    The filename and seed guards proved a cell is the SAME cell; this proves it
    would be run the SAME WAY.  Without it a reconstruction can drop a
    load-bearing parameter and still pass every other check — which is how a
    low-rank cell was re-run with a tenth of its recorded pilot budget, collapsing
    its preconditioner, and the verification pass still reported it green.

    Compares the FULL committed parameter dictionaries, not a chosen subset, and
    reports any key present in the artifact whose replayed value would differ.
    Values the warmup adapts at run time are excluded by name, with a comment
    each, rather than by being quietly absent.
    """
    from tuningfork.base_method import (
        BASE_METHODS,
        default_params_for,
        default_value_for_space,
    )
    from tuningfork.warmup import WARMUPS

    violations: list[str] = []
    warmup = WARMUPS[cfg.warmup_name]
    base_method = BASE_METHODS[cfg.sampler_name]

    # --- warmup_params the emit would record ---
    replayed_warmup = {
        space.name: default_value_for_space(space)
        for space in getattr(warmup, "default_hp_space", ())
    }
    replayed_warmup.update(cfg.warmup_kwargs_override or {})
    replayed_warmup["n_warmup"] = cfg.n_warmup
    replayed_warmup["num_chains"] = cfg.num_chains
    if cfg.target_acceptance is not None:
        replayed_warmup["target_acceptance"] = cfg.target_acceptance

    # These values are supplied to the typed LRD warmup rather than the MCLMC
    # kernel. Historical artifacts duplicate ``k_rank`` in both dictionaries.
    lrd_keys = {"k_rank", "pilot_n_warmup", "pilot_n_samples"}

    committed_warmup = dict(
        (committed.get("warmups") or [{}])[0].get("params")
        or committed.get("warmup_params")
        or {}
    )
    for key, want in committed_warmup.items():
        if key not in replayed_warmup:
            violations.append(f"warmup_params[{key!r}]={want!r} would not be replayed")
        elif replayed_warmup[key] != want:
            violations.append(
                f"warmup_params[{key!r}]: committed {want!r}, replay "
                f"{replayed_warmup[key]!r}"
            )

    # --- base_method_params the emit would record ---
    # step_size / inverse_mass_matrix / L are adapted by the warmup, so the
    # committed values are outputs and cannot be predicted before running.
    adapted = {"step_size", "inverse_mass_matrix", "L"}
    replayed_kernel = {
        k: v for k, v in default_params_for(base_method).items() if k not in adapted
    }
    if cfg.sampler_name in ("dynamic_hmc", "dmhmc"):
        replayed_kernel.pop("num_integration_steps", None)
    replayed_kernel.update(cfg.sampler_kwargs_override or {})

    committed_kernel = dict(committed.get("base_method_params") or {})
    for key in adapted | lrd_keys:
        committed_kernel.pop(key, None)
    if cfg.warmup_inner_kernel is not None:
        # transform_warmup_state derives this from the warmup's own trajectory
        # lengths; it re-derives on replay rather than being passed in.
        committed_kernel.pop("num_integration_steps", None)

    for key, want in committed_kernel.items():
        if key not in replayed_kernel:
            violations.append(
                f"base_method_params[{key!r}]={want!r} would not be replayed"
            )
        elif replayed_kernel[key] != want:
            violations.append(
                f"base_method_params[{key!r}]: committed {want!r}, replay "
                f"{replayed_kernel[key]!r}"
            )

    # --- structural fields outside the two parameter dicts ---
    # These change what the run DOES, so dropping one is the same class of defect
    # as dropping a parameter, and the dictionary comparison above cannot see it.
    # inverse_mass_matrix_path is deliberately absent: it is an output.
    structural = {
        "step_policy": cfg.step_policy,
        "warmup_inner_kernel": cfg.warmup_inner_kernel,
        "init_strategy": cfg.init_strategy,
    }
    for key, replayed in structural.items():
        want = committed.get(key)
        if want != replayed:
            violations.append(f"{key}: committed {want!r}, replay {replayed!r}")

    # warmup_num_chains is a per-phase chain count that the emit entry point does
    # not accept at all, so a recipe carrying one cannot be reproduced by it.
    if committed.get("warmup_num_chains"):
        violations.append(
            f"warmup_num_chains={committed['warmup_num_chains']!r} is not an "
            f"argument of the emit path, so it cannot be replayed"
        )

    # --- the float precision the run executed at ---
    # Every comparison above is of a recorded PARAMETER.  x64 is not one: it
    # follows the model's requires_x64 and otherwise the ambient environment, so a
    # replay can execute the same cell in float32 that was committed in float64
    # and every parameter check stays green — which is what happened to 15 cells,
    # while both gates reported 0 mismatches.  The artifact already records the
    # flag, so comparing it costs nothing and closes that axis.
    #
    # Vacuous where the committed side IS the artifact being reconstructed (the
    # plan-side, self-reproducibility use); live where committed comes from
    # another revision (the artifact-side use).
    committed_x64 = recorded_x64(committed)
    if committed_x64 != cfg.recorded_x64 and cfg.key not in PRECISION_FLIP_CELLS:
        violations.append(
            f"machine_info.jax_x64_enabled: committed {committed_x64!r}, replay "
            f"{cfg.recorded_x64!r} — the run executed at a different float "
            f"precision than the run it reproduces"
        )
    return violations


def reconstruct(
    recipe_path: Path, source_path: Path | None = None
) -> CellConfig | Skip:
    """Rebuild the emit call for one committed recipe, or decline to.

    ``source_path`` supplies the artifact to read the configuration FROM, when
    that differs from the artifact the emit will write TO.  Required once any
    cell has been re-emitted on the branch: reading the working tree would
    then reconstruct from a file this process itself produced, so a
    mis-reconstruction becomes self-confirming and is never noticed.
    """
    source_path = source_path or recipe_path
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.catalog._estimator_provenance import (
        HEADLINE_ESTIMATOR_EXCLUDED_MODELS,
    )
    from tuningfork.model import MODELS
    from tuningfork.recipes import Recipe
    from tuningfork.recipes._certification_runner import RECIPE_SEED
    from tuningfork.warmup import WARMUPS

    def skip(reason: str) -> Skip:
        return Skip(recipe_path, reason)

    recipe = json.loads(source_path.read_text())
    model_name = recipe_path.parent.parent.name

    if model_name in HEADLINE_ESTIMATOR_EXCLUDED_MODELS:
        return skip("model is excluded from the estimator migration")
    if recipe_path.name.startswith("smc__"):
        return skip(
            "SMC headline is an importance-weight ESS, not an autocorrelation one"
        )

    effort = recipe.get("effort")
    if effort == "failed":
        return skip("failed recipe carries no headline")
    if recipe.get("base_method_name") in _VI_METHODS:
        return skip("VI base method carries a null headline")
    if effort not in _EMITTABLE_EFFORTS:
        return skip(
            f"effort {effort!r} is not emittable by this driver "
            f"(a HIGH recipe's configuration includes a hyperparameter search "
            f"that the artifact does not record)"
        )

    # Read warmup identity and parameters from the LOADED recipe, never from raw
    # JSON.  Two on-disk schemas are in use — a flat warmup_params dict and a
    # warmups list — and 148 of the committed recipes use the list form with no
    # flat dict at all.  Reading the raw key silently yields {} for those, which
    # drops target_acceptance and reruns a curvature-sensitive model at the
    # default 0.8.  Recipe.load normalises both forms; ask it rather than
    # reimplementing the fallback.
    try:
        loaded = Recipe.load(source_path)
    except Exception as exc:  # noqa: BLE001
        return skip(f"recipe does not load: {type(exc).__name__}: {exc}")
    warmup_name = loaded.warmup_name or ""
    sampler_name = recipe.get("base_method_name") or ""

    for name, registry, label in (
        (model_name, MODELS, "MODELS"),
        (warmup_name, WARMUPS, "WARMUPS"),
        (sampler_name, BASE_METHODS, "BASE_METHODS"),
    ):
        if not name or name not in registry:
            return skip(f"{name!r} is not registered in {label}")

    warmup, base_method = WARMUPS[warmup_name], BASE_METHODS[sampler_name]
    if not warmup.is_compatible(sampler_name):
        return skip(f"{warmup_name} is not compatible with {sampler_name}")

    budget = recipe.get("calibration_budget") or {}
    warmup_params = loaded.warmup_params or {}
    n_warmup = warmup_params.get("n_warmup", budget.get("n_warmup"))
    n_samples = budget.get("n_samples", warmup_params.get("n_samples"))
    num_chains = warmup_params.get("num_chains", budget.get("num_chains"))

    # Warmup hyperparameters beyond the three explicit emit arguments live in the
    # warmup's own declared space and reach the runner through
    # warmup_kwargs_override.  Not replaying them silently substitutes the
    # registry default: window_adaptation_low_rank_imm.max_rank and the VI
    # warmups' num_optimization_steps both change what the warmup actually does.
    warmup_kwargs_override = _reconstruct_warmup_kwargs(warmup, warmup_params)

    warmup_inner_kernel = recipe.get("warmup_inner_kernel")
    override, notes, blocker = _reconstruct_sampler_kwargs(
        recipe, base_method, warmup_inner_kernel
    )
    if blocker is not None:
        return skip(blocker)

    cell_key = f"{model_name}/{recipe_path.name}"
    if cell_key in HISTORICAL_NON_REEMITTABLE_CELLS:
        return skip(HISTORICAL_NON_REEMITTABLE_CELLS[cell_key])
    config_correction = cell_key in CONFIG_CORRECTION_CELLS
    missing = [
        label
        for label, value in (
            ("n_warmup", n_warmup),
            ("n_samples", n_samples),
            ("num_chains", num_chains),
        )
        if value is None
    ]
    if missing:
        if not config_correction:
            return skip(
                f"sample budget {missing} absent from the artifact — the committed "
                f"headline is not reproducible under any stated protocol"
            )
        n_warmup = n_warmup if n_warmup is not None else _STANDARD_PROTOCOL["n_warmup"]
        n_samples = (
            n_samples if n_samples is not None else _STANDARD_PROTOCOL["n_samples"]
        )
        num_chains = (
            num_chains if num_chains is not None else _STANDARD_PROTOCOL["num_chains"]
        )

    # A reconstructed emit must write back to the file it came from.  If it would
    # land on a different name it is a different cell: the emit would clobber a
    # neighbour and orphan the source, which no downstream check would notice
    # because both files would still parse.  Asked of the real save() path rather
    # than reimplemented, so it cannot drift from it.
    policy_tag = _policy_tag(recipe_path.name)
    written_name = _roundtrip_filename(
        source_path,
        _combined_filename_tag(sampler_name, warmup_inner_kernel, policy_tag),
    )
    if written_name != recipe_path.name:
        return skip(
            f"a re-emit would write {written_name!r}, not {recipe_path.name!r} — "
            f"the filename carries a modifier this driver cannot reproduce, so "
            f"emitting would clobber a different recipe and orphan this one"
        )

    seed = RECIPE_SEED
    committed_seed = recipe.get("tuning_seed")
    if config_correction:
        notes.append(CONFIG_CORRECTION_CELLS[cell_key])
        committed_seed = None
    if committed_seed:
        match = next(
            (
                s
                for s in _seed_candidates()
                if _predicted_tuning_seed(s) == committed_seed
            ),
            None,
        )
        if match is None:
            return skip(
                f"no known master seed derives the recorded tuning_seed "
                f"{committed_seed}, so the run that produced this recipe cannot "
                f"be reproduced from the artifact"
            )
        seed = match
        if seed != RECIPE_SEED:
            notes.append(f"emitted from master seed {seed}, not the default")

    return CellConfig(
        recipe_path=recipe_path,
        model_name=model_name,
        warmup_name=warmup_name,
        sampler_name=sampler_name,
        effort=effort,
        n_warmup=int(n_warmup),
        n_samples=int(n_samples),
        num_chains=int(num_chains),
        seed=seed,
        target_acceptance=warmup_params.get("target_acceptance"),
        sampler_kwargs_override=override,
        warmup_kwargs_override=warmup_kwargs_override,
        step_policy=recipe.get("step_policy"),
        policy_tag=policy_tag,
        warmup_inner_kernel=warmup_inner_kernel,
        init_strategy=recipe.get("init_strategy"),
        variant_label=recipe.get("variant_label"),
        config_correction=config_correction,
        recorded_x64=recorded_x64(recipe),
        notes=notes,
    )


def survey(from_rev: str | None = None) -> tuple[list[CellConfig], list[Skip]]:
    """Reconstruct every catalog recipe; partition into emittable and skipped.

    With ``from_rev``, configurations are read from that revision rather than the
    working tree, so partially-swept state cannot feed back into the plan.
    """
    import subprocess
    import tempfile

    ok: list[CellConfig] = []
    skipped: list[Skip] = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(CATALOG.glob("*/recipes/*.json")):
            source = p
            if from_rev:
                rel = p.relative_to(REPO_ROOT)
                proc = subprocess.run(
                    ["git", "show", f"{from_rev}:{rel}"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    skipped.append(Skip(p, f"absent at {from_rev}"))
                    continue
                source = Path(tmp) / rel
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(proc.stdout)
            result = reconstruct(p, source_path=source)
            if isinstance(result, CellConfig):
                # A reconstruction that drops a load-bearing parameter passes
                # every other guard, so the fidelity check is what makes the
                # gate's green light mean anything.
                bad = config_fidelity_violations(result, json.loads(source.read_text()))
                if bad:
                    skipped.append(Skip(p, "config fidelity: " + "; ".join(bad)))
                    continue
                ok.append(result)
            else:
                skipped.append(result)
    return ok, skipped


def recertify(
    recipe_path: Path,
    *,
    seed: int | None = None,
    source_revision: str | None = None,
    catalog_root: Path = CATALOG,
    verbose: bool = True,
) -> Any:
    """Re-run one committed cell under ITS OWN recorded configuration, in place.

    This is the sanctioned re-emission path for a recert: every field
    ``emit_low_recipe_for_cell`` accepts is read off the source artifact via
    ``reconstruct()`` and forwarded explicitly, rather than hand-assembled by a
    caller who has to remember every override the cell records.

    That hand-assembly is exactly the defect class this closes.
    ``horseshoe/medium__nuts__window_adaptation_diag_imm.json`` lost its pinned
    ``base_method_params["max_num_doublings"] = 15`` during the 2026-07-30
    recert sweep: the re-emit call that produced it forwarded ``seed`` and
    ``target_acceptance`` but never built a ``sampler_kwargs_override`` dict,
    so the run silently executed at the registry default (10) instead of the
    artifact's own recorded value, and the recipe it wrote down described a
    different, easier configuration than the one that had actually needed 15.
    Going through ``reconstruct()`` instead means the override dict is READ
    from the artifact's ``base_method_params``, not retyped from memory, so
    there is no step at which a key can be silently left out.

    Parameters
    ----------
    recipe_path
        The recipe to re-run and overwrite in place.  Its filename fixes the
        cell identity; a mismatched reconstruction (see ``reconstruct``'s
        filename round-trip guard) is refused rather than clobbering a
        neighbour.
    seed
        Master seed for this run.  ``None`` (default) reproduces the seed
        ``reconstruct()`` recovers from the source artifact's own
        ``tuning_seed`` -- an exact same-seed replay.  Pass an explicit seed
        for a disclosed reseed (Belief#1176): every OTHER field still comes
        from the source artifact, so choosing a new seed cannot also silently
        drop a kernel kwarg the way a fully hand-assembled call can.
    source_revision
        Read the configuration to replay from this git revision instead of
        the working tree.  Required whenever ``recipe_path`` on disk is
        itself the defective artifact being repaired: reconstructing from a
        file this call is about to overwrite makes a dropped parameter
        self-confirming, because the source no longer claims it either, so
        there is nothing left to notice is missing.  ``None`` (default)
        reconstructs from ``recipe_path`` as it currently sits on disk, which
        is correct for a cell being recertified for the first time (nothing
        has dropped anything from it yet).
    catalog_root
        Root the re-emitted recipe is written under.  Defaults to the real
        catalog so an in-place recert lands where the cell already lives;
        pass a scratch directory for a dry run.
    verbose
        Forwarded to ``emit_low_recipe_for_cell``.

    Returns
    -------
    CellResult
        The outcome of the generated run. Every valid PASS, REVIEW, FAIL, or
        ERROR attempt is appended to the recipe with its receipt and evidence;
        the materialized current view is updated without erasing prior
        attempts. Invalid intent or a corrupt existing target fails closed.
    """
    source_path = recipe_path
    _scratch: Path | None = None
    if source_revision is not None:
        import subprocess
        import tempfile

        rel = recipe_path.relative_to(REPO_ROOT)
        proc = subprocess.run(
            ["git", "show", f"{source_revision}:{rel}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise ValueError(
                f"{rel} not found at {source_revision}: {proc.stderr.strip()}"
            )
        fd, tmp_name = tempfile.mkstemp(suffix=".json")
        with open(fd, "w") as fh:
            fh.write(proc.stdout)
        source_path = Path(tmp_name)
        _scratch = source_path

    try:
        cfg = reconstruct(recipe_path, source_path=source_path)
    finally:
        if _scratch is not None:
            _scratch.unlink(missing_ok=True)

    if isinstance(cfg, Skip):
        raise ValueError(f"cannot recertify {recipe_path}: {cfg.reason}")

    from tuningfork.recipes._base import Effort
    from tuningfork.recipes._certification_runner import emit_low_recipe_for_cell

    return emit_low_recipe_for_cell(
        model_name=cfg.model_name,
        warmup_name=cfg.warmup_name,
        sampler_name=cfg.sampler_name,
        n_warmup=cfg.n_warmup,
        n_samples=cfg.n_samples,
        num_chains=cfg.num_chains,
        seed=seed if seed is not None else cfg.seed,
        target_acceptance=cfg.target_acceptance,
        sampler_kwargs_override=cfg.sampler_kwargs_override,
        warmup_kwargs_override=cfg.warmup_kwargs_override,
        step_policy=cfg.step_policy,
        policy_tag=cfg.policy_tag,
        effort=Effort(cfg.effort),
        warmup_inner_kernel=cfg.warmup_inner_kernel,
        init_strategy=cfg.init_strategy,
        variant_label=cfg.variant_label,
        catalog_root=catalog_root,
        verbose=verbose,
    )


# Skip reasons that are scope decisions, not reconstruction failures.
_OUT_OF_SCOPE = (
    "model is excluded",
    "SMC headline",
    "failed recipe",
    "VI base method",
    "historical MCLMC-LRD artifact",
)


def _is_reconstruction_failure(skip: Skip) -> bool:
    return not any(skip.reason.startswith(prefix) for prefix in _OUT_OF_SCOPE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Reconstruct every cell and report; emit nothing. Exits 1 if any "
        "in-scope cell cannot be reconstructed.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print one shell-quoted emit invocation per reconstructable cell",
    )
    parser.add_argument("--json", help="Write the reconstructed configs here")
    parser.add_argument(
        "--from-rev",
        help="Read configurations from this revision instead of the working "
        "tree. Use the PRE-SWEEP commit once any cell has been re-emitted.",
    )
    parser.add_argument(
        "--recertify",
        help="Re-run ONE cell (path to its recipe JSON, relative or absolute) "
        "under its own recorded configuration and overwrite it in place. Every "
        "field comes from reconstruct(), not from a hand-typed call, so a "
        "reseed cannot silently drop a kernel kwarg.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="With --recertify: master seed for the run (a disclosed reseed). "
        "Omit to replay the exact seed the source artifact's tuning_seed "
        "implies.",
    )
    parser.add_argument(
        "--source-revision",
        help="With --recertify: read the configuration to replay from this "
        "revision instead of the working tree. Required when the recipe on "
        "disk is itself the defective artifact being repaired.",
    )
    args = parser.parse_args()

    if args.recertify:
        path = Path(args.recertify)
        if not path.is_absolute():
            path = REPO_ROOT / path
        result = recertify(
            path,
            seed=args.seed,
            source_revision=args.source_revision,
        )
        print(f"verdict: {result.verdict}")
        if result.note:
            print(f"note: {result.note}")
        return 0 if result.verdict == "PASS" else 1

    ok, skipped = survey(args.from_rev)
    failures = [s for s in skipped if _is_reconstruction_failure(s)]
    out_of_scope = [s for s in skipped if not _is_reconstruction_failure(s)]

    if args.plan:
        for c in ok:
            print(json.dumps(_as_invocation(c)))
        return 0

    print(f"reconstructable cells : {len(ok)}")
    print(f"out of scope          : {len(out_of_scope)}")
    print(f"RECONSTRUCTION FAILED : {len(failures)}")

    by_flag: dict[str, int] = {}
    for c in ok:
        for flag in _config_flags(c):
            by_flag[flag] = by_flag.get(flag, 0) + 1
    print("\nnon-default configuration among reconstructable cells:")
    for flag, n in sorted(by_flag.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {flag}")

    noted = [c for c in ok if c.notes]
    if noted:
        print(f"\ncells carrying a reconstruction note ({len(noted)}):")
        for c in noted:
            print(f"  {c.key:<70} {'; '.join(c.notes)}")

    if failures:
        print("\nCELLS THAT WILL NOT BE EMITTED:")
        for s in failures:
            print(f"  {s.key:<70} {s.reason}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "reconstructable": [_as_invocation(c) for c in ok],
                    "failed": [{"cell": s.key, "reason": s.reason} for s in failures],
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")

    return 1 if failures else 0


def _config_flags(c: CellConfig) -> list[str]:
    flags = []
    if c.warmup_inner_kernel:
        flags.append("warmup_inner_kernel")
    if c.step_policy:
        flags.append("step_policy")
    if c.policy_tag:
        flags.append("policy_tag")
    if c.init_strategy:
        flags.append("init_strategy")
    if c.variant_label:
        flags.append("variant_label")
    if c.sampler_kwargs_override:
        flags.append("sampler_kwargs_override")
    if c.n_warmup != 1000:
        flags.append("non-default n_warmup")
    if c.n_samples != 1000:
        flags.append("non-default n_samples")
    if c.num_chains != 4:
        flags.append("non-default num_chains")
    if c.target_acceptance not in (None, 0.8):
        flags.append("non-default target_acceptance")
    if c.effort != "low":
        flags.append(f"effort={c.effort}")
    if c.config_correction:
        flags.append("config correction (not a replay)")
    return flags


def _as_invocation(c: CellConfig) -> dict[str, Any]:
    return {
        "cell": c.key,
        "model_name": c.model_name,
        "warmup_name": c.warmup_name,
        "sampler_name": c.sampler_name,
        "effort": c.effort,
        "n_warmup": c.n_warmup,
        "n_samples": c.n_samples,
        "num_chains": c.num_chains,
        "seed": c.seed,
        "target_acceptance": c.target_acceptance,
        "sampler_kwargs_override": c.sampler_kwargs_override,
        "warmup_kwargs_override": c.warmup_kwargs_override,
        "step_policy": c.step_policy,
        "policy_tag": c.policy_tag,
        "warmup_inner_kernel": c.warmup_inner_kernel,
        "init_strategy": c.init_strategy,
        "variant_label": c.variant_label,
        "config_correction": c.config_correction,
        "notes": c.notes,
    }


if __name__ == "__main__":
    sys.exit(main())
