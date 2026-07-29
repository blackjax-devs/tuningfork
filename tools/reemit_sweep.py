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

# The low-rank-diagonal certification sweep's own defaults, used when a recipe
# of that family records its budget as per-seed evidence rather than flat fields.
_LRD_DEFAULT_N_SAMPLES = 1000
_LRD_DEFAULT_NUM_CHAINS = 4

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
    "lotka_volterra/low__nuts__window_adaptation_low_rank_imm.json": (
        "headline was stamped from cached chain statistics, so the artifact "
        "records no sample budget at all; emitted under the standard protocol"
    ),
}


@dataclass
class CellConfig:
    """A reconstructed emit call for one committed recipe."""

    recipe_path: Path
    model_name: str
    warmup_name: str
    sampler_name: str
    effort: str
    harness: str
    n_warmup: int
    n_samples: int
    num_chains: int
    seed: int
    target_acceptance: float | None = None
    sampler_kwargs_override: dict[str, Any] | None = None
    step_policy: dict[str, Any] | None = None
    policy_tag: str | None = None
    warmup_inner_kernel: str | None = None
    init_strategy: dict[str, Any] | None = None
    config_correction: bool = False
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

    implicit = "nuts" if sampler_name in WARMUP_SUBSTITUTE_METHOD_NAMES else sampler_name
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
    - the certification-sweep constants.

    A brute-force search over master seeds up to 4e7 confirmed these account for
    every ``tuning_seed`` in the committed corpus; anything else is skipped rather
    than guessed at.
    """
    from tuningfork.recipes._recipe_runner import RECIPE_SEED

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
    from tuningfork.calibration.tune import default_params_for
    from tuningfork.recipes._recipe_runner import _RECIPE_PROVENANCE_KEYS

    pinned = dict(recipe.get("base_method_params") or {})
    defaults = default_params_for(base_method)
    notes: list[str] = []

    # Never replayable: adapted by warmup, non-serialisable, or consumer-only.
    for k in ("step_size", "inverse_mass_matrix", "L", *_RECIPE_PROVENANCE_KEYS):
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


def reconstruct(recipe_path: Path) -> CellConfig | Skip:
    """Rebuild the emit call for one committed recipe, or decline to."""
    from tuningfork.base_method import BASE_METHODS
    from tuningfork.catalog._estimator_provenance import (
        HEADLINE_ESTIMATOR_EXCLUDED_MODELS,
    )
    from tuningfork.model import MODELS
    from tuningfork.recipes._recipe_runner import RECIPE_SEED
    from tuningfork.warmup import WARMUPS

    def skip(reason: str) -> Skip:
        return Skip(recipe_path, reason)

    recipe = json.loads(recipe_path.read_text())
    model_name = recipe_path.parent.parent.name

    if model_name in HEADLINE_ESTIMATOR_EXCLUDED_MODELS:
        return skip("model is excluded from the estimator migration")
    if recipe_path.name.startswith("smc__"):
        return skip("SMC headline is an importance-weight ESS, not an autocorrelation one")

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

    warmups = recipe.get("warmups") or []
    warmup_name = recipe.get("warmup_name") or (warmups[0].get("name") if warmups else "")
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
    warmup_params = recipe.get("warmup_params") or {}
    n_warmup = warmup_params.get("n_warmup", budget.get("n_warmup"))
    n_samples = budget.get("n_samples", warmup_params.get("n_samples"))
    num_chains = warmup_params.get("num_chains", budget.get("num_chains"))

    # The low-rank-diagonal family is produced by its own certification sweep,
    # not by emit_low_recipe_for_cell, and records its budget as per-seed
    # evidence rather than flat fields.  Route it, do not fail it.
    warmup_inner_kernel = recipe.get("warmup_inner_kernel")
    override, notes, blocker = _reconstruct_sampler_kwargs(
        recipe, base_method, warmup_inner_kernel
    )
    if blocker is not None:
        return skip(blocker)

    harness = "recipe_runner"
    if sampler_name == "mclmc_lrd" or warmup_name == "mclmc_lrd_tuning":
        harness = "mclmc_lrd"
        n_samples = n_samples if n_samples is not None else _LRD_DEFAULT_N_SAMPLES
        num_chains = num_chains if num_chains is not None else _LRD_DEFAULT_NUM_CHAINS
    cell_key = f"{model_name}/{recipe_path.name}"
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
        recipe_path,
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
    if harness == "mclmc_lrd":
        # This family stamps a literal certification seed rather than deriving one,
        # so the derived-seed check does not apply; the sweep reruns its own
        # certification and the recorded seed identifies which one won.
        notes.append(f"certification sweep; recorded cert seed {committed_seed}")
        committed_seed = None
    if committed_seed:
        match = next(
            (s for s in _seed_candidates() if _predicted_tuning_seed(s) == committed_seed),
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
        harness=harness,
        n_warmup=int(n_warmup),
        n_samples=int(n_samples),
        num_chains=int(num_chains),
        seed=seed,
        target_acceptance=warmup_params.get("target_acceptance"),
        sampler_kwargs_override=override,
        step_policy=recipe.get("step_policy"),
        policy_tag=policy_tag,
        warmup_inner_kernel=warmup_inner_kernel,
        init_strategy=recipe.get("init_strategy"),
        config_correction=config_correction,
        notes=notes,
    )


def survey() -> tuple[list[CellConfig], list[Skip]]:
    """Reconstruct every catalog recipe; partition into emittable and skipped."""
    ok: list[CellConfig] = []
    skipped: list[Skip] = []
    for p in sorted(CATALOG.glob("*/recipes/*.json")):
        result = reconstruct(p)
        (ok if isinstance(result, CellConfig) else skipped).append(result)  # type: ignore[arg-type]
    return ok, skipped


# Skip reasons that are scope decisions, not reconstruction failures.
_OUT_OF_SCOPE = (
    "model is excluded",
    "SMC headline",
    "failed recipe",
    "VI base method",
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
    args = parser.parse_args()

    ok, skipped = survey()
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
    if c.harness != "recipe_runner":
        flags.append(f"harness={c.harness}")
    return flags


def _as_invocation(c: CellConfig) -> dict[str, Any]:
    return {
        "cell": c.key,
        "model_name": c.model_name,
        "warmup_name": c.warmup_name,
        "sampler_name": c.sampler_name,
        "effort": c.effort,
        "harness": c.harness,
        "n_warmup": c.n_warmup,
        "n_samples": c.n_samples,
        "num_chains": c.num_chains,
        "seed": c.seed,
        "target_acceptance": c.target_acceptance,
        "sampler_kwargs_override": c.sampler_kwargs_override,
        "step_policy": c.step_policy,
        "policy_tag": c.policy_tag,
        "warmup_inner_kernel": c.warmup_inner_kernel,
        "init_strategy": c.init_strategy,
        "config_correction": c.config_correction,
        "notes": c.notes,
    }


if __name__ == "__main__":
    sys.exit(main())
