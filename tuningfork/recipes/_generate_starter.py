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
"""Starter recipe emission — recipe-generation pipeline.

.. note::

   As of recipe generation launch (2026-05-17), this script targets the
   *real* LOW-effort cells (gate-passes-at-first-emit), not the earlier
   placeholder recipes which were deleted on the cleanup-and-simplify branch.
   The MEDIUM / HIGH escalation helpers (``emit_medium_recipes`` /
   ``emit_high_recipes``) remain for Statistician-driven escalation.

This script generates candidate recipes for the conventional cells in the
``(model, warmup, sampler)`` space — pairs where the warmup is the natural
adaptation procedure for the sampler (see ``NATURAL_WARMUP_FOR_SAMPLER`` below).
Each candidate is a ``Recipe`` with ``effort=Effort.LOW``; the Statistician
auto-gate (``tuningfork.calibration.statistician_gate``) decides whether to
commit (PASS) or escalate to MEDIUM / HIGH.  See ``_base.py``'s ``Effort``
docstring for the per-tier semantics.

Usage
-----

  cd tuningfork
  uv run python -m tuningfork.recipes._generate_starter
  uv run python -m tuningfork.recipes._generate_starter --only radon
  uv run python -m tuningfork.recipes._generate_starter --sampler nuts

Flag semantics:
  - ``--only <m>``    restrict to one model (must be in STARTER_MODEL_NAMES)
  - ``--warmup <w>``  restrict to one warmup (``"no_warmup"`` or ``"window_adaptation_diag_imm"``)
  - ``--sampler <s>`` restrict to one base method

The script is idempotent: re-running overwrites existing files with fresh
provenance timestamps.

MEDIUM / HIGH escalation
------------------------

The ``emit_medium_recipes`` and ``emit_high_recipes`` functions are helpers
the Statistician can call during escalation; they are NOT exposed via the CLI.
"""

import time
from pathlib import Path

import jax

from tuningfork._version import __version__ as _tuningfork_version
from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.recipes._base import Recipe
from tuningfork.recipes.emit_mclmc_lrd import _emit_mclmc_lrd_recipes_impl
from tuningfork.warmup import WARMUPS

# Starter model suite: 14 models covering different dimensionalities and geometry types.
# Models are organized by complexity:
# - Simple Gaussians: mvn_10, ill_cond_50 (ill-conditioned)
# - Nonlinear: neals_funnel, banana, eight_schools_ncp
# - Discrete mixture: gmm_25 (25-mode Gaussian mixture)
# - Logistic regression: logistic_synthetic, german_credit
# - High-dimensional: horseshoe (204-D), radon (391-D), irt_2pl (144-D)
# - State-space: stoch_vol (503-D)
# - Differential equations: lotka_volterra (7-D), gp_regression (203-D)
STARTER_MODEL_NAMES = [
    "mvn_10",
    "ill_cond_50",
    "neals_funnel",
    "eight_schools_ncp",
    "banana",
    "gmm_25",
    "logistic_synthetic",
    "german_credit",
    "horseshoe",
    "radon",
    "irt_2pl",
    "stoch_vol",
    "irt_1pl",
    "lgcp",
    "lotka_volterra",
    "gp_regression",
]

# Base methods used in starter recipes (LOW tier: emit_low_recipes)
# gradient-free / no-warmup methods included so Recipe.from_default_config
# stubs them out.  elliptical_slice is listed here; it requires
# extra_required_kwargs (prior_cov/prior_mean) for actual emission — the
# specialised wiring lands with the statistician's model+gate spec (Phase 8B.3).
ALL_METHOD_NAMES = ["hmc", "nuts", "mala", "barker", "rwm", "mclmc", "elliptical_slice"]

# Methods eligible for MEDIUM warmup-only recipes (window_adaptation_diag_imm
# compatibility).  rmhmc added in Phase 8B.3; its sampler template lives at
# _templates/samplers/rmhmc.py.tmpl (IMM→mass_matrix conversion inlined).
MEDIUM_METHOD_NAMES = ["nuts", "hmc", "rmhmc"]

# Methods eligible for MCLMC-LRD recipes (mclmc_lrd_tuning compatibility).
# Only mclmc is compatible; listed as a sequence for symmetry with
# MEDIUM_METHOD_NAMES and to support future additions.
MCLMC_LRD_METHOD_NAMES = ["mclmc"]

# ---------------------------------------------------------------------------
# Conventional pairing map
# ---------------------------------------------------------------------------
# The "natural" warmup for each base method — the warmup that the BlackJAX
# community / literature pairs with that sampler by default.  Under the
# Effort taxonomy (see _base.py), LOW recipes operate on these conventional
# pairings; MEDIUM recipes explore unconventional but technically-possible
# combinations (e.g., window_adaptation_diag_imm + mala, window_adaptation_diag_imm + rmhmc); HIGH recipes
# add tuned warmup HPs and model-specific injection.
NATURAL_WARMUP_FOR_SAMPLER: dict[str, str] = {
    # Window-adaptation family (window_adaptation_diag_imm's compatible_methods)
    "hmc": "window_adaptation_diag_imm",
    "nuts": "window_adaptation_diag_imm",
    "mhmc": "window_adaptation_diag_imm",
    "barker": "window_adaptation_diag_imm",
    "mala": "window_adaptation_diag_imm",
    "rmhmc": "window_adaptation_diag_imm",
    # MCLMC family
    "mclmc": "mclmc_tuning",
    "adjusted_mclmc": "adjusted_mclmc_tuning",
    "adjusted_mclmc_dynamic": "adjusted_mclmc_tuning",
    # Multi-chain adaptation
    "ghmc": "meads",
    "dynamic_hmc": "chees",
    "dmhmc": "chees",
    # Gradient-free / specialised — no canonical warmup
    "rwm": "no_warmup",
    "irmh": "no_warmup",
    "additive_step_random_walk": "no_warmup",
    "elliptical_slice": "no_warmup",
    "mgrad_gaussian": "no_warmup",
    "orbital_hmc": "no_warmup",
    "laplace_hmc": "no_warmup",
    "laplace_dhmc": "no_warmup",
    "laplace_mhmc": "no_warmup",
    "laplace_dmhmc": "no_warmup",
    # VI as base method (the warmup *is* the optimization; no separate warmup)
    "meanfield_vi": "no_warmup",
    "fullrank_vi": "no_warmup",
}

# Root of the starter/ directory (relative to this file)
_CATALOG_ROOT = Path(__file__).parent.parent / "catalog"


def emit_low_recipes(
    seed: int = 0,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit LOW-effort candidate recipes with default sampler configuration.

    Creates starter recipes with library defaults; candidates are evaluated by
    the Statistician auto-gate (``tuningfork.calibration.statistician_gate``)
    to determine if they pass or require escalation to MEDIUM/HIGH.

    Idempotent — overwrites existing ``low__*.json`` files in place.

    Parameters
    ----------
    seed
        Random seed for potential future use (currently unused; LOW is deterministic).
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``ALL_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    _ = seed  # currently unused; kept for symmetry with emit_medium_recipes
    generated: list[Path] = []
    repo_root = Path(__file__).parent.parent.parent.parent

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in ALL_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]
            recipe = Recipe.from_default_config(
                posterior,
                base_method,
                tuningfork_version=_tuningfork_version,
            )
            path = recipe.save(_CATALOG_ROOT)
            generated.append(path)
            try:
                pretty = path.relative_to(repo_root)
            except ValueError:
                # _CATALOG_ROOT was redirected outside the repo (e.g. tests using tmp_path);
                # fall back to absolute path so logging never crashes.
                pretty = path
            print(f"  LOW   {pretty}")

    return generated


def emit_medium_recipes(
    seed: int = 0,
    n_warmup: int = 1000,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit MEDIUM-effort candidate recipes using window_adaptation_diag_imm warmup adaptation.

    Runs warmup with window_adaptation_diag_imm adaptation to produce tuned step-size and
    inverse-mass-matrix; these candidates are evaluated by the Statistician
    auto-gate to assess improvement over LOW baselines.

    Idempotent: re-running overwrites with deterministic content (same seed → same key).

    Compatibility is limited by WARMUPS["window_adaptation_diag_imm"].compatible_methods
    to samplers that support window adaptation (hmc, nuts, barker, mala).
    MEDIUM recipes are emitted for nuts and hmc in the starter set.

    Parameters
    ----------
    seed
        Base random seed; fold_in derives per-recipe keys deterministically.
    n_warmup
        Number of warmup adaptation steps (default 1000).
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``MEDIUM_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    window_adaptation_diag_imm = WARMUPS["window_adaptation_diag_imm"]
    generated: list[Path] = []
    repo_root = Path(__file__).parent.parent.parent.parent

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]

            # Check compatibility
            if not window_adaptation_diag_imm.is_compatible(method_name):
                print(
                    f"  SKIP  {model_name}/{method_name}: "
                    f"window_adaptation_diag_imm incompatible"
                )
                continue

            # Derive a deterministic per-recipe key via fold_in.
            # hash() can produce values outside uint32; mask to 32 bits.
            hash_val = hash((model_name, method_name)) & 0xFFFFFFFF
            key = jax.random.fold_in(jax.random.key(seed), hash_val)

            recipe = Recipe.from_warmup_only(
                posterior,
                base_method,
                window_adaptation_diag_imm,
                n_warmup=n_warmup,
                rng_key=key,
                tuningfork_version=_tuningfork_version,
            )
            path = recipe.save(_CATALOG_ROOT)
            generated.append(path)
            try:
                pretty = path.relative_to(repo_root)
            except ValueError:
                pretty = path
            print(f"  MEDIUM {pretty}")

    return generated


def emit_mclmc_lrd_recipes(
    seed: int = 0,
    n_warmup: int = 1000,
    model_names: list[str] | None = None,
    sampler: str | None = None,
    *,
    calibrate: bool = False,
    cert_seeds: tuple[int, ...] = (11111, 22222, 33333),
    n_samples: int = 1000,
    num_chains: int = 4,
    k_rank: int = 40,
    pilot_n_warmup: int = 1000,
    pilot_n_samples: int = 1000,
) -> list[Path]:
    """Emit MCLMC-LRD candidate recipes using mclmc_lrd_tuning warmup.

    Thin delegate to
    ``tuningfork.recipes.emit_mclmc_lrd._emit_mclmc_lrd_recipes_impl``.
    This is the single documented public entry point for LRD recipe emission;
    all implementation logic lives in that module.

    ``calibrate=False`` (default): emit a single MEDIUM-effort stub recipe per
    model — one LRD warmup run (NUTS pilot → rank-k SVD →
    ``mclmc_find_L_and_step_size``) with a deterministic per-recipe key.

    ``calibrate=True``: run the full 3-seed cert sweep, gate on R̂/ESS/div,
    bake the best PASS seed into a LOW recipe with LRD IMM sidecar.

    Idempotent: re-running overwrites existing files with fresh provenance.

    Parameters
    ----------
    seed
        Base random seed for the ``calibrate=False`` stub path.  Ignored when
        ``calibrate=True`` (use ``cert_seeds`` instead).
    n_warmup
        Number of LRD adaptation steps (``lrd_num_steps`` in upstream).
        Default 1000.
    model_names
        Restrict to these models.  ``None`` = all ``STARTER_MODEL_NAMES``.
    sampler
        Restrict to this base-method name (e.g. ``"mclmc"``).
        ``None`` = all of ``MCLMC_LRD_METHOD_NAMES``.
    calibrate
        ``False``: emit stub.  ``True``: run full cert sweep.
    cert_seeds
        Seeds for the cert sweep.  Used only when ``calibrate=True``.
        Default ``(11111, 22222, 33333)``.
    n_samples
        Post-warmup samples per chain (``calibrate=True`` only).
    num_chains
        Chains for the gate check (``calibrate=True`` only).
    k_rank
        LRD approximation rank.  Default 40.
    pilot_n_warmup
        Diagonal MCLMC pilot warmup steps (``pilot_num_warmup`` in upstream).
        Default 1000.  Certified configs: german_credit 5000, ill_cond_50 1000.
    pilot_n_samples
        Pilot samples for SVD geometry estimation (``pilot_num_samples`` in
        upstream).  Default 1000.  Certified configs: german_credit 5000,
        ill_cond_50 10000.

    Returns
    -------
    list[Path]
        Paths of written recipe JSON files.
    """
    repo_root = Path(__file__).parent.parent.parent.parent
    names: list[str] = (
        list(model_names) if model_names is not None else list(STARTER_MODEL_NAMES)
    )
    paths = _emit_mclmc_lrd_recipes_impl(
        names,
        calibrate=calibrate,
        seed=seed,
        cert_seeds=cert_seeds,
        n_warmup=n_warmup,
        n_samples=n_samples,
        num_chains=num_chains,
        k_rank=k_rank,
        pilot_n_warmup=pilot_n_warmup,
        pilot_n_samples=pilot_n_samples,
        sampler=sampler,
        catalog_root=_CATALOG_ROOT,
        tuningfork_version=_tuningfork_version,
    )
    label = "MCLMC_LRD(cal)" if calibrate else "MCLMC_LRD"
    for p in paths:
        try:
            pretty = p.relative_to(repo_root)
        except ValueError:
            pretty = p
        print(f"  {label} {pretty}")
    return paths


def emit_high_recipes(
    seed: int = 0,
    n_trials: int = 20,
    model_names: list[str] | None = None,
    sampler: str | None = None,
) -> list[Path]:
    """Emit HIGH-effort candidate recipes using Bayesian optimization over sampler hyperparameters.

    Runs BO tuning Bayesian optimization (via ``tune_algorithm``) to search for
    improved sampler hyperparameters with window_adaptation_diag_imm warmup; the result is
    a HIGH-effort recipe with tuned parameters and difficulty profile.
    Candidates are evaluated by the Statistician auto-gate to assess whether
    the extra optimization effort improved the headline metric.

    Runs ``tune_algorithm`` at ``n_trials=20`` (starter default; production
    recipes use 50).  Converts each ``TuningResult`` to a HIGH ``Recipe``
    via ``Recipe.from_tuning_result``.

    Compatibility is limited to ``nuts`` and ``hmc`` in the starter set.

    Parameters
    ----------
    seed
        Base random seed; ``jax.random.fold_in`` derives per-recipe keys
        deterministically from ``(model_name, method_name, "high")``.
    n_trials
        Total Optuna trials per study (including the injected default trial
        0).  Default is 20 for starter recipes; production uses 50.
    model_names
        If set, restrict to this list of model names.  ``None`` = all
        ``STARTER_MODEL_NAMES``.
    sampler
        If set, restrict to this single base-method name (e.g., ``"nuts"``).
        ``None`` = iterate all of ``MEDIUM_METHOD_NAMES``.

    Returns
    -------
    List of Path objects pointing to written JSON files.
    """
    from tuningfork.calibration.tune import tune_algorithm

    window_adaptation_diag_imm = WARMUPS["window_adaptation_diag_imm"]
    generated: list[Path] = []

    for model_name in model_names or STARTER_MODEL_NAMES:
        posterior = MODELS[model_name]
        for method_name in MEDIUM_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]

            if not window_adaptation_diag_imm.is_compatible(method_name):
                print(
                    f"  SKIP  {model_name}/{method_name}: window_adaptation_diag_imm incompatible"
                )
                continue

            # Derive a deterministic per-recipe key via fold_in.
            # hash() can produce values outside uint32; mask to 32 bits.
            hash_val = hash((model_name, method_name, "high")) & 0xFFFFFFFF
            key = jax.random.fold_in(jax.random.key(seed), hash_val)

            print(f"  tuning {model_name} + {method_name} " f"(n_trials={n_trials})...")
            t0 = time.perf_counter()
            result = tune_algorithm(
                posterior,
                base_method,
                n_trials=n_trials,
                n_seeds=2,
                n_chains=2,
                n_samples=400,
                n_warmup=500,
                rng_key=key,
                sampler="tpe",
                warmup_name="window_adaptation_diag_imm",
            )
            elapsed = time.perf_counter() - t0

            recipe = Recipe.from_tuning_result(
                result,
                posterior=posterior,
                base_method=base_method,
                warmup=window_adaptation_diag_imm,
                tuningfork_version=_tuningfork_version,
            )
            path = recipe.save(_CATALOG_ROOT)
            generated.append(path)
            print(
                f"    done in {elapsed:.1f}s; "
                f"best_score={result.best_score:.4f}; "
                f"default_works={result.difficulty.default_works}"
            )

    return generated


def main() -> None:
    """Emit candidate recipes for the Statistician auto-gate.

    See the module docstring for the full Usage section.  Three filter flags
    (``--only``, ``--warmup``, ``--sampler``) compose intersectionally; each
    defaults to "all" and narrows the matrix when set.

    Effort is gate-driven: only LOW candidates are emitted here by default;
    MEDIUM and HIGH recipes come from Statistician escalation workflows.
    """
    import argparse

    valid_warmups = {
        "no_warmup",
        "window_adaptation_diag_imm",
        "mclmc_tuning",
        "mclmc_lrd_tuning",
    }
    # MEDIUM_METHOD_NAMES (rmhmc) are not in ALL_METHOD_NAMES but must be
    # reachable via --sampler so emit_medium_recipes can be targeted directly.
    valid_samplers = set(ALL_METHOD_NAMES) | set(MEDIUM_METHOD_NAMES)

    parser = argparse.ArgumentParser(
        description=(
            "Emit candidate recipes (LOW=default config, "
            "MEDIUM=warmup-tuned, HIGH=BO-optimized).  All three filters "
            "compose; each defaults to 'all'.  MEDIUM/HIGH recipes come from "
            "Statistician escalation workflows when LOW gate fails or "
            "exploration of unconventional pairings is needed."
        )
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Restrict to one model.  Must be in STARTER_MODEL_NAMES.  "
            "Default: all starter models."
        ),
    )
    parser.add_argument(
        "--warmup",
        default=None,
        choices=sorted(valid_warmups),
        help=(
            "Restrict to one warmup.  'no_warmup' applies to the LOW pass "
            "for samplers without trajectory adaptation; 'window_adaptation_diag_imm' "
            "applies to NUTS/HMC and other window-compatible samplers; "
            "'mclmc_lrd_tuning' runs the LRD-preconditioned MCLMC warmup "
            "(NUTS pilot + SVD + vmapped mclmc_find_L_and_step_size) for mclmc.  "
            "Default: all (no_warmup + window_adaptation_diag_imm); "
            "mclmc_lrd_tuning must be requested explicitly."
        ),
    )
    parser.add_argument(
        "--sampler",
        default=None,
        choices=sorted(valid_samplers),
        help=(
            "Restrict to one base method.  Default: all of "
            "{hmc, nuts, mala, barker, rwm, mclmc}."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help=(
            "Run the full cert sweep when emitting mclmc_lrd_tuning recipes "
            "(warmup + 4-chain sampling, R̂/ESS/div gate, bake best seed).  "
            "Requires --warmup mclmc_lrd_tuning.  Default: emit stub recipe only."
        ),
    )
    parser.add_argument(
        "--cert-seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Space-separated integer seeds for the cert sweep "
            "(e.g. --cert-seeds 11111 22222 33333).  "
            "Used only with --calibrate.  Default: (11111, 22222, 33333)."
        ),
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of warmup steps for the cert sweep (mclmc_lrd_tuning).  "
            "Used only with --warmup mclmc_lrd_tuning.  "
            "Default: 1000."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of post-warmup samples per chain for the gate check.  "
            "Used only with --warmup mclmc_lrd_tuning --calibrate.  "
            "Default: 1000."
        ),
    )
    parser.add_argument(
        "--k-rank",
        type=int,
        default=None,
        metavar="K",
        help=(
            "LRD approximation rank.  "
            "Used only with --warmup mclmc_lrd_tuning.  "
            "Default: 40."
        ),
    )
    parser.add_argument(
        "--pilot-n-warmup",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Diagonal MCLMC pilot warmup steps (pilot_num_warmup in upstream).  "
            "Used only with --warmup mclmc_lrd_tuning --calibrate.  "
            "Default: 1000.  Certified configs: german_credit 5000, "
            "ill_cond_50 1000."
        ),
    )
    parser.add_argument(
        "--pilot-n-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Pilot samples for SVD geometry estimation "
            "(pilot_num_samples in upstream).  "
            "Used only with --warmup mclmc_lrd_tuning --calibrate.  "
            "Default: 1000.  Certified configs: german_credit 5000, "
            "ill_cond_50 10000."
        ),
    )
    args = parser.parse_args()

    # ── Validation ──────────────────────────────────────────────────────────
    if args.only is not None and args.only not in STARTER_MODEL_NAMES:
        raise SystemExit(
            f"--only {args.only!r} not in STARTER_MODEL_NAMES "
            f"= {STARTER_MODEL_NAMES}"
        )

    names: list[str] | None = [args.only] if args.only is not None else None

    # The warmup filter selects which pass to run.
    # `no_warmup`                → emit_low_recipes   (no adaptation; identity warmup)
    # `window_adaptation_diag_imm` → emit_medium_recipes (window adaptation)
    # `mclmc_lrd_tuning`         → emit_mclmc_lrd_recipes (LRD MCLMC warmup;
    #                                NOT included in the default run — must be
    #                                requested explicitly via --warmup mclmc_lrd_tuning
    #                                because the NUTS pilot makes it substantially
    #                                more expensive than the window_adaptation pass)
    do_no_warmup = args.warmup in (None, "no_warmup")
    do_window_adaptation_diag_imm = args.warmup in (None, "window_adaptation_diag_imm")
    do_mclmc_lrd_tuning = args.warmup == "mclmc_lrd_tuning"

    # ── Echo selection ──────────────────────────────────────────────────────
    selection = []
    if args.only is not None:
        selection.append(f"model={args.only}")
    if args.warmup is not None:
        selection.append(f"warmup={args.warmup}")
    if args.sampler is not None:
        selection.append(f"sampler={args.sampler}")
    if args.calibrate:
        selection.append("calibrate=True")
    if selection:
        print(f"Emitting candidates filtered by: {', '.join(selection)}")
    else:
        print("Emitting ALL default candidates (no filters set).")

    no_warmup_paths: list[Path] = []
    window_adaptation_diag_imm_paths: list[Path] = []
    mclmc_lrd_paths: list[Path] = []

    if do_no_warmup:
        print(
            "\nEmitting candidates for warmup=no_warmup "
            f"({'all algorithms' if args.sampler is None else args.sampler})..."
        )
        no_warmup_paths = emit_low_recipes(model_names=names, sampler=args.sampler)

    if do_window_adaptation_diag_imm:
        print(
            "\nEmitting candidates for warmup=window_adaptation_diag_imm "
            f"({'nuts/hmc' if args.sampler is None else args.sampler})..."
        )
        window_adaptation_diag_imm_paths = emit_medium_recipes(
            model_names=names, sampler=args.sampler
        )

    if do_mclmc_lrd_tuning:
        print(
            "\nEmitting candidates for warmup=mclmc_lrd_tuning "
            f"({'mclmc' if args.sampler is None else args.sampler})..."
        )
        _cert_seeds = (
            tuple(args.cert_seeds)
            if args.cert_seeds is not None
            else (11111, 22222, 33333)
        )
        _extra_kwargs: dict = {}
        if args.n_warmup is not None:
            _extra_kwargs["n_warmup"] = args.n_warmup
        if args.n_samples is not None:
            _extra_kwargs["n_samples"] = args.n_samples
        if args.k_rank is not None:
            _extra_kwargs["k_rank"] = args.k_rank
        if args.pilot_n_warmup is not None:
            _extra_kwargs["pilot_n_warmup"] = args.pilot_n_warmup
        if args.pilot_n_samples is not None:
            _extra_kwargs["pilot_n_samples"] = args.pilot_n_samples
        mclmc_lrd_paths = emit_mclmc_lrd_recipes(
            model_names=names,
            sampler=args.sampler,
            calibrate=args.calibrate,
            cert_seeds=_cert_seeds,
            **_extra_kwargs,
        )

    total = (
        len(no_warmup_paths)
        + len(window_adaptation_diag_imm_paths)
        + len(mclmc_lrd_paths)
    )
    print(
        f"\n✓ Emitted {len(no_warmup_paths)} no_warmup + "
        f"{len(window_adaptation_diag_imm_paths)} window_adaptation_diag_imm + "
        f"{len(mclmc_lrd_paths)} mclmc_lrd = {total} candidates.  "
        f"Next: Statistician gate."
    )


if __name__ == "__main__":
    main()
