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
"""Tests for Recipe emission logic (from warmup/tuning results).

This file contains all tests that run actual MCMC chains or tuning algorithms.
These are marked @pytest.mark.slow individually (not at module level, per PR-4 rules).

Tests: from_warmup_only and render_instructions_medium_real.

History: test_medium_recipe_exists_and_has_warmup_data (parametrized over 6
(model × {hmc, nuts}) combos) was removed 2026-05-17 as a slow-CI fix —
the MEDIUM placeholder recipes it asserted-existence-of had been deleted in
PR #6 commit 3 (715a82c, "recipes: delete stale low/medium/high starter
recipes"), but the test surgery in that commit missed this slow-only test
because we don't run slow locally. Real MEDIUM recipes are produced by
Recipe Phase 1+ pipeline; their existence-on-disk is no longer a test gate.
"""

from pathlib import Path

import jax
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.catalog._estimator_provenance import HEADLINE_ESTIMATOR_EXCLUDED_MODELS
from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR
from tuningfork.model import MODELS
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._instructions import render_instructions
from tuningfork.warmup import WARMUPS

# ---------------------------------------------------------------------------
# MEDIUM and HIGH constructors (require actual warmup)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_from_warmup_only_window_adaptation_diag_imm_nuts() -> None:
    """from_warmup_only with window_adaptation_diag_imm + NUTS returns a MEDIUM recipe.

    Verifies:
    - effort = MEDIUM
    - warmup_name = "window_adaptation_diag_imm"
    - base_method_params contains both step_size (from defaults) and
      inverse_mass_matrix (from warmup adaptation)
    - calibration_budget["n_warmup"] == 200
    - calibration_budget["wall_seconds_estimate"] > 0
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup = WARMUPS["window_adaptation_diag_imm"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(0),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "window_adaptation_diag_imm"
    assert recipe.model_name == "mvn_10"
    assert recipe.base_method_name == "nuts"

    # base_method_params must include both the default step_size (loguniform
    # 70th-pctile ≈ 0.126) AND the warmup-adapted inverse_mass_matrix.
    assert "step_size" in recipe.base_method_params
    assert "inverse_mass_matrix" in recipe.base_method_params

    # IMM must be a list (coerced from jax.Array by _to_jsonable).
    imm = recipe.base_method_params["inverse_mass_matrix"]
    assert isinstance(imm, list), f"inverse_mass_matrix should be list, got {type(imm)}"
    assert len(imm) == 10  # mvn_10 is 10-D

    # calibration_budget fields
    assert recipe.calibration_budget["n_warmup"] == 200
    assert recipe.calibration_budget["wall_seconds_estimate"] > 0
    assert recipe.calibration_budget["trials"] == 0

    # warmup_params records the input config
    assert recipe.warmup_params["n_warmup"] == 200

    # headline_metric is None for MEDIUM (no post-warmup samples)
    assert recipe.headline_metric is None

    # instructions must be non-empty prose
    assert isinstance(recipe.instructions, str)
    assert len(recipe.instructions) > 10


@pytest.mark.slow
def test_from_warmup_only_mclmc_tuning_metadata() -> None:
    """from_warmup_only with mclmc_tuning threads _total_tuning_steps into calibration_budget.

    Threading the ``_total_tuning_steps`` metadata key from mclmc_tuning into
    adapted_params with an underscore prefix.  from_warmup_only must capture it
    in calibration_budget and strip it from base_method_params.
    """
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["mclmc"]
    warmup = WARMUPS["mclmc_tuning"]

    recipe = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup,
        n_warmup=200,
        rng_key=jax.random.key(1),
    )

    assert recipe.effort == Effort.MEDIUM
    assert recipe.warmup_name == "mclmc_tuning"

    # _total_tuning_steps must appear in calibration_budget (threaded from metadata).
    assert "_total_tuning_steps" in recipe.calibration_budget
    assert isinstance(recipe.calibration_budget["_total_tuning_steps"], int)

    # _total_tuning_steps must NOT appear in base_method_params (stripped).
    assert "_total_tuning_steps" not in recipe.base_method_params

    # MCLMC adapted params (L, step_size) must be in base_method_params.
    assert "step_size" in recipe.base_method_params
    assert "L" in recipe.base_method_params


@pytest.mark.slow
def test_from_warmup_only_incompatible_raises() -> None:
    """from_warmup_only with an incompatible (warmup, base_method) pair raises ValueError."""
    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    # mclmc_tuning is only compatible with mclmc, not nuts.
    warmup = WARMUPS["mclmc_tuning"]

    with pytest.raises(ValueError, match="not compatible"):
        Recipe.from_warmup_only(
            posterior,
            base_method,
            warmup,
            n_warmup=100,
            rng_key=jax.random.key(0),
        )


@pytest.mark.slow
def test_render_instructions_medium_real() -> None:
    """render_instructions on a real MEDIUM recipe returns meaningful prose."""

    posterior = MODELS["mvn_10"]
    base_method = BASE_METHODS["nuts"]
    warmup_sw = WARMUPS["window_adaptation_diag_imm"]

    # --- MEDIUM ---
    medium = Recipe.from_warmup_only(
        posterior,
        base_method,
        warmup_sw,
        n_warmup=100,
        rng_key=jax.random.key(7),
    )
    prose_m = render_instructions(medium)
    assert isinstance(prose_m, str)
    assert len(prose_m) > 20
    assert "window_adaptation_diag_imm" in prose_m


# test_medium_recipe_exists_and_has_warmup_data (parametrized over 6 combos)
# removed 2026-05-17 — see module docstring "History" section.


# ---------------------------------------------------------------------------

_CATALOG = Path(__file__).parent.parent.parent / "tuningfork" / "catalog"

# grads_per_step for samplers whose grad_count_per_step is a compile-time constant.
# VI is excluded: vi.step draws from a Gaussian (0 real grads); the sampling-phase
# total_grad_evals is a step-count artefact, not a gradient count.
_CONST_GRAD_PER_STEP = {
    "mclmc": 2,
    "mclmc_lrd": 2,
    "mala": 1,
    "barker": 1,
    "ghmc": 1,
    "mgrad_gaussian": 1,
    "orbital_hmc": 1,
}


# Pre-existing hand-assembled recipes with 5-s.f. rounded basis values.
# The mismatch is a float-precision rounding artefact (relative error ~1.7e-8),
# NOT the gate-vs-headline ESS bug this test guards against.  Each entry is
# tracked as a follow-up re-emit; do not add new entries without a tracking issue.
_KNOWN_ROUNDING_DEFECTS: frozenset[str] = frozenset(
    [
        # stoch_vol flatinit: hand-assembled, no provenance stamps, basis rounded to
        # 5 s.f. (373.91).  headline_metric stored in float64, derived in float32 →
        # 1.7e-8 relative error.  Requires a proper re-emit (integrity review §4).
        "stoch_vol/low__mclmc_lrd__mclmc_lrd_tuning_flatinit.json",
    ]
)


@pytest.mark.fast
def test_catalog_headline_basis_reproduces_headline_metric() -> None:
    """Every committed recipe: headline_metric == basis.min_bulk_ess / basis.total_grad_evals.

    Both emit paths construct the basis so this holds EXACTLY (the main runner
    back-derives min_bulk_ess = headline × grad_evals; the LRD path divides the
    same headline ESS it used for the metric).  A relative tolerance of 1e-9 is
    therefore correct — a 5% tolerance silently admits a recipe whose basis was
    written from the *gate* estimator whenever the two estimators happen to agree
    numerically, which is exactly how the ill_cond_50 LRD recipe escaped review.

    Note — exact back-derivation vs direction test: the signal "basis < gate" was
    used earlier as a heuristic for the gate-vs-headline ESS bug, but it is not
    reliable: where the two estimators nearly coincide (e.g. ill_cond_50 LRD at
    gate/basis=1.0004, lgcp mclmc at 1.0012) a direction test cannot discriminate.
    The exact form (abs(hm - derived) < 1e-9 * scale) is necessary and sufficient
    because both emit paths construct basis.min_bulk_ess from the SAME value they
    used to compute headline_metric, so any mismatch is a genuine provenance error.

    BLIND SPOT — gradient-free samplers on the main runner path: the main runner's
    gradient-free branch (_recipe_runner.py around line 1985-2000) sets BOTH
    headline_metric and basis.min_bulk_ess from gate_verdict.min_bulk_ess (the gate
    estimator), making the recipe internally self-consistent.  A future
    elliptical_slice / rwm / irmh recipe would therefore PASS this test while using
    the gate estimator for its headline.  This test is a CONSISTENCY assertion, not
    a PROVENANCE assertion.  Today no gradient-free recipe carries a headline_metric
    (only failed__ stubs exist), so this blind spot is latent.  Filed separately.

    Known pre-existing rounding defects from hand-assembled recipes are listed in
    ``_KNOWN_ROUNDING_DEFECTS``; they use float32 basis values rounded to 5 s.f.
    and require a proper re-emit in a follow-up PR.
    """
    import json

    failures = []
    for p in sorted(_CATALOG.glob("*/recipes/*.json")):
        key = f"{p.parent.parent.name}/{p.name}"
        if key in _KNOWN_ROUNDING_DEFECTS:
            continue
        d = json.loads(p.read_text())
        hm = d.get("headline_metric")
        hb = d.get("headline_basis") or {}
        ess, tge = hb.get("min_bulk_ess"), hb.get("total_grad_evals")
        if hm is None or ess is None or not tge:
            continue  # failed/stub recipe, null headline (e.g. VI), or gradient-free
        derived = ess / tge
        if abs(hm - derived) > 1e-9 * max(abs(hm), abs(derived)):
            failures.append(
                f"{key}: headline_metric={hm!r} but "
                f"basis-derived={derived!r} (basis.min_bulk_ess={ess!r}, "
                f"total_grad_evals={tge}). headline_basis must store the HEADLINE "
                f"ESS (blackjax effective_sample_size), not the gate ESS (ess_bulk)."
            )
    assert (
        not failures
    ), "headline_basis does not reproduce headline_metric:\n" + "\n".join(failures)


@pytest.mark.fast
def test_catalog_constant_grad_total_grad_evals_are_exact() -> None:
    """Constant-grad samplers: total_grad_evals == n_samples * num_chains * grads_per_step.

    This is the invariant the pre-be73ad8 grad_counter bug violated (the vmapped
    per-step count collapsed to (num_chains,) and the sum came out n_samples times
    too small).  It is checkable from the artifact alone, so it is the durable
    guard against the whole class — not just the 14 recipes fixed by hand.

    VI is excluded from the map because vi.step draws from a Gaussian (0 real
    grads); the sampling-phase total_grad_evals records step counts, not gradients.
    """
    import json

    failures = []
    for p in sorted(_CATALOG.glob("*/recipes/*.json")):
        fam = p.name.split("__")[1] if "__" in p.name else ""
        if fam not in _CONST_GRAD_PER_STEP:
            continue
        d = json.loads(p.read_text())
        cb = d.get("calibration_budget") or {}
        hb = d.get("headline_basis") or {}
        ns, nc, tge = (
            cb.get("n_samples"),
            cb.get("num_chains"),
            hb.get("total_grad_evals"),
        )
        if tge is None or not ns or not nc:
            continue
        expected = ns * nc * _CONST_GRAD_PER_STEP[fam]
        if tge != expected:
            failures.append(
                f"{p.parent.parent.name}/{p.name}: total_grad_evals={tge} but "
                f"n_samples({ns}) * num_chains({nc}) * grads_per_step"
                f"({_CONST_GRAD_PER_STEP[fam]}) = {expected}"
            )
    assert not failures, "stale grad-eval counters:\n" + "\n".join(failures)


@pytest.mark.fast
def test_catalog_headline_basis_declares_the_headline_estimator() -> None:
    """No committed recipe may claim a headline estimator other than ``ess_bulk``.

    The exact-reproduction invariant above cannot see this.  A basis written from
    the wrong estimator is still internally self-consistent — it reproduces its own
    headline perfectly — so consistency checks pass it.  Only a recorded provenance
    stamp distinguishes the two, which is why ``headline_basis["ess_estimator"]``
    exists.

    Coverage grows as the corpus is re-emitted: recipes predating the stamp carry
    no ``ess_estimator`` key and are counted, not failed.  The assertion is on the
    recipes that DO declare one, so a wrong declaration fails from the first
    re-emitted cell onward.
    """
    import json

    violations = []
    stamped = 0
    unstamped = 0
    for p in sorted(_CATALOG.glob("*/recipes/*.json")):
        if p.parent.parent.name in HEADLINE_ESTIMATOR_EXCLUDED_MODELS:
            continue
        hb = json.loads(p.read_text()).get("headline_basis") or {}
        if not hb:
            continue
        declared = hb.get("ess_estimator")
        if declared is None:
            unstamped += 1
            continue
        stamped += 1
        if declared != HEADLINE_ESS_ESTIMATOR:
            violations.append(
                f"{p.parent.parent.name}/{p.name}: ess_estimator={declared!r}, "
                f"expected {HEADLINE_ESS_ESTIMATOR!r}"
            )

    assert not violations, (
        f"recipes declare a non-headline ESS estimator "
        f"({stamped} stamped, {unstamped} predate the stamp):\n" + "\n".join(violations)
    )


@pytest.mark.fast
def test_estimator_exclusions_are_live_and_still_excluded() -> None:
    """The estimator exclusion list must describe reality, in both directions.

    An allowlist that is never checked rots into a lie.  Two ways this one could:
    it could name a model the catalog no longer has (a dead entry that silently
    stops excluding anything), or the model could be re-measured onto the current
    estimator, leaving an entry that wrongly tells readers its numbers are on the
    legacy one.  Both are asserted here, so the list stays load-bearing rather
    than decorative.
    """
    import json

    problems = []
    for model, reason in HEADLINE_ESTIMATOR_EXCLUDED_MODELS.items():
        model_dir = _CATALOG / model
        if not model_dir.is_dir():
            problems.append(f"{model}: excluded but absent from the catalog")
            continue
        if not reason.strip():
            problems.append(f"{model}: excluded with an empty reason")
        for p in sorted(model_dir.glob("recipes/*.json")):
            declared = (json.loads(p.read_text()).get("headline_basis") or {}).get(
                "ess_estimator"
            )
            if declared is not None:
                problems.append(
                    f"{model}/{p.name}: declares ess_estimator={declared!r}, so it "
                    f"HAS been re-measured — drop {model} from the exclusion list "
                    f"instead of telling readers its headline is on the old estimator"
                )
    assert not problems, "estimator exclusion list is stale:\n" + "\n".join(problems)


@pytest.mark.fast
def test_catalog_headline_basis_legacy_ess_is_never_the_headline_ess() -> None:
    """Where both estimators are recorded, they must be distinct measurements.

    ``min_bulk_ess_classic_legacy`` exists to attribute a headline change to the
    estimator rather than to fresh draws.  A path that copied the headline value
    into it would make every ``estimator_ratio`` exactly 1.0 and destroy the
    attribution while looking perfectly well-formed.
    """
    import json

    failures = []
    checked = 0
    for p in sorted(_CATALOG.glob("*/recipes/*.json")):
        hb = json.loads(p.read_text()).get("headline_basis") or {}
        ess, legacy = hb.get("min_bulk_ess"), hb.get("min_bulk_ess_classic_legacy")
        ratio = hb.get("estimator_ratio")
        if ess is None or legacy is None:
            continue
        checked += 1
        if ratio is None:
            failures.append(
                f"{p.parent.parent.name}/{p.name}: both estimators recorded but "
                f"estimator_ratio is null"
            )
        elif abs(ratio - ess / legacy) > 1e-9 * max(abs(ratio), abs(ess / legacy)):
            failures.append(
                f"{p.parent.parent.name}/{p.name}: estimator_ratio={ratio!r} but "
                f"min_bulk_ess/min_bulk_ess_classic_legacy={ess / legacy!r}"
            )
    assert not failures, f"({checked} checked)\n" + "\n".join(failures)


@pytest.mark.fast
def test_lrd_emit_stores_headline_ess_not_gate_ess_in_basis(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression test: _emit_lrd_cert_sweep stores headline ESS (not gate ESS) in basis.

    Mocks a cert-seed result whose gate ESS (500) and headline ESS (125) differ 4×
    and whose ess_per_grad is consistent with the HEADLINE ESS.  Reverting
    emit_mclmc_lrd.py to ``best["min_bulk_ess"]`` makes this fail.

    On a real run the two keys now hold the same estimator, so the mock keeps them
    apart deliberately: reading the gate key would be a provenance error even
    where the numbers happen to agree, and only a mock can still see it.
    """
    pytest.importorskip("numpyro")
    from blackjax.mcmc.integrators import LowRankInverseMassMatrix

    import tuningfork.recipes.emit_mclmc_lrd as _mod
    from tuningfork.recipes import Recipe

    sigma_row = jax.numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    imm = LowRankInverseMassMatrix(
        sigma=jax.numpy.stack([sigma_row, sigma_row * 2]),
        U=jax.numpy.stack([jax.numpy.eye(5, 3), jax.numpy.eye(5, 3)]),
        lam=jax.numpy.stack([jax.numpy.array([1.0, 0.5, 0.25])] * 2),
    )
    fake = {
        "seed": 42,
        "verdict": "PASS",
        "rhat_max": 1.001,
        "min_bulk_ess": 500.0,  # GATE ESS (separate leaf traversal)
        "min_bulk_ess_headline": 125.0,  # HEADLINE ESS (metrics.headline)
        "min_bulk_ess_classic_legacy": 100.0,  # pre-switch estimator, same draws
        "n_divergences": 0,
        "div_rate": 0.0,
        "ess_per_grad": 0.025,  # == 125 / 5000, i.e. headline-consistent
        "total_grad_evals": 5000,
        "wall_seconds": 0.5,
        "adapted_params": {
            "step_size": jax.numpy.array(0.12345678),
            "L": jax.numpy.array(9.87654321),
            "inverse_mass_matrix": imm,
        },
    }
    monkeypatch.setattr(_mod, "_run_cert_seed", lambda **_kw: fake)

    written = _mod._emit_lrd_cert_sweep(
        ["ill_cond_50"],
        cert_seeds=(42,),
        n_warmup=100,
        n_samples=10,
        num_chains=2,
        k_rank=3,
        catalog_root=tmp_path,
        variant_label="mclmc_lrd",
    )
    recipe = Recipe.load(written[0])
    basis = recipe.headline_basis or {}
    assert basis["min_bulk_ess"] == fake["min_bulk_ess_headline"], (
        f"headline_basis.min_bulk_ess={basis['min_bulk_ess']} — expected the HEADLINE "
        f"ESS {fake['min_bulk_ess_headline']}, got the GATE ESS. emit_mclmc_lrd must "
        "read best['min_bulk_ess_headline']."
    )
    derived = basis["min_bulk_ess"] / basis["total_grad_evals"]
    assert abs(recipe.headline_metric - derived) < 1e-12
    assert basis["ess_estimator"] == "ess_bulk"
    assert basis["min_bulk_ess_classic_legacy"] == 100.0
    assert basis["estimator_ratio"] == pytest.approx(125.0 / 100.0)
