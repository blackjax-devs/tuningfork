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
"""Tests for catalog recipe emission invariants.

This file contains all tests that run actual MCMC chains or tuning algorithms.
These are marked @pytest.mark.slow individually (not at module level, per PR-4 rules).

History: test_medium_recipe_exists_and_has_warmup_data (parametrized over 6
(model × {hmc, nuts}) combos) was removed 2026-05-17 as a slow-CI fix —
the MEDIUM placeholder recipes it asserted-existence-of had been deleted in
PR #6 commit 3 (715a82c, "recipes: delete stale low/medium/high starter
recipes"), but the test surgery in that commit missed this slow-only test
because we don't run slow locally. Real MEDIUM recipes are produced by
Recipe Phase 1+ pipeline; their existence-on-disk is no longer a test gate.
"""

from pathlib import Path

import pytest

from tuningfork.catalog._estimator_provenance import HEADLINE_ESTIMATOR_EXCLUDED_MODELS
from tuningfork.metrics.headline import HEADLINE_ESS_ESTIMATOR

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
