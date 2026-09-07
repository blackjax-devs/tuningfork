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
"""Bounded fixed-vs-randomised trajectory-length sensitivity example.

This is the reproducible example for
``docs/examples/trajectory-length-sensitivity.md``: one warmup is paid, its
geometry is frozen, and two sampling arms that differ **only** in their
trajectory-length distribution run against it.  Both arms report the same
first-moment, even (squared) and cross expectands, so a first-moment-only
comparison cannot hide a slower squared or cross function.

Everything runs through codegen -- ``Recipe.step_policy`` and ``no_warmup`` with
a pinned ``step_size``/``inverse_mass_matrix`` already express "randomised L on
geometry frozen after warmup", so no hand-written sampling script is involved.

**These assertions are structural.** The example exists to show that the
comparison is composed and costed correctly, not to decide whether randomising
the trajectory length helps. Nothing here asserts a winner, and no observed
direction from one model, one geometry and one seed generalises. The direction
actually observed on this Gaussian, with its conditions and limitations, is
recorded in the accompanying document.
"""

from __future__ import annotations

import numpy as np
import pytest

from tuningfork.catalog import (
    CostAccounting,
    compare_reports,
    execute_recipe,
    expectand_report,
    sampling_grad_evals_from_chain_stats,
)
from tuningfork.recipes import Effort, Recipe
from tuningfork.recipes._sample_stats import SAMPLE_STAT_PREFIX

pytestmark = pytest.mark.e2e

MODEL = "mvn_10"
SEED = 20260907
N_WARMUP = 200
N_DRAWS = 300
N_CHAINS = 4
FIXED_L = 4
POLICY = {"kind": "uniform_int", "low": 2, "high": 7}
CPU = {"JAX_PLATFORM_NAME": "cpu"}

#: First moment, even (squared) and cross quantities of the same draws.  Shared
#: verbatim by both arms -- a comparison over different expectands is not a
#: comparison.
EXPECTANDS = {
    "x_0": lambda s: s["x"][..., 0],
    "x_0_sq": lambda s: s["x"][..., 0] ** 2,
    "x_0x1": lambda s: s["x"][..., 0] * s["x"][..., 1],
}


def _recipe(sampler, warmup, base_params, warmup_params, budget, step_policy=None):
    return Recipe(
        model_name=MODEL,
        base_method_name=sampler,
        warmup_name=warmup,
        effort=Effort.LOW,
        base_method_params=base_params,
        warmup_params=warmup_params,
        headline_metric=None,
        sample_quality=None,
        calibration_budget=budget,
        difficulty=None,
        instructions="",
        step_policy=step_policy,
        tuning_seed=SEED,
    )


def _split_artifact(path):
    """Return ``(draws, chain_stats)`` from a generated ``.npz`` artifact."""
    n = len(SAMPLE_STAT_PREFIX)
    with np.load(str(path), allow_pickle=False) as archive:
        draws = {
            k: np.asarray(archive[k])
            for k in archive.files
            if not k.startswith(SAMPLE_STAT_PREFIX)
        }
        stats = {
            k[n:]: np.asarray(archive[k])
            for k in archive.files
            if k.startswith(SAMPLE_STAT_PREFIX)
        }
    return draws, stats


@pytest.fixture(scope="module")
def sensitivity_run(tmp_path_factory):
    """Pay one warmup, freeze its geometry, and run both arms against it."""
    root = tmp_path_factory.mktemp("trajectory_sensitivity")

    warmup_recipe = _recipe(
        "hmc",
        "window_adaptation_diag_imm",
        {"step_size": 0.5, "num_integration_steps": FIXED_L},
        {"n_warmup": N_WARMUP, "target_acceptance_rate": 0.8},
        {"n_samples": 1, "num_chains": 1},
    )
    warmup_result = execute_recipe(
        warmup_recipe,
        root / "warmup",
        num_samples=1,
        num_chains=1,
        num_warmup=N_WARMUP,
        warmup_num_chains=[1],
        timeout=600,
        env=CPU,
    )
    geometry = warmup_result.telemetry.geometry
    frozen = {
        "step_size": float(np.asarray(geometry["step_size"]).reshape(-1)[0]),
        "inverse_mass_matrix": np.asarray(geometry["inverse_mass_matrix"])
        .reshape(-1)
        .tolist(),
    }
    # The warmup arm also took one sampling transition, which cost gradients.
    # Deriving it keeps the shared cost fully accounted; leaving it out would
    # (correctly) propagate "unknown" into every downstream comparison.
    _, warmup_stats = _split_artifact(warmup_result.artifact_path)
    warmup_cost = CostAccounting.from_telemetry(
        warmup_result.telemetry,
        sampling_transition_grad_evals=sampling_grad_evals_from_chain_stats(
            warmup_stats, "hmc", expected_topology=(1, 1)
        ),
    ).relabel("shared_frozen_warmup")

    budget = {"n_samples": N_DRAWS, "num_chains": N_CHAINS}
    arms = {
        "fixed_L4": _recipe(
            "hmc",
            "no_warmup",
            {**frozen, "num_integration_steps": FIXED_L},
            {"n_warmup": 0},
            budget,
        ),
        "uniform_L2_6": _recipe(
            "dynamic_hmc",
            "no_warmup",
            dict(frozen),
            {"n_warmup": 0},
            budget,
            step_policy=POLICY,
        ),
    }

    out = {"frozen": frozen, "warmup_cost": warmup_cost, "arms": {}}
    for label, recipe in arms.items():
        result = execute_recipe(
            recipe,
            root / label,
            num_samples=N_DRAWS,
            num_chains=N_CHAINS,
            num_warmup=0,
            warmup_num_chains=[N_CHAINS],
            timeout=900,
            env=CPU,
        )
        draws, stats = _split_artifact(result.artifact_path)
        derivation = sampling_grad_evals_from_chain_stats(
            stats,
            recipe.base_method_name,
            expected_topology=(N_CHAINS, N_DRAWS),
        )
        own_cost = CostAccounting.from_telemetry(
            result.telemetry, sampling_transition_grad_evals=derivation
        ).relabel(label)
        # Standalone: this alternative would have to pay the warmup itself.
        standalone = CostAccounting.combine(
            [out["warmup_cost"], own_cost],
            view="standalone",
            phases_are_disjoint_sequential=True,
            source=f"{label}__standalone",
        )
        out["arms"][label] = {
            "draws": draws,
            "stats": stats,
            "derivation": derivation,
            "own_cost": own_cost,
            "report": expectand_report(draws, EXPECTANDS, cost=standalone, label=label),
        }
    return out


def test_both_arms_ran_the_same_frozen_geometry(sensitivity_run):
    """The arms differ in trajectory length and in nothing else."""
    for arm in sensitivity_run["arms"].values():
        assert arm["draws"]["x"].shape == (N_CHAINS, N_DRAWS, 10)

    frozen = sensitivity_run["frozen"]
    assert frozen["step_size"] > 0
    assert len(frozen["inverse_mass_matrix"]) == 10


def test_trajectory_lengths_differ_exactly_as_the_recipes_declare(sensitivity_run):
    fixed = sensitivity_run["arms"]["fixed_L4"]["stats"]["num_integration_steps"]
    varied = sensitivity_run["arms"]["uniform_L2_6"]["stats"]["num_integration_steps"]

    assert np.all(fixed == FIXED_L)
    assert varied.min() >= POLICY["low"]
    assert varied.max() <= POLICY["high"] - 1
    assert varied.min() < varied.max(), "the randomised arm did not vary L"


def test_both_arms_report_the_same_expectands(sensitivity_run):
    labels = [
        tuple(arm["report"].by_label()) for arm in sensitivity_run["arms"].values()
    ]
    assert labels[0] == labels[1] == ("x_0", "x_0_sq", "x_0x1")


def test_every_expectand_is_defined_on_both_arms(sensitivity_run):
    for label, arm in sensitivity_run["arms"].items():
        for entry in arm["report"].entries:
            assert entry.is_defined, f"{label}.{entry.label} was undefined"
            assert entry.degeneracy == "none"


def test_frozen_arms_record_a_known_zero_warmup_not_an_unknown(sensitivity_run):
    """`no_warmup` really executed zero warmup gradients; that is a value."""
    for arm in sensitivity_run["arms"].values():
        own = arm["own_cost"]
        assert own.warmup_grad_evals == 0
        assert own.reason_for("warmup_grad_evals") == ""


def test_transition_gradients_are_recovered_and_stay_a_subtotal(sensitivity_run):
    for label, arm in sensitivity_run["arms"].items():
        derivation = arm["derivation"]
        assert derivation.count is not None, derivation.reason
        expected = int(arm["stats"]["num_integration_steps"].sum())
        assert derivation.count == expected
        assert "rejected transitions included" in derivation.basis
        assert derivation.excluded, f"{label} claimed a complete gradient total"
        print(f"\n{label}: sampling transition gradients = {derivation.count}")


def test_the_shared_warmup_is_charged_once_combined_and_to_each_standalone(
    sensitivity_run,
):
    warmup_cost = sensitivity_run["warmup_cost"]
    arms = sensitivity_run["arms"]
    assert warmup_cost.warmup_grad_evals is not None
    assert warmup_cost.warmup_grad_evals > 0
    assert warmup_cost.sampling_transition_grad_evals is not None

    for arm in arms.values():
        standalone = arm["report"].cost
        assert standalone.view == "standalone"
        assert standalone.warmup_grad_evals == warmup_cost.warmup_grad_evals

    experiment = CostAccounting.combine(
        [warmup_cost, *(arm["own_cost"] for arm in arms.values())],
        view="combined",
        phases_are_disjoint_sequential=True,
        source="experiment",
    )
    assert experiment.warmup_grad_evals == warmup_cost.warmup_grad_evals
    # Every contributor's transitions, the warmup arm's own included -- it took
    # one sampling step, and that step cost gradients too.
    assert experiment.sampling_transition_grad_evals == (
        warmup_cost.sampling_transition_grad_evals
        + sum(arm["own_cost"].sampling_transition_grad_evals for arm in arms.values())
    )


def test_the_comparison_is_cost_normalised_and_states_what_it_omits(
    sensitivity_run,
):
    arms = sensitivity_run["arms"]
    comparison = compare_reports(
        arms["fixed_L4"]["report"], arms["uniform_L2_6"]["report"]
    )

    assert comparison.cost_normalised_available
    assert comparison.cost_views == ("standalone", "standalone")
    assert comparison.excluded_grad_work

    for row in comparison.rows:
        if row.statistic == "rank_rhat":
            # Never ratioed, never divided by a cost.
            assert row.ratio is None
            assert row.per_transition_grad_eval is None
            continue
        assert row.ratio is not None
        assert row.per_second is not None
        assert row.per_transition_grad_eval is not None
        assert all(v is not None for v in row.per_transition_grad_eval)

    # No assertion is made about which arm wins: one model, one frozen
    # geometry, one seed and short chains do not decide that.  The numbers are
    # printed so `pytest -s` reproduces the table in
    # docs/examples/trajectory-length-sensitivity.md.
    print(f"\ncost views: {comparison.cost_views}")
    print(
        f"gradient work excluded from the denominator: "
        f"{len(comparison.excluded_grad_work)} categories"
    )
    for row in comparison.rows:
        if row.statistic == "rank_rhat":
            continue
        print(
            f"  {row.expectand:8s} {row.statistic:13s} "
            f"{comparison.baseline_label}={row.baseline:9.1f} "
            f"{comparison.candidate_label}={row.candidate:9.1f} "
            f"ratio={row.ratio:.3f}"
        )


def test_the_report_renders_for_a_human(sensitivity_run):
    for arm in sensitivity_run["arms"].values():
        text = arm["report"].to_text()
        assert "raw-mean ESS" in text
        assert "bulk ESS" in text
        # Costs are stated, including the ones that stay unknown.
        assert "costs (" in text
        assert "compile_seconds: unknown" in text
        print(f"\n{text}")
