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
"""Deterministic tests for opt-in named-expectand reports.

Every fixture is seeded or literal, so each assertion is reproducible without
running a sampler.  Tests that only exercise trace evaluation, degeneracy
classification, cost accounting or comparison logic are ``fast``; tests that
call a diagnostics backend are ``slow`` because those paths trace JAX.
"""

from __future__ import annotations

import numpy as np
import pytest

from tuningfork.catalog.expectands import (
    BACKENDS,
    CostAccounting,
    ExpectandReport,
    compare_reports,
    expectand_report,
    expectand_traces,
    sampling_grad_evals_from_chain_stats,
)

# --------------------------------------------------------------------------
# Fixtures — all deterministic
# --------------------------------------------------------------------------

N_CHAINS = 4
N_DRAWS = 200


@pytest.fixture
def draws() -> dict[str, np.ndarray]:
    """Well-behaved IID draws: (4, 200, 2)."""
    rng = np.random.default_rng(20260907)
    return {"theta": rng.standard_normal((N_CHAINS, N_DRAWS, 2))}


def _moment_expectands() -> dict:
    """First moment, even (squared) and cross quantities of the same draws."""
    return {
        "theta_0": lambda s: s["theta"][..., 0],
        "theta_0_sq": lambda s: s["theta"][..., 0] ** 2,
        "theta_0x1": lambda s: s["theta"][..., 0] * s["theta"][..., 1],
    }


# --------------------------------------------------------------------------
# Named traces
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_traces_are_named_and_keep_draw_topology(draws):
    traces = expectand_traces(draws, _moment_expectands())

    assert set(traces) == {"theta_0", "theta_0_sq", "theta_0x1"}
    for trace in traces.values():
        assert trace.shape == (N_CHAINS, N_DRAWS)
    np.testing.assert_allclose(traces["theta_0_sq"], draws["theta"][..., 0] ** 2)
    np.testing.assert_allclose(
        traces["theta_0x1"], draws["theta"][..., 0] * draws["theta"][..., 1]
    )


@pytest.mark.fast
def test_original_draws_are_preserved(draws):
    before = {name: arr.copy() for name, arr in draws.items()}

    def mutating(samples):
        # A careless expectand must not be able to corrupt the cached draws.
        with pytest.raises(ValueError):
            samples["theta"][0, 0, 0] = 999.0
        return samples["theta"][..., 0]

    expectand_traces(draws, {"probe": mutating})

    for name, arr in draws.items():
        np.testing.assert_array_equal(arr, before[name])


@pytest.mark.fast
def test_vector_valued_expectand_yields_one_row_per_component(draws):
    traces = expectand_traces(draws, {"theta": lambda s: s["theta"]})
    assert traces["theta"].shape == (N_CHAINS, N_DRAWS, 2)


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad, match",
    [
        ({}, "at least one named function"),
        ({"wrong_shape": lambda s: s["theta"][0, ..., 0]}, "draw topology"),
    ],
)
def test_trace_validation(draws, bad, match):
    with pytest.raises(ValueError, match=match):
        expectand_traces(draws, bad)


@pytest.mark.fast
def test_mismatched_draw_topology_is_rejected(draws):
    mixed = dict(draws)
    mixed["other"] = np.zeros((N_CHAINS, N_DRAWS + 1))
    with pytest.raises(ValueError, match="same \\(n_chains, n_draws\\) topology"):
        expectand_traces(mixed, {"theta_0": lambda s: s["theta"][..., 0]})


@pytest.mark.fast
def test_single_chain_draws_are_rejected():
    with pytest.raises(ValueError, match="multi-chain layout"):
        expectand_traces({"theta": np.zeros(10)}, {"x": lambda s: s["theta"]})


# --------------------------------------------------------------------------
# Cost accounting — unknown is never zero
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_default_cost_is_unknown_not_zero():
    report = expectand_report({"theta": np.zeros((2, 4))}, {"c": lambda s: s["theta"]})
    cost = report.cost
    assert cost.is_fully_accounted is False
    assert set(cost.unknown_components) == set(CostAccounting.COMPONENTS)
    for component in CostAccounting.COMPONENTS:
        assert getattr(cost, component) is None
        assert cost.reason_for(component)


@pytest.mark.fast
def test_cost_from_telemetry_reads_only_existing_schema_fields():
    class FakeTelemetry:
        timing_seconds = {"warmup": 2.0, "sampling": 8.0, "total": 11.5}
        warmup_grad_evals = 640
        warmup_grad_evals_reason = "n_warmup * (num_integration_steps + 1)"

    cost = CostAccounting.from_telemetry(FakeTelemetry())

    assert cost.warmup_seconds == 2.0
    assert cost.sampling_seconds == 8.0
    assert cost.total_seconds == 11.5
    assert cost.warmup_grad_evals == 640
    # Not in the telemetry schema — must stay unknown, with a reason.
    assert cost.sampling_transition_grad_evals is None
    assert "warmup gradient evaluations only" in cost.reason_for(
        "sampling_transition_grad_evals"
    )
    assert cost.compile_seconds is None
    assert "not subtracted" in cost.reason_for("compile_seconds")
    assert "wall clocks include JIT compilation" in cost.notes
    assert cost.is_fully_accounted is False


@pytest.mark.fast
def test_cost_from_telemetry_preserves_a_null_gradient_count():
    class FakeTelemetry:
        timing_seconds = {"warmup": 1.0, "sampling": 1.0, "total": 2.0}
        warmup_grad_evals = None
        warmup_grad_evals_reason = "warmup kernel has no exact gradient bound"

    cost = CostAccounting.from_telemetry(FakeTelemetry())

    assert cost.warmup_grad_evals is None
    assert cost.reason_for("warmup_grad_evals") == (
        "warmup kernel has no exact gradient bound"
    )


@pytest.mark.fast
def test_cost_from_recipe_uses_calibration_budget_walls():
    class FakeRecipe:
        calibration_budget = {
            "warmup_wall_seconds": 3.0,
            "sampling_wall_seconds": 12.0,
            "n_samples": 1000,
        }

    cost = CostAccounting.from_recipe(FakeRecipe())

    assert cost.warmup_seconds == 3.0
    assert cost.sampling_seconds == 12.0
    assert cost.total_seconds == 15.0
    assert cost.warmup_grad_evals is None
    assert cost.sampling_transition_grad_evals is None


@pytest.mark.fast
def test_cost_from_recipe_without_walls_stays_unknown():
    class FakeRecipe:
        calibration_budget = {"n_samples": 1000}

    cost = CostAccounting.from_recipe(FakeRecipe())

    assert cost.warmup_seconds is None
    assert cost.total_seconds is None
    assert "no warmup_wall_seconds" in cost.reason_for("warmup_seconds")


# --------------------------------------------------------------------------
# Degenerate expectands — undefined, not a spurious number
# --------------------------------------------------------------------------


def _degenerate_report(trace: np.ndarray, backend: str = "blackjax") -> ExpectandReport:
    return expectand_report({"raw": trace}, {"q": lambda s: s["raw"]}, backend=backend)


@pytest.mark.fast
@pytest.mark.parametrize("backend", BACKENDS)
def test_global_constant_expectand_is_undefined(backend):
    entry = _degenerate_report(np.full((N_CHAINS, N_DRAWS), 2.5), backend).entries[0]

    assert entry.degeneracy == "global_constant"
    assert entry.raw_mean_ess is None
    assert entry.bulk_ess is None
    assert entry.tail_ess is None
    assert entry.rank_rhat is None
    assert entry.is_defined is False
    assert "zero-variance" in entry.undefined_reasons["bulk_ess"]
    assert entry.tie_fraction == pytest.approx(1.0 - 1 / (N_CHAINS * N_DRAWS))


@pytest.mark.fast
@pytest.mark.parametrize("backend", BACKENDS)
def test_per_lane_constant_expectand_is_undefined(backend):
    trace = np.repeat(np.array([1.0, 2.0, 3.0, 4.0])[:, None], N_DRAWS, axis=1)
    entry = _degenerate_report(trace, backend).entries[0]

    assert entry.degeneracy == "per_chain_constant"
    assert entry.is_defined is False
    for statistic in ("raw_mean_ess", "bulk_ess", "tail_ess", "rank_rhat"):
        assert entry.value(statistic) is None
        assert "within-chain variance is zero" in entry.undefined_reasons[statistic]


@pytest.mark.fast
def test_non_finite_expectand_is_undefined():
    trace = np.ones((N_CHAINS, N_DRAWS))
    trace[1, 7] = np.nan
    entry = _degenerate_report(trace).entries[0]

    assert entry.degeneracy == "non_finite"
    assert entry.is_defined is False
    assert "non-finite" in entry.undefined_reasons["raw_mean_ess"]


@pytest.mark.slow
def test_partially_constant_chains_warn_but_still_report(draws):
    trace = np.asarray(draws["theta"][..., 0]).copy()
    trace[0, :] = trace[0, 0]
    entry = _degenerate_report(trace).entries[0]

    assert entry.degeneracy == "none"
    assert entry.is_defined is True
    assert any("chains are constant" in w for w in entry.warnings)


# --------------------------------------------------------------------------
# Repeated values — flagged, and backend-sensitive
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_repeated_values_are_flagged_with_a_backend_caveat():
    rng = np.random.default_rng(913)
    trace = (rng.random((N_CHAINS, N_DRAWS)) < 0.1).astype(float)
    entry = _degenerate_report(trace).entries[0]

    assert entry.degeneracy == "none"
    assert entry.n_distinct == 2
    assert entry.tie_fraction > 0.99
    assert any("repeated values" in w for w in entry.warnings)
    assert any("backend='arviz'" in w for w in entry.warnings)


@pytest.mark.slow
def test_backends_disagree_on_a_tied_trace_and_the_report_says_which():
    """A tied indicator is exactly where the backend choice is load-bearing.

    Both numbers are reported under their own backend name; neither is
    presented as the value.
    """
    rng = np.random.default_rng(913)
    trace = (rng.random((N_CHAINS, N_DRAWS)) < 0.1).astype(float)

    bj = _degenerate_report(trace, "blackjax").entries[0]
    az = _degenerate_report(trace, "arviz").entries[0]

    assert bj.backend == "blackjax"
    assert az.backend == "arviz"
    # Rank-normalised statistics diverge by orders of magnitude on ties.
    assert az.bulk_ess is not None and bj.bulk_ess is not None
    assert az.bulk_ess > 10 * bj.bulk_ess
    # The raw-mean ESS, which applies no rank normalisation, stays comparable.
    assert bj.raw_mean_ess is not None and az.raw_mean_ess is not None
    assert 0.5 < bj.raw_mean_ess / az.raw_mean_ess < 2.0


@pytest.mark.slow
def test_backends_agree_on_a_well_behaved_continuous_trace(draws):
    bj = expectand_report(draws, _moment_expectands(), backend="blackjax").by_label()
    az = expectand_report(draws, _moment_expectands(), backend="arviz").by_label()

    for label in bj:
        for statistic in ("bulk_ess", "tail_ess", "rank_rhat"):
            lhs, rhs = bj[label].value(statistic), az[label].value(statistic)
            assert lhs is not None and rhs is not None
            assert lhs == pytest.approx(rhs, rel=1e-4)


# --------------------------------------------------------------------------
# Report composition
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_raw_mean_ess_is_reported_beside_rank_diagnostics(draws):
    report = expectand_report(
        draws, _moment_expectands(), label="baseline", backend="blackjax"
    )

    assert [e.label for e in report.entries] == [
        "theta_0",
        "theta_0_sq",
        "theta_0x1",
    ]
    for entry in report.entries:
        # Four distinct statistics, none substituted for another.
        assert entry.raw_mean_ess is not None
        assert entry.bulk_ess is not None
        assert entry.tail_ess is not None
        assert entry.rank_rhat is not None
        assert entry.backend == "blackjax"

    rows = report.to_rows()
    assert {"raw_mean_ess", "bulk_ess", "tail_ess", "rank_rhat"} <= set(rows[0])


@pytest.mark.slow
def test_antithetic_first_moment_does_not_speak_for_the_squared_function():
    """A perfectly antithetic first moment sits next to a badly mixing square.

    ``x_t`` alternates sign, so its raw-mean ESS is inflated far above the draw
    count while ``x_t**2`` is a slowly drifting sequence.  Quoting only the
    first-moment number is what this report exists to prevent.
    """
    rng = np.random.default_rng(4)
    magnitude = (
        np.abs(np.cumsum(rng.standard_normal((N_CHAINS, N_DRAWS)), axis=1)) + 1.0
    )
    sign = np.where(np.arange(N_DRAWS) % 2 == 0, 1.0, -1.0)
    trace = magnitude * sign

    report = expectand_report(
        {"x": trace},
        {"x": lambda s: s["x"], "x_sq": lambda s: s["x"] ** 2},
    )
    entries = report.by_label()

    assert entries["x"].raw_mean_ess > N_CHAINS * N_DRAWS
    assert entries["x_sq"].raw_mean_ess < entries["x"].raw_mean_ess / 10


@pytest.mark.slow
def test_to_text_renders_undefined_and_unknown_as_words(draws):
    class FakeRecipe:
        calibration_budget = {"warmup_wall_seconds": 3.0}

    report = expectand_report(
        {"const": np.full((N_CHAINS, N_DRAWS), 1.0), **draws},
        {"c": lambda s: s["const"], "theta_0": lambda s: s["theta"][..., 0]},
        cost=CostAccounting.from_recipe(FakeRecipe()),
        label="run",
    )
    text = report.to_text()
    cost_block = text.split("costs")[1]

    assert "undefined" in text
    assert "unknown" in text
    # The unmeasured sampling wall renders as a reason, never as a zero.
    assert "sampling_seconds: unknown" in cost_block
    assert "total_seconds: unknown" in cost_block
    assert "warmup_seconds: 3.0" in cost_block
    assert "wall clocks include JIT compilation" in text


@pytest.mark.fast
def test_unknown_backend_is_rejected(draws):
    with pytest.raises(ValueError, match="backend must be one of"):
        expectand_report(draws, _moment_expectands(), backend="stan")


# --------------------------------------------------------------------------
# Comparison — costs carried, never guessed
# --------------------------------------------------------------------------


def _fake_report(label: str, scale: float, cost: CostAccounting) -> ExpectandReport:
    rng = np.random.default_rng(11)
    trace = rng.standard_normal((N_CHAINS, N_DRAWS)) * scale
    return expectand_report(
        {"x": trace}, {"x": lambda s: s["x"]}, cost=cost, label=label
    )


_FULL_COST = CostAccounting(
    warmup_seconds=1.0,
    sampling_seconds=4.0,
    total_seconds=5.0,
    warmup_grad_evals=100,
    sampling_transition_grad_evals=400,
    compile_seconds=0.5,
    source="test",
)


@pytest.mark.slow
def test_comparison_normalises_by_cost_only_when_both_sides_measured():
    baseline = _fake_report("A", 1.0, _FULL_COST)
    candidate = _fake_report("B", 2.0, _FULL_COST)

    comparison = compare_reports(baseline, candidate)

    assert comparison.cost_normalised_available is True
    row = next(r for r in comparison.rows if r.statistic == "bulk_ess")
    assert row.ratio == pytest.approx(1.0, rel=1e-6)  # same seed, rescaled
    assert row.per_second is not None
    assert row.per_second[0] == pytest.approx(row.baseline / 5.0)
    assert row.per_transition_grad_eval is not None
    assert row.per_transition_grad_eval[1] == pytest.approx(row.candidate / 500.0)


@pytest.mark.slow
def test_comparison_refuses_cost_normalisation_when_a_cost_is_unknown():
    baseline = _fake_report("A", 1.0, _FULL_COST)
    candidate = _fake_report("B", 2.0, CostAccounting(source="test"))

    comparison = compare_reports(baseline, candidate)

    assert comparison.cost_normalised_available is False
    assert any("B.total_seconds" in b for b in comparison.cost_blockers)
    row = next(r for r in comparison.rows if r.statistic == "bulk_ess")
    assert row.per_second is None
    assert row.per_transition_grad_eval is None
    assert row.cost_blocked_by
    # The uncosted ESS ratio is still available; only the cost-normalised
    # figures are withheld.
    assert row.ratio is not None


@pytest.mark.slow
def test_comparison_blocks_a_ratio_against_an_undefined_statistic():
    cost = _FULL_COST
    baseline = expectand_report(
        {"x": np.full((N_CHAINS, N_DRAWS), 3.0)},
        {"x": lambda s: s["x"]},
        cost=cost,
        label="A",
    )
    candidate = _fake_report("B", 1.0, cost)

    comparison = compare_reports(baseline, candidate)
    row = next(r for r in comparison.rows if r.statistic == "raw_mean_ess")

    assert row.baseline is None
    assert row.ratio is None
    assert any("A.x.raw_mean_ess undefined" in b for b in row.ratio_blocked_by)


@pytest.mark.slow
def test_comparison_reports_backend_mismatch_and_unmatched_expectands(draws):
    baseline = expectand_report(
        draws, {"theta_0": lambda s: s["theta"][..., 0]}, backend="blackjax", label="A"
    )
    candidate = expectand_report(
        draws,
        {
            "theta_0": lambda s: s["theta"][..., 0],
            "theta_1": lambda s: s["theta"][..., 1],
        },
        backend="arviz",
        label="B",
    )

    comparison = compare_reports(baseline, candidate)

    assert comparison.backend_mismatch == ("blackjax", "arviz")
    assert comparison.only_in_candidate == ("theta_1",)
    assert comparison.only_in_baseline == ()


@pytest.mark.slow
def test_rank_rhat_is_carried_but_never_ratioed_or_cost_normalised():
    """R-hat is a convergence ratio judged against its own threshold.

    "R-hat per second" and "candidate R-hat over baseline R-hat" are both
    meaningless, so the comparison reports both values and withholds the
    derived figures.
    """
    baseline = _fake_report("A", 1.0, _FULL_COST)
    candidate = _fake_report("B", 2.0, _FULL_COST)

    comparison = compare_reports(baseline, candidate)
    row = next(r for r in comparison.rows if r.statistic == "rank_rhat")

    assert row.baseline is not None and row.candidate is not None
    assert row.ratio is None
    assert any("not a rate" in b for b in row.ratio_blocked_by)
    assert row.per_second is None
    assert row.per_transition_grad_eval is None
    assert any("cost normalisation does not apply" in b for b in row.cost_blocked_by)

    # The ESS statistics on the same run are still normalised.
    ess_row = next(r for r in comparison.rows if r.statistic == "bulk_ess")
    assert ess_row.per_second is not None


@pytest.mark.slow
def test_sparse_ties_do_not_raise_the_backend_caveat():
    """A handful of repeated states is normal after MCMC rejections.

    The caveat is a display threshold; ``tie_fraction`` is reported either way.
    """
    rng = np.random.default_rng(77)
    trace = rng.standard_normal((N_CHAINS, N_DRAWS))
    trace[0, 5] = trace[0, 4]  # one repeated state out of 800

    entry = _degenerate_report(trace).entries[0]

    assert entry.tie_fraction == pytest.approx(1 / (N_CHAINS * N_DRAWS))
    assert entry.tie_severity == "minor"


@pytest.mark.slow
def test_tie_caveat_threshold_is_explicit():
    rng = np.random.default_rng(77)
    trace = rng.standard_normal((N_CHAINS, N_DRAWS))
    trace[0, 5] = trace[0, 4]

    loud = expectand_report(
        {"raw": trace}, {"q": lambda s: s["raw"]}, tie_caveat_threshold=0.0
    ).entries[0]

    assert any("repeated values" in w for w in loud.warnings)
    assert loud.tie_fraction == pytest.approx(1 / (N_CHAINS * N_DRAWS))


# --------------------------------------------------------------------------
# Shared already-paid warmup: standalone vs combined accounting
# --------------------------------------------------------------------------


def _shared_warmup_cost() -> CostAccounting:
    return CostAccounting(
        warmup_seconds=1.0,
        sampling_seconds=0.0,
        total_seconds=1.5,
        warmup_grad_evals=1200,
        sampling_transition_grad_evals=0,
        compile_seconds=None,
        unknown_reasons={"compile_seconds": "not separately measured"},
        source="shared_warmup",
    )


def _frozen_arm_cost(label: str, sampling: float, grads: int) -> CostAccounting:
    return CostAccounting(
        warmup_seconds=0.0,
        sampling_seconds=sampling,
        total_seconds=sampling + 1.0,
        warmup_grad_evals=0,
        sampling_transition_grad_evals=grads,
        compile_seconds=None,
        unknown_reasons={"compile_seconds": "not separately measured"},
        source=label,
    )


@pytest.mark.fast
def test_combine_sums_disjoint_sequential_phases():
    combined = CostAccounting.combine(
        [_shared_warmup_cost(), _frozen_arm_cost("arm_a", 0.6, 8000)],
        view="standalone",
        phases_are_disjoint_sequential=True,
    )

    assert combined.view == "standalone"
    assert combined.warmup_seconds == 1.0
    assert combined.total_seconds == pytest.approx(3.1)
    assert combined.warmup_grad_evals == 1200
    assert combined.sampling_transition_grad_evals == 8000
    assert combined.provenance() == ("shared_warmup", "arm_a")
    # compile_seconds was unknown on both sides and stays unknown.
    assert combined.compile_seconds is None
    assert "not separately measured" in combined.reason_for("compile_seconds")


@pytest.mark.fast
def test_known_zero_is_not_unknown():
    """A no_warmup arm really executed zero warmup gradients."""
    arm = _frozen_arm_cost("arm_a", 0.6, 8000)

    assert arm.warmup_grad_evals == 0
    assert arm.reason_for("warmup_grad_evals") == ""
    assert "warmup_grad_evals" not in arm.unknown_components

    combined = CostAccounting.combine(
        [arm, _frozen_arm_cost("arm_b", 0.9, 8049)],
        view="combined",
        phases_are_disjoint_sequential=True,
    )
    assert combined.warmup_grad_evals == 0


@pytest.mark.fast
def test_combine_refuses_to_sum_overlapping_wall_clocks():
    combined = CostAccounting.combine(
        [_shared_warmup_cost(), _frozen_arm_cost("arm_a", 0.6, 8000)],
        view="combined",
        phases_are_disjoint_sequential=False,
    )

    # Counts are additive under overlap; wall clocks are not.
    assert combined.warmup_grad_evals == 1200
    assert combined.sampling_transition_grad_evals == 8000
    for component in ("warmup_seconds", "sampling_seconds", "total_seconds"):
        assert getattr(combined, component) is None
        assert "disjoint" in combined.reason_for(component)
    assert any("were not summed" in n for n in combined.notes)


@pytest.mark.fast
def test_combine_propagates_an_unknown_contributor_with_its_reason():
    partial = CostAccounting(
        warmup_seconds=2.0,
        unknown_reasons={"warmup_grad_evals": "warmup kernel has no exact bound"},
        source="partial_arm",
    )
    combined = CostAccounting.combine(
        [_shared_warmup_cost(), partial],
        view="combined",
        phases_are_disjoint_sequential=True,
    )

    assert combined.warmup_grad_evals is None
    reason = combined.reason_for("warmup_grad_evals")
    assert "partial_arm" in reason
    assert "no exact bound" in reason
    assert combined.is_fully_accounted is False


@pytest.mark.fast
def test_combine_refuses_to_charge_a_shared_warmup_twice():
    """The guard against double-charging an already-included contributor."""
    warmup = _shared_warmup_cost()
    standalone_a = CostAccounting.combine(
        [warmup, _frozen_arm_cost("arm_a", 0.6, 8000)],
        view="standalone",
        phases_are_disjoint_sequential=True,
    )

    with pytest.raises(ValueError, match="shared_warmup"):
        CostAccounting.combine(
            [standalone_a, warmup],
            view="combined",
            phases_are_disjoint_sequential=True,
        )


@pytest.mark.fast
def test_combined_view_charges_a_shared_warmup_once_standalone_charges_it_twice():
    """The two views of the same experiment, and why they differ."""
    warmup = _shared_warmup_cost()
    arm_a = _frozen_arm_cost("arm_a", 0.6, 8000)
    arm_b = _frozen_arm_cost("arm_b", 0.9, 8049)

    standalone_a = CostAccounting.combine(
        [warmup, arm_a], view="standalone", phases_are_disjoint_sequential=True
    )
    standalone_b = CostAccounting.combine(
        [warmup, arm_b], view="standalone", phases_are_disjoint_sequential=True
    )
    experiment = CostAccounting.combine(
        [warmup, arm_a, arm_b], view="combined", phases_are_disjoint_sequential=True
    )

    # Each arm, asked what it would cost alone, pays the full warmup.
    assert standalone_a.warmup_grad_evals == 1200
    assert standalone_b.warmup_grad_evals == 1200
    # The experiment paid for it once.
    assert experiment.warmup_grad_evals == 1200
    assert (
        experiment.sampling_transition_grad_evals
        == arm_a.sampling_transition_grad_evals + arm_b.sampling_transition_grad_evals
    )
    assert experiment.view == "combined"
    assert set(experiment.provenance()) == {"shared_warmup", "arm_a", "arm_b"}


@pytest.mark.fast
@pytest.mark.parametrize("view", ["as_measured", "nonsense"])
def test_combine_rejects_a_view_it_cannot_produce(view):
    with pytest.raises(ValueError, match="standalone"):
        CostAccounting.combine(
            [_shared_warmup_cost(), _frozen_arm_cost("a", 1.0, 1)],
            view=view,
            phases_are_disjoint_sequential=True,
        )


# --------------------------------------------------------------------------
# Recovering the transition-gradient subtotal from persisted statistics
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_transition_grads_sum_recorded_steps_including_rejections():
    stats = {"num_integration_steps": np.full((N_CHAINS, 10), 4)}

    derivation = sampling_grad_evals_from_chain_stats(
        stats, "hmc", expected_topology=(N_CHAINS, 10)
    )

    assert derivation.count == N_CHAINS * 10 * 4
    assert "num_integration_steps" in derivation.basis
    assert "rejected transitions included" in derivation.basis
    assert derivation.source_fields == ("num_integration_steps",)
    # A subtotal, never a claimed total.
    assert derivation.excluded
    assert any("initialization" in item for item in derivation.excluded)


@pytest.mark.fast
def test_transition_grads_handle_a_variable_length_sampler():
    rng = np.random.default_rng(5)
    steps = rng.integers(2, 7, size=(N_CHAINS, 10))
    stats = {"num_integration_steps": steps}

    derivation = sampling_grad_evals_from_chain_stats(stats, "dynamic_hmc")

    assert derivation.count == int(steps.sum())


@pytest.mark.fast
def test_transition_grads_refuse_a_partial_record():
    stats = {"num_integration_steps": np.full((N_CHAINS - 1, 10), 4)}

    derivation = sampling_grad_evals_from_chain_stats(
        stats, "hmc", expected_topology=(N_CHAINS, 10)
    )

    assert derivation.count is None
    assert "does not cover every lane" in derivation.reason


@pytest.mark.fast
def test_transition_grads_refuse_ragged_statistics():
    stats = {
        "num_integration_steps": np.full((N_CHAINS, 10), 4),
        "energy": np.zeros((N_CHAINS, 9)),
    }

    derivation = sampling_grad_evals_from_chain_stats(stats, "hmc")

    assert derivation.count is None
    assert "ragged" in derivation.reason


@pytest.mark.fast
def test_transition_grads_refuse_when_the_needed_field_was_not_persisted():
    derivation = sampling_grad_evals_from_chain_stats(
        {"energy": np.zeros((2, 3))}, "hmc"
    )

    assert derivation.count is None
    assert "not persisted" in derivation.reason


@pytest.mark.fast
def test_transition_grads_refuse_empty_or_unknown_inputs():
    assert sampling_grad_evals_from_chain_stats({}, "hmc").count is None
    unknown = sampling_grad_evals_from_chain_stats(
        {"num_integration_steps": np.ones((2, 3))}, "not_a_sampler"
    )
    assert unknown.count is None
    assert "unknown base method" in unknown.reason


@pytest.mark.fast
def test_zero_recorded_transitions_is_a_known_zero():
    derivation = sampling_grad_evals_from_chain_stats(
        {"num_integration_steps": np.zeros((2, 0))}, "hmc"
    )

    assert derivation.count == 0
    assert derivation.reason == ""
    assert derivation.excluded


@pytest.mark.fast
def test_telemetry_accepts_a_derivation_and_carries_its_basis_and_exclusions():
    class FakeTelemetry:
        timing_seconds = {"warmup": 1.0, "sampling": 2.0, "total": 4.0}
        warmup_grad_evals = 640
        warmup_grad_evals_reason = "n_warmup * (num_integration_steps + 1)"

    derivation = sampling_grad_evals_from_chain_stats(
        {"num_integration_steps": np.full((2, 5), 3)}, "hmc"
    )
    cost = CostAccounting.from_telemetry(
        FakeTelemetry(), sampling_transition_grad_evals=derivation
    )

    assert cost.sampling_transition_grad_evals == 30
    assert any("basis" in note for note in cost.notes)
    assert cost.excluded_grad_work == derivation.excluded


@pytest.mark.fast
def test_telemetry_keeps_a_refused_derivation_unknown_with_its_reason():
    class FakeTelemetry:
        timing_seconds = {"warmup": 1.0, "sampling": 2.0, "total": 4.0}
        warmup_grad_evals = 640
        warmup_grad_evals_reason = "n_warmup * (num_integration_steps + 1)"

    derivation = sampling_grad_evals_from_chain_stats({}, "hmc")
    cost = CostAccounting.from_telemetry(
        FakeTelemetry(), sampling_transition_grad_evals=derivation
    )

    assert cost.sampling_transition_grad_evals is None
    assert "no per-step chain statistics" in cost.reason_for(
        "sampling_transition_grad_evals"
    )
    assert cost.excluded_grad_work == ()


# --------------------------------------------------------------------------
# Ties are always disclosed; severity only changes the wording
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_single_tie_is_still_disclosed_below_the_threshold():
    """Severity is display-only; it never suppresses the backend disclosure."""
    rng = np.random.default_rng(77)
    trace = rng.standard_normal((N_CHAINS, N_DRAWS))
    trace[0, 5] = trace[0, 4]

    entry = _degenerate_report(trace, "blackjax").entries[0]

    assert entry.tie_severity == "minor"
    assert entry.tie_fraction == pytest.approx(1 / (N_CHAINS * N_DRAWS))
    disclosure = [w for w in entry.warnings if "repeated values" in w]
    assert disclosure, "a tie below the threshold must still be disclosed"
    assert "ordinal ranks" in disclosure[0]
    assert "backend='arviz'" in disclosure[0]


@pytest.mark.slow
def test_material_ties_raise_the_severity_not_the_existence_of_disclosure():
    rng = np.random.default_rng(913)
    trace = (rng.random((N_CHAINS, N_DRAWS)) < 0.1).astype(float)

    entry = _degenerate_report(trace, "blackjax").entries[0]

    assert entry.tie_severity == "material"
    assert any("largely repeated values" in w for w in entry.warnings)


@pytest.mark.slow
def test_a_tie_free_trace_discloses_nothing():
    rng = np.random.default_rng(3)
    trace = rng.standard_normal((N_CHAINS, N_DRAWS))

    entry = _degenerate_report(trace).entries[0]

    assert entry.tie_severity == "none"
    assert entry.tie_fraction == 0.0
    assert not any("repeated values" in w for w in entry.warnings)


@pytest.mark.slow
def test_comparison_carries_cost_views_and_gradient_exclusions():
    cost = CostAccounting(
        warmup_seconds=1.0,
        sampling_seconds=4.0,
        total_seconds=5.0,
        warmup_grad_evals=100,
        sampling_transition_grad_evals=400,
        compile_seconds=0.5,
        source="test",
        view="standalone",
        excluded_grad_work=("initialization",),
    )
    comparison = compare_reports(
        _fake_report("A", 1.0, cost), _fake_report("B", 2.0, cost)
    )

    assert comparison.cost_views == ("standalone", "standalone")
    assert comparison.excluded_grad_work == ("initialization",)
    row = next(r for r in comparison.rows if r.statistic == "bulk_ess")
    assert row.per_transition_grad_eval is not None
    assert row.per_transition_grad_eval[0] == pytest.approx(row.baseline / 500.0)
