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
    assert cost.sampling_grad_evals is None
    assert "warmup gradient evaluations only" in cost.reason_for("sampling_grad_evals")
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
    assert cost.sampling_grad_evals is None


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
    sampling_grad_evals=400,
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
    assert row.per_grad_eval is not None
    assert row.per_grad_eval[1] == pytest.approx(row.candidate / 500.0)


@pytest.mark.slow
def test_comparison_refuses_cost_normalisation_when_a_cost_is_unknown():
    baseline = _fake_report("A", 1.0, _FULL_COST)
    candidate = _fake_report("B", 2.0, CostAccounting(source="test"))

    comparison = compare_reports(baseline, candidate)

    assert comparison.cost_normalised_available is False
    assert any("B.total_seconds" in b for b in comparison.cost_blockers)
    row = next(r for r in comparison.rows if r.statistic == "bulk_ess")
    assert row.per_second is None
    assert row.per_grad_eval is None
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
    assert row.per_grad_eval is None
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
    assert not any("repeated values" in w for w in entry.warnings)


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
