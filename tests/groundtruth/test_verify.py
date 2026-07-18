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
"""Tests for verify_groundtruth.

All tests are fast (no JAX traces, no chain runs) — they exercise the gate
and coherence logic using committed GT artifacts as synthetic "generated"
summaries or by injecting crafted dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary
from tuningfork.groundtruth._emit import compute_summary_stats
from tuningfork.groundtruth._verify import (
    _TAU_SCI,
    _check_coherence,
    _check_gate,
    verify_groundtruth,
)

pytestmark = pytest.mark.fast

# --------------------------------------------------------------------------- #
# compute_summary_stats unit tests (M1 coverage)
# --------------------------------------------------------------------------- #


def test_compute_summary_stats_between_chain_se_correct_axis() -> None:
    """between_chain_se correctly uses chain axis=0, not draw axis=1 (M1 guard).

    This test catches an nc/ns axis-swap bug in compute_summary_stats.

    Setup: 2 chains, 20 draws each.  Chain 0 draws are all +10; chain 1 draws
    are all -10.  With the correct formula (mean over draws per chain, then std
    over chains), chain_means = [+10, -10] and:
        between_chain_se = std([+10, -10], ddof=1) / sqrt(2) = sqrt(2) / sqrt(2) = 10.

    With the M1 bug (axes swapped so nc=20, ns=2), "chain_means" would be 20
    means of pairs of draws.  Pairs are interleaved [+10, -10] so "chain means"
    alternate ~0, and std(20 × 0) / sqrt(20) ≈ 0 — clearly wrong.

    Asserting be_se > 5 (vs near-0 for the bug) is a tight discriminator.
    """
    rng = np.random.default_rng(0)
    ns = 20
    # Chain 0: all +10; chain 1: all -10 (deterministic, no noise)
    chain0 = np.full((ns, 1), 10.0) + rng.normal(0, 1e-6, (ns, 1))
    chain1 = np.full((ns, 1), -10.0) + rng.normal(0, 1e-6, (ns, 1))
    positions = {"x": np.stack([chain0, chain1], axis=0)}  # (2, 20, 1)

    per_site, _, _ = compute_summary_stats(positions)
    be_se = float(np.asarray(per_site["x"]["between_chain_se"]).ravel()[0])

    # Correct formula: std(+10, -10, ddof=1) / sqrt(2) = (10 * sqrt(2)) / sqrt(2) = 10
    # M1 bug: std of ~20 pairs averaging ≈ 0 / sqrt(20) ≈ 0
    assert be_se > 5.0, (
        f"between_chain_se={be_se:.4f} — expected ≈10.0 for 2 chains at ±10. "
        "Likely cause: nc/ns axis swap in compute_summary_stats "
        "(chain axis treated as draw axis)."
    )


# --------------------------------------------------------------------------- #
# _check_gate unit tests
# --------------------------------------------------------------------------- #


def _gate_summary(
    *,
    max_rhat: float = 1.005,
    min_bulk_ess: float = 500.0,
    total_divergences: int = 0,
    min_e_bfmi: float | None = 0.5,
    n_total: int = 100_000,
    generator: str = "nuts_perchain",
) -> dict[str, Any]:
    """Build a minimal summary_v2 dict for gate testing."""
    return {
        "generator": generator,
        "n_total": n_total,
        "quality_gate": {
            "max_rhat": max_rhat,
            "min_bulk_ess": min_bulk_ess,
            "total_divergences": total_divergences,
            "min_e_bfmi": min_e_bfmi,
        },
    }


def test_gate_pass_nuts() -> None:
    """Clean NUTS summary passes gate."""
    passed, details = _check_gate(_gate_summary())
    assert passed
    assert details["rhat_ok"]
    assert details["ess_ok"]
    assert details["div_ok"]
    assert details["ebfmi_ok"]


def test_gate_fail_rhat() -> None:
    """max_rhat > 1.01 fails gate."""
    passed, details = _check_gate(_gate_summary(max_rhat=1.015))
    assert not passed
    assert not details["rhat_ok"]


def test_gate_fail_ess() -> None:
    """min_bulk_ess < 400 fails gate for NUTS."""
    passed, details = _check_gate(_gate_summary(min_bulk_ess=350.0))
    assert not passed
    assert not details["ess_ok"]


def test_gate_fail_divergences() -> None:
    """High divergence rate fails gate."""
    passed, details = _check_gate(_gate_summary(total_divergences=200, n_total=100_000))
    assert not passed
    assert not details["div_ok"]


def test_gate_fail_ebfmi() -> None:
    """min_e_bfmi < 0.3 fails gate for NUTS."""
    passed, details = _check_gate(_gate_summary(min_e_bfmi=0.25))
    assert not passed
    assert not details["ebfmi_ok"]


def test_gate_analytic_skips_nuts_checks() -> None:
    """Analytic models skip ESS / div / E-BFMI checks."""
    summary = _gate_summary(
        generator="analytic_iid",
        min_bulk_ess=10.0,
        total_divergences=100,
        min_e_bfmi=0.1,
        max_rhat=1.005,
    )
    passed, details = _check_gate(summary)
    assert passed, "analytic model with bad ESS/div/ebfmi should still pass"
    assert details["ess_ok"]
    assert details["div_ok"]
    assert details["ebfmi_ok"]


def test_gate_no_ebfmi_field() -> None:
    """Missing min_e_bfmi field is treated as no constraint."""
    passed, _ = _check_gate(_gate_summary(min_e_bfmi=None))
    assert passed


# --------------------------------------------------------------------------- #
# Helpers for coherence tests
# --------------------------------------------------------------------------- #


def _per_site_summary(
    mean: list[float],
    se: list[float],
    std: list[float] | None = None,
) -> dict[str, Any]:
    """Build a minimal per_site dict for one site."""
    n = len(mean)
    return {
        "mean": mean,
        "between_chain_se": se,
        "std": std if std is not None else [1.0] * n,
        "q05": [0.0] * n,
        "q95": [0.0] * n,
        "bulk_ess": [500.0] * n,
        "tail_ess": [500.0] * n,
        "rhat": [1.005] * n,
    }


def _coh_summary(
    per_site: dict[str, Any],
    n_chains: int = 10,
) -> dict[str, Any]:
    """Wrap a per_site dict into a minimal summary_v2-shaped dict."""
    return {"per_site": per_site, "n_chains": n_chains}


# --------------------------------------------------------------------------- #
# _check_coherence unit tests — existing behaviour
# --------------------------------------------------------------------------- #


def test_coherence_pass_identical_summaries() -> None:
    """Identical generated and committed summaries → z=0 → pass."""
    per_site = {"mu": _per_site_summary([1.0, 2.0], [0.01, 0.01])}
    summary = _coh_summary(per_site)
    passed, results, meta = _check_coherence(summary, summary)
    assert passed
    assert all(r["max_z"] == pytest.approx(0.0) for r in results)


def test_coherence_fail_large_deviation() -> None:
    """Generated mean deviates by 10σ → fail (dual gate: large z and large mat)."""
    gen = _coh_summary({"mu": _per_site_summary([11.0], [0.1])})
    com = _coh_summary({"mu": _per_site_summary([1.0], [0.1], std=[1.0])})
    passed, results, meta = _check_coherence(gen, com)
    # new se_denom = sqrt(0.1^2 + 0.1^2) = 0.1*sqrt(2)
    expected_z = 10.0 / (0.1 * np.sqrt(2))
    assert not passed
    assert results[0]["max_z"] == pytest.approx(expected_z, rel=1e-5)
    # materiality: |11-1|/std=10 >> TAU_SCI → hard FAIL dim
    assert len(results[0]["hard_fail_dims"]) > 0


def test_coherence_se_floor() -> None:
    """Sites with near-zero SE use _SE_FLOOR to avoid division by zero."""
    gen = _coh_summary({"mu": _per_site_summary([1.0], [0.0])})
    com = _coh_summary({"mu": _per_site_summary([1.0], [0.0])})
    passed, results, meta = _check_coherence(gen, com)
    assert passed
    assert np.isfinite(results[0]["max_z"])


def test_coherence_skip_missing_site() -> None:
    """Sites in generated but not committed are skipped (logged, no error)."""
    gen = _coh_summary(
        {
            "mu": _per_site_summary([1.0], [0.01]),
            "sigma": _per_site_summary([0.5], [0.005]),
        }
    )
    com = _coh_summary({"mu": _per_site_summary([1.0], [0.01])})
    passed, results, meta = _check_coherence(gen, com)
    assert passed
    assert len(results) == 1  # sigma was skipped


# --------------------------------------------------------------------------- #
# _check_coherence — NEW behaviour: denominator, dimension-aware, materiality
# --------------------------------------------------------------------------- #


def test_coherence_denominator_sqrt_vs_max() -> None:
    """Equal-SE case: new sqrt denominator = old max denom × sqrt(2), so z is halved by sqrt(2).

    Old formula: denom = max(se_new, se_com) = se
    New formula: denom = sqrt(se_new^2 + se_com^2) = se*sqrt(2)

    For equal se_new=se_com=se, the new z equals old z / sqrt(2).
    """
    se = 0.1
    delta = 10.0  # mean difference
    # Use large std_com so materiality check doesn't affect which formula we measure
    gen = _coh_summary({"x": _per_site_summary([delta], [se])})
    com = _coh_summary({"x": _per_site_summary([0.0], [se], std=[1000.0])})

    passed, results, meta = _check_coherence(gen, com)
    new_z = results[0]["max_z"]

    # Old denominator: max(se, se) = se = 0.1 → old_z = delta/se = 100.0
    old_z = delta / se
    expected_new_z = old_z / np.sqrt(2)
    assert new_z == pytest.approx(expected_new_z, rel=1e-5), (
        f"new_z={new_z:.4f} should equal old_z/sqrt(2)={expected_new_z:.4f}. "
        "Denominator not using sqrt formula."
    )


def test_coherence_high_d_dimension_aware_pass() -> None:
    """High-D model (D=1500): max_z=7 passes as REVIEW under new gate, fails old fixed-3.0.

    With D=1500 and n_chains=10:
        z_crit = t.ppf(1 - 0.05/3000, 18) ≈ 5.48

    max_z=7 > z_crit=5.48, but all deltas are immaterial (mat << 0.05), so the
    dual gate classifies the hottest dim as REVIEW (not FAIL) and the gate passes.
    Under the old fixed threshold of 3.0, max_z=7 would be a hard FAIL.
    """
    D = 1500
    se = 0.01
    std_c = 100.0  # large committed std → tiny materiality
    se_denom = np.sqrt(2) * se  # for equal se_new=se_com
    target_z = 7.0
    delta_hot = target_z * se_denom  # delta that gives z=target_z

    mean_gen = np.zeros(D)
    mean_gen[0] = delta_hot  # one "hot" dimension

    gen = _coh_summary(
        {
            "theta": {
                "mean": mean_gen.tolist(),
                "between_chain_se": [se] * D,
                "std": [1.0] * D,
                "q05": [0.0] * D,
                "q95": [0.0] * D,
                "bulk_ess": [500.0] * D,
                "tail_ess": [500.0] * D,
                "rhat": [1.005] * D,
            }
        },
        n_chains=10,
    )
    com = _coh_summary(
        {
            "theta": {
                "mean": [0.0] * D,
                "between_chain_se": [se] * D,
                "std": [std_c] * D,  # large committed std → small materiality
                "q05": [0.0] * D,
                "q95": [0.0] * D,
                "bulk_ess": [500.0] * D,
                "tail_ess": [500.0] * D,
                "rhat": [1.005] * D,
            }
        },
        n_chains=10,
    )

    passed, results, meta = _check_coherence(gen, com)

    # Gate passes despite max_z >> 3.0 (old threshold)
    assert passed, (
        f"D=1500 immaterial case should pass under new gate. "
        f"D_total={meta['D_total']} z_crit={meta['z_crit']:.3f} max_z={results[0]['max_z']:.3f}"
    )
    assert meta["D_total"] == D
    # The hot dim (dim 0) has z=7 > 3.0; under old gate this would FAIL
    hot_z = results[0]["z_per_dim"][0]
    assert hot_z == pytest.approx(target_z, rel=1e-4)
    # materiality of hot dim << TAU_SCI
    hot_mat = results[0]["mat_per_dim"][0]
    assert hot_mat < _TAU_SCI, f"hot_mat={hot_mat:.5f} should be < TAU_SCI={_TAU_SCI}"


def test_coherence_materiality_hard_fail() -> None:
    """Large z AND large Δμ/σ (≥ TAU_SCI) → hard FAIL."""
    se = 0.01
    delta = 1.0  # large shift
    std_c = 1.0  # std_com=1 → mat=delta/std=1.0 >> 0.05
    gen = _coh_summary({"mu": _per_site_summary([delta], [se])})
    com = _coh_summary({"mu": _per_site_summary([0.0], [se], std=[std_c])})

    passed, results, meta = _check_coherence(gen, com)
    assert not passed, "Large z + large mat should hard-FAIL the gate"
    assert results[0]["verdict"] == "FAIL"
    assert len(results[0]["hard_fail_dims"]) > 0


def test_coherence_materiality_review_pass() -> None:
    """Large z but tiny Δμ/σ (< TAU_SCI) → REVIEW verdict, gate passes."""
    # se=1e-6 → se_denom=sqrt(2)*1e-6; delta=0.001 → z≈707 >> z_crit
    # std_com=100 → mat=0.001/100=1e-5 << TAU_SCI=0.05
    se = 1e-6
    delta = 0.001
    std_c = 100.0

    gen = _coh_summary({"x": _per_site_summary([delta], [se])})
    com = _coh_summary({"x": _per_site_summary([0.0], [se], std=[std_c])})

    passed, results, meta = _check_coherence(gen, com)
    assert (
        passed
    ), "Large z but immaterial (mat << TAU_SCI) should be REVIEW (counts as pass)"
    assert results[0]["verdict"] == "REVIEW"
    assert len(results[0]["review_dims"]) > 0
    assert len(results[0]["hard_fail_dims"]) == 0
    # Confirm materiality is indeed below threshold
    hot_mat = results[0]["mat_per_dim"][0]
    assert hot_mat < _TAU_SCI, f"mat={hot_mat:.6f} should be < TAU_SCI={_TAU_SCI}"


def test_coherence_missing_committed_site_fails() -> None:
    """A site in the committed summary absent from generated → hard FAIL."""
    gen = _coh_summary({"mu": _per_site_summary([1.0], [0.01])})
    com = _coh_summary(
        {
            "mu": _per_site_summary([1.0], [0.01]),
            "sigma": _per_site_summary([0.5], [0.005]),  # present in committed only
        }
    )

    passed, results, meta = _check_coherence(gen, com)
    assert not passed, "Missing committed site should cause the gate to FAIL"
    assert "sigma" in meta["missing_committed_sites"]


# --------------------------------------------------------------------------- #
# Self-consistency integration: german_credit split-half coherence
# --------------------------------------------------------------------------- #

_GERMAN_CREDIT_DRAWS = (
    Path(__file__).parent.parent.parent
    / "tuningfork"
    / "catalog"
    / "german_credit"
    / "groundtruth_samples"
    / "blackjax"
    / "draws.npz"
)


def _half_summary(arr: np.ndarray) -> dict[str, Any]:
    """Compute a minimal per-site summary from draws array (n_chains, n_draws, D).

    Uses only numpy so this stays a fast test (no JAX/blackjax.diagnostics).
    Fields: mean, std, between_chain_se — sufficient for _check_coherence.
    """
    nc, ns, D = arr.shape
    flat = arr.reshape(nc, ns, D)
    chain_means = flat.mean(axis=1)  # (nc, D)
    pooled = flat.reshape(-1, D)  # (nc*ns, D)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0, ddof=1)
    be_se = chain_means.std(axis=0, ddof=1) / np.sqrt(nc)
    return {
        "per_site": {
            "beta": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "between_chain_se": be_se.tolist(),
            }
        },
        "n_chains": nc,
    }


@pytest.mark.skipif(
    not _GERMAN_CREDIT_DRAWS.exists(),
    reason="german_credit draws.npz not present in catalog",
)
def test_coherence_self_consistency_german_credit() -> None:
    """Split german_credit 10-chain run into two halves; halves should be coherent.

    german_credit: beta shape (10, 10000, 26), D=26.
    Split: chains 0–4 vs chains 5–9.  Both halves sample the same posterior so
    their means should agree within SE — the coherence gate should PASS.

    This tests the new formula end-to-end on real catalog draws without
    requiring a full GT regeneration.  Covers the 7-model scenario via the
    high-D + self-consistency design (real-data complement to synthetic tests).
    """
    data = np.load(_GERMAN_CREDIT_DRAWS)
    beta = data["beta"]  # (10, 10000, 26)

    # Deterministic split: first vs second half of chains
    summary_A = _half_summary(beta[:5])  # chains 0–4
    summary_B = _half_summary(beta[5:])  # chains 5–9

    passed, results, meta = _check_coherence(summary_A, summary_B, alpha=0.05)

    assert passed, (
        f"Half-split coherence check failed on german_credit. "
        f"D_total={meta['D_total']} nu={meta['nu']} z_crit={meta['z_crit']:.3f}. "
        f"Failing sites: {[r for r in results if not r['passed']]}"
    )
    assert meta["D_total"] == 26
    assert not meta["missing_committed_sites"]


# --------------------------------------------------------------------------- #
# verify_groundtruth integration: committed GT verifies against itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ["mvn_10", "radon"])
def test_verify_committed_gt_passes_against_itself(model_name: str) -> None:
    """Committed GT summary passes gate + coherence when compared to itself."""
    committed = load_committed_summary(model_name)
    gt_draws = committed_gt_dir(model_name) / "draws.npz"

    result = verify_groundtruth(
        model_name,
        committed,
        gt_draws,
        print_results=False,
    )
    assert result, (
        f"{model_name}: committed GT failed verify against itself "
        "(gate or coherence check)"
    )


def test_verify_returns_false_on_gate_fail() -> None:
    """verify_groundtruth returns False when gate fails."""
    committed = load_committed_summary("mvn_10")
    # Inject a bad rhat into the committed summary
    bad_summary = dict(committed)
    bad_summary["quality_gate"] = dict(committed["quality_gate"])
    bad_summary["quality_gate"]["max_rhat"] = 1.5  # clearly fails
    gt_draws = committed_gt_dir("mvn_10") / "draws.npz"

    result = verify_groundtruth(
        "mvn_10",
        bad_summary,
        gt_draws,
        print_results=False,
    )
    assert not result


def test_verify_returns_false_on_coherence_fail() -> None:
    """verify_groundtruth returns False when coherence fails."""
    committed = load_committed_summary("mvn_10")
    # Shift all per_site means by 100× posterior std → z >> z_crit and mat >> TAU_SCI
    bad_summary = dict(committed)
    bad_per_site = {}
    for site, stats in committed["per_site"].items():
        bad_stats = dict(stats)
        mean_arr = np.asarray(stats["mean"])
        std_arr = np.asarray(stats["std"])
        bad_stats["mean"] = (mean_arr + 100.0 * std_arr).tolist()
        bad_per_site[site] = bad_stats
    bad_summary["per_site"] = bad_per_site
    gt_draws = committed_gt_dir("mvn_10") / "draws.npz"

    result = verify_groundtruth(
        "mvn_10",
        bad_summary,
        gt_draws,
        print_results=False,
    )
    assert not result
