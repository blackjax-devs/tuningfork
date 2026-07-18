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
# _check_coherence — formula pin tests (SF-1) and edge-case guards (SF-2, SF-3)
# --------------------------------------------------------------------------- #


def test_coherence_threshold_pin_formula() -> None:
    """Pin z_crit and nu for two known (D_total, n_chains) points (SF-1 guard).

    Two mutations that both shipped green under the materiality-masked suite:
      (a) alpha/(2*D_total) → alpha/D_total
      (b) nu = 2*(n_chains-1) → n_chains-1

    These assertions are load-bearing: they verify that BOTH the 2*D Bonferroni
    factor and the 2*(n_chains-1) df formula are correct, independently of the
    materiality gate masking them in pass/fail scenarios.

    Pin points (alpha=0.05, n_chains=10, nu=18):
      D=390  → z_crit ≈ 4.8514   (radon model size)
      D=26   → z_crit ≈ 3.6281   (german_credit model size)
    """
    from scipy import stats as scipy_stats

    alpha = 0.05

    def _make(D: int, n_chains: int = 10) -> dict:
        return _coh_summary(
            {"x": _per_site_summary([0.0] * D, [0.01] * D)}, n_chains=n_chains
        )

    # Pin point 1: D=390, n_chains=10 → nu=18, z_crit≈4.8514
    D1 = 390
    _, _, meta1 = _check_coherence(_make(D1), _make(D1))
    assert meta1["nu"] == 18, f"nu should be 2*(10-1)=18, got {meta1['nu']}"
    assert meta1["D_total"] == D1
    expected_z1 = float(scipy_stats.t.ppf(1.0 - alpha / (2.0 * D1), 18))
    assert meta1["z_crit"] == pytest.approx(expected_z1, rel=1e-5)
    assert meta1["z_crit"] == pytest.approx(4.8514, rel=1e-3), (
        f"z_crit for D={D1}: got {meta1['z_crit']:.4f}, expected ≈4.8514. "
        "Check: alpha/(2*D) vs alpha/D, nu=2*(nc-1) vs nc-1."
    )

    # Pin point 2: D=26, n_chains=10 → nu=18, z_crit≈3.6281
    D2 = 26
    _, _, meta2 = _check_coherence(_make(D2), _make(D2))
    assert meta2["nu"] == 18
    assert meta2["D_total"] == D2
    expected_z2 = float(scipy_stats.t.ppf(1.0 - alpha / (2.0 * D2), 18))
    assert meta2["z_crit"] == pytest.approx(expected_z2, rel=1e-5)
    assert meta2["z_crit"] == pytest.approx(
        3.6281, rel=1e-3
    ), f"z_crit for D={D2}: got {meta2['z_crit']:.4f}, expected ≈3.6281."


def test_coherence_shape_shrink_fails() -> None:
    """Matched site with fewer generated dims than committed → hard FAIL (SF-2 guard).

    A 999σ shift in the dropped committed dims must not silently pass.
    Generated has D=1, committed has D=3 (huge shift in dims 1,2 of committed).
    """
    gen = _coh_summary({"x": _per_site_summary([0.0], [0.01])})  # D=1
    com = _coh_summary(
        {
            "x": _per_site_summary(
                [0.0, 999.0, 999.0],
                [0.01, 0.01, 0.01],
                std=[1.0, 1.0, 1.0],
            )
        }
    )  # D=3, huge shift in dropped dims 1,2

    passed, results, meta = _check_coherence(gen, com)
    assert not passed, (
        "Shape-shrink (gen D=1 < committed D=3) should hard-FAIL the gate even "
        "when the checked dim is fine — dropped dims have 999σ shifts."
    )
    assert "x" in meta["shrunk_sites"], f"shrunk_sites should contain 'x'; got {meta}"
    assert results[0]["verdict"] == "FAIL"
    assert results[0].get("shape_shrink") is True


def test_coherence_materiality_boundary_strict_gt() -> None:
    """Materiality uses strict > so mat=_TAU_SCI exactly is REVIEW, not FAIL (SF-3).

    This boundary matters for irt_2pl, which lands exactly at 0.050σ shift —
    under the old >=, an exactly-on-boundary shift would hard-FAIL. With strict
    >, it is correctly treated as REVIEW (immaterial).

    Setup: se=1e-6 ensures z >> z_crit (so the dim IS flagged by z).  The
    materiality delta is varied around the 0.05σ boundary:
      delta = _TAU_SCI * std_com        → mat = 0.05  → strict > fails → REVIEW
      delta = _TAU_SCI * std_com + 1e-9 → mat > 0.05 → strict > passes → FAIL
    """
    se = 1e-6  # tiny SE → z >> z_crit regardless of delta
    std_c = 1.0

    # At the boundary: mat = exactly TAU_SCI → REVIEW (strict > is False)
    delta_at = _TAU_SCI * std_c  # = 0.05
    gen_at = _coh_summary({"x": _per_site_summary([delta_at], [se])})
    com_at = _coh_summary({"x": _per_site_summary([0.0], [se], std=[std_c])})
    passed_at, results_at, _ = _check_coherence(gen_at, com_at)
    assert (
        passed_at
    ), f"mat exactly at boundary ({_TAU_SCI}) should be REVIEW (strict >), not FAIL."
    assert results_at[0]["verdict"] == "REVIEW"

    # Just above the boundary: mat = TAU_SCI + epsilon → FAIL (strict > is True)
    delta_above = _TAU_SCI * std_c + 1e-9
    gen_above = _coh_summary({"x": _per_site_summary([delta_above], [se])})
    com_above = _coh_summary({"x": _per_site_summary([0.0], [se], std=[std_c])})
    passed_above, results_above, _ = _check_coherence(gen_above, com_above)
    assert (
        not passed_above
    ), f"mat just above boundary ({_TAU_SCI}+eps) should hard-FAIL."
    assert results_above[0]["verdict"] == "FAIL"


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


def _is_real_npz(path: Path) -> bool:
    """True only if `path` is a real .npz (zip) file, not an unfetched git-LFS pointer.

    .npz files are zip archives starting with the local-file-header magic bytes
    b"PK\x03\x04". Unfetched git-LFS pointers are text files starting with
    b"version https://git-lfs...".
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except OSError:
        return False


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
    not _is_real_npz(_GERMAN_CREDIT_DRAWS),
    reason="german_credit draws.npz absent or an unfetched git-LFS pointer",
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
    """verify_groundtruth returns False when coherence fails (and emits a warning)."""
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

    # The coherence FAIL emits a UserWarning (SF-4 non-silent-REVIEW requirement).
    with pytest.warns(UserWarning, match="FAIL"):
        result = verify_groundtruth(
            "mvn_10",
            bad_summary,
            gt_draws,
            print_results=False,
        )
    assert not result


def test_verify_review_emits_warning_when_print_false() -> None:
    """REVIEW dims emit a UserWarning even when print_results=False (SF-4 guard).

    Constructs a generated summary for mvn_10 (D=10) where every dim has:
      - tiny SE (1e-10) → z >> z_crit  (flagged by the threshold test)
      - small absolute shift (0.025 in posterior units, mat≈0.025 < 0.05) → REVIEW

    With print_results=False nothing is printed, but verify_groundtruth must still
    emit a UserWarning so the signal is never fully swallowed.
    """
    committed = load_committed_summary("mvn_10")

    # Build a generated summary: copy gate metadata (passes), inject REVIEW coherence.
    gen_per_site = {}
    for site, stats in committed["per_site"].items():
        gen_stats = dict(stats)
        mean_arr = np.asarray(stats["mean"])
        # Small absolute shift: mat ≈ 0.025/std ≈ 0.025 < 0.05 (immaterial)
        gen_stats["mean"] = (mean_arr + 0.025).tolist()
        # Tiny SE: z ≈ 0.025/se_committed >> z_crit (statistically flagged)
        gen_stats["between_chain_se"] = [1e-10] * len(mean_arr)
        gen_per_site[site] = gen_stats

    gen_summary = dict(committed)
    gen_summary["per_site"] = gen_per_site

    with pytest.warns(UserWarning, match="REVIEW"):
        result = verify_groundtruth("mvn_10", gen_summary, print_results=False)

    # REVIEW is not a FAIL — gate passes
    assert result, "REVIEW coherence verdict should return True (immaterial)"
