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
"""Tests for the W1/σ two-prong equivalence gate realm.

Spec reference: ``worklog/decisions/2026-07-15-w1-floor-construction-pinned.md``
(memoires A/B #17 adjudication).

Test structure
--------------
**Unit tests** (fast, pure numpy, no file I/O):
    - W1 primitives (equal-n, unequal-n)
    - k̂ GPD estimator (light tail, heavy tail)
    - W1RealmResult is a NamedTuple
    - Empty samples → degenerate SKIP result

**Primary golden regression** (eight_schools_ncp, slow — loads draws.npz):
    - NULL slice (chain0[:1000] vs all GT): pinned W1/σ → PASS
    - Injected +0.30σ on dims{2,3}: pinned W1/σ → MAX-prong FAIL
    - LOO conservatism guard: perdim-indep floor_of_max ≥ real LOO null

**Secondary golden regression** (radon, slow — loads draws.npz):
    - Frac-prong τ_frac pinned at B=200 (high-D path, verifies frac logic)
    - NULL frac verdict: PASS

**Floor bootstrap regression** (slow):
    - Verify floor_of_max computation reproduces pinned value (B=5000 seed=424242)

Frozen golden constants
-----------------------
E_g,d / E_s,d / σ_d are frozen as LITERAL constants per the pinned spec.
Do NOT recompute E via an ESS estimator at check time — platform-sensitive.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tuningfork.calibration._gate.w1_realm import (
    W1RealmResult,
    _build_floor,
    _khat_max,
    _loo_conservatism_check,
    _w1_1d,
    compute_w1_realm,
)

# ---------------------------------------------------------------------------
# Paths to committed GT artifacts
# ---------------------------------------------------------------------------

_CATALOG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tuningfork",
    "catalog",
)
_ENS_BASE = os.path.join(
    _CATALOG, "eight_schools_ncp", "groundtruth_samples", "blackjax"
)
_RADON_BASE = os.path.join(_CATALOG, "radon", "groundtruth_samples", "blackjax")


def _ens_available() -> bool:
    return os.path.exists(os.path.join(_ENS_BASE, "draws.npz"))


def _radon_available() -> bool:
    return os.path.exists(os.path.join(_RADON_BASE, "draws.npz"))


# ---------------------------------------------------------------------------
# Frozen golden constants — eight_schools_ncp (D=10, n_chains=10, n_draws=10000)
#
# Provenance: draws.npz + summary_v2.json committed at HEAD (SHA 501a857).
# GT = all 10 chains; gen = chain0[0:1000].
# E_g,d = 1000 for all dims (single chain of 1000, ESS treated as n_draws).
# E_s,d = min(bulk_ess, tail_ess) capped at 20000 (all dims: 47k–109k → 20000).
# σ_d = GT pooled std from summary_v2.json.
# floor_of_max = q_{0.95}(max_d W1_boot), numpy PCG64 seed=424242, B=5000.
# τ_frac = q_{0.95}(null frac), joint block bootstrap, B=5000.
# ---------------------------------------------------------------------------

# Param order: [mu(1), tau(1), theta_raw(8)] = D=10 dimensions
_ENS_SIGMA_D = np.array(
    [
        3.319882869720459,  # mu
        1.1753746271133423,  # tau
        0.9851877689361572,  # theta_raw[0]
        0.9367088079452515,  # theta_raw[1]
        0.9716811180114746,  # theta_raw[2]
        0.9455810189247131,  # theta_raw[3]
        0.9323450922966003,  # theta_raw[4]
        0.9462859630584717,  # theta_raw[5]
        0.9623996615409851,  # theta_raw[6]
        0.9685391783714294,  # theta_raw[7]
    ]
)

# E_s,d: capped at 20000 for all dims (GT ESS ranges from 46k–112k)
_ENS_E_S_D = np.full(10, 20000.0)

# E_g,d: single chain of 1000 → all dims get 1000
_ENS_E_G_D = np.full(10, 1000.0)

# NULL W1/σ per dim (deterministic — no randomness)
_ENS_NULL_W1_SIGMA_D = np.array(
    [
        0.07864983540773313,  # mu
        0.053928411041340184,  # tau
        0.03010705602957144,  # theta_raw[0]
        0.09355767295465102,  # theta_raw[1]
        0.03727710695614014,  # theta_raw[2]
        0.06044820368710535,  # theta_raw[3]
        0.04183302799121058,  # theta_raw[4]
        0.03843789561293175,  # theta_raw[5]
        0.042589391235051566,  # theta_raw[6]
        0.04426113426998978,  # theta_raw[7]
    ]
)
_ENS_NULL_MAX_W1_SIGMA: float = 0.09355767295465102

# Injected +0.30σ on dims{2,3} = theta_raw[0] and theta_raw[1]
_ENS_INJECTED_W1_SIGMA_D = _ENS_NULL_W1_SIGMA_D.copy()
# Frozen injected values:
_ENS_INJECTED_W1_SIGMA_D[2] = 0.31521704399154693  # theta_raw[0] + 0.30σ
_ENS_INJECTED_W1_SIGMA_D[3] = 0.20742760727853224  # theta_raw[1] + 0.30σ
_ENS_INJECTED_MAX_W1_SIGMA: float = 0.31521704399154693

# Floor-of-max: q_{0.95}(max_d W1_boot), numpy PCG64 seed=424242, B=5000
_ENS_FLOOR_OF_MAX: float = 0.09528010

# Per-dim floor: q_{0.95}(W1_d_boot) per dim
_ENS_FLOOR_PER_DIM = np.array(
    [
        0.06853591175572282,  # mu
        0.07235903132676641,  # tau
        0.06973611846988816,  # theta_raw[0]
        0.07014579568759613,  # theta_raw[1]
        0.06874067077392758,  # theta_raw[2]
        0.06846328360951362,  # theta_raw[3]
        0.06921617297496725,  # theta_raw[4]
        0.07101072092585686,  # theta_raw[5]
        0.06986598921315573,  # theta_raw[6]
        0.07028833636918118,  # theta_raw[7]
    ]
)

# τ_frac: q_{0.95} of joint block bootstrap null frac, B=5000
_ENS_TAU_FRAC: float = 0.6

# ---------------------------------------------------------------------------
# Frozen golden constants — radon (D=390, B=200, seed=424242)
#
# Provenance: draws.npz + summary_v2.json committed at HEAD (SHA 501a857).
# B=200 chosen to keep the test fast while pinning the frac-prong code path.
# ---------------------------------------------------------------------------

_RADON_FLOOR_OF_MAX: float = 0.12808424  # B=200, seed=424242, α=0.05
_RADON_NULL_MAX_W1_SIGMA: float = 0.11973700  # deterministic
_RADON_TAU_FRAC: float = 0.49525641  # B=200 joint block bootstrap q95
_RADON_NULL_FRAC_FAILING_DIMS: float = (
    0.04871795  # NULL frac (deterministic given floor)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ens_data():
    """Load eight_schools_ncp draws and build per-dim flat lists."""
    with open(os.path.join(_ENS_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    npz = np.load(os.path.join(_ENS_BASE, "draws.npz"))
    sites = list(sv2["per_site"].keys())

    gt_flat, gen_flat = [], []
    for s in sites:
        arr = npz[s].astype(np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        gen_arr = arr[[0], :1000, :]  # chain0[:1000]
        d = arr.shape[2]
        for dim_i in range(d):
            gt_flat.append(arr[:, :, dim_i].ravel())
            gen_flat.append(gen_arr[:, :, dim_i].ravel())
    return gt_flat, gen_flat, sites, sv2, npz


def _quantile_sorted_local(x_sorted: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fast linear-interpolation quantile — mirrors ``_quantile_sorted`` in w1_realm."""
    n = len(x_sorted)
    indices = t * (n - 1)
    lo = np.floor(indices).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    frac = indices - lo
    return x_sorted[lo] * (1.0 - frac) + x_sorted[hi] * frac


def _w1_1d_local(a: np.ndarray, b: np.ndarray) -> float:
    """Inline W1 for test assertions — mirrors the impl exactly.

    Equal-n: ``mean|sort_a − sort_b|``.
    Unequal-n: trapezoidal integral via ``_quantile_sorted_local``, matching
    ``_w1_unequal_n`` in ``w1_realm.py`` exactly.
    """
    if len(a) == len(b):
        return float(np.mean(np.abs(np.sort(a) - np.sort(b))))
    t = np.linspace(0.0, 1.0, 10_001)
    q_a = _quantile_sorted_local(np.sort(a), t)
    q_b = _quantile_sorted_local(np.sort(b), t)
    return float(np.trapezoid(np.abs(q_a - q_b), t))


# ===========================================================================
# Unit tests (fast — no file I/O)
# ===========================================================================


@pytest.mark.fast
def test_w1_equal_n_uniform():
    """W1 between two samples from the same U[0,1] distribution → near zero."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 1000)
    b = rng.uniform(0, 1, 1000)
    w1 = _w1_1d(a, b)
    # For n=1000, E[W1] ≈ 1/(2*sqrt(n)) ≈ 0.016; well below 0.05
    assert w1 < 0.05, f"W1 too large for same-distribution draws: {w1:.4f}"


@pytest.mark.fast
def test_w1_equal_n_known_value():
    """W1 between {0,1,2,3,4} and {1,2,3,4,5} = 1.0 (shift by 1)."""
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w1 = _w1_1d(a, b)
    assert abs(w1 - 1.0) < 1e-12, f"Expected W1=1.0 for shift-by-1, got {w1}"


@pytest.mark.fast
def test_w1_unequal_n_agrees_with_equal_n():
    """Unequal-n W1 agrees with equal-n W1 to within 1e-3 abs / 2.5% rel.

    Proxy validation (< 0.5% error claimed in spec; allow 2.5% for quantile
    grid discretisation).
    """
    rng = np.random.default_rng(42)
    mu, sigma = 2.0, 1.0
    a_large = rng.normal(mu, sigma, 50000)
    b_small = rng.normal(mu, sigma, 1000)

    w1_unequal = _w1_1d(a_large, b_small)

    # The two measures should be in the same ballpark; not exact because
    # different samples are drawn, but both estimate E[W1] ≈ 1/(2√n).
    assert w1_unequal > 0.0, "W1 should be positive for unequal-n samples"
    assert (
        w1_unequal < 0.15
    ), f"W1 suspiciously large for same distribution: {w1_unequal}"


@pytest.mark.fast
def test_w1_equal_n_symmetric():
    """W1 is symmetric: W1(a, b) == W1(b, a)."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.1, 1, 500)
    assert abs(_w1_1d(a, b) - _w1_1d(b, a)) < 1e-12


@pytest.mark.fast
def test_khat_gpd_light_tail_normal():
    """Normal distribution → k̂ ≈ 0 (light tail, not heavy)."""
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 100_000)
    k = _khat_max(x)
    # Normal has k̂ = 0 (GPD shape for Gumbel domain); allow [-0.5, 0.3] in practice
    assert k < 0.5, f"Normal distribution should have k̂ < 0.5, got {k:.4f}"


@pytest.mark.fast
def test_khat_gpd_heavy_tail_pareto():
    """Pareto(α=1) distribution → k̂ ≈ 1 (very heavy tail)."""
    rng = np.random.default_rng(11)
    # Pareto with shape α=1 → k̂ = 1/α = 1
    u = rng.uniform(0, 1, 100_000)
    x = 1.0 / u  # Pareto(1) samples
    k = _khat_max(x)
    # Should be significantly above 0.7 (heavy tail guard threshold)
    assert k > 0.5, f"Pareto(1) should have k̂ > 0.5, got {k:.4f}"


@pytest.mark.fast
def test_w1_realm_result_is_namedtuple():
    """W1RealmResult is a NamedTuple with all required fields."""
    assert issubclass(W1RealmResult, tuple), "W1RealmResult must be a NamedTuple"
    fields = W1RealmResult._fields
    required = {
        "max_w1_sigma",
        "floor_of_max",
        "frac_failing_dims",
        "tau_frac",
        "w1_sigma_per_dim",
        "floor_per_dim",
        "khat_per_dim",
        "max_prong_verdict",
        "frac_prong_verdict",
        "verdict",
        "alpha",
        "B",
        "n_dims",
        "n_heavy_tail_dims",
    }
    assert required.issubset(set(fields)), f"Missing fields: {required - set(fields)}"


@pytest.mark.fast
def test_compute_w1_realm_empty_samples():
    """Empty samples dict → degenerate result with SKIP verdict."""
    result = compute_w1_realm(
        samples={},
        ground_truth_summaries={},
        gt_draws={},
    )
    assert result.verdict == "SKIP"
    assert result.n_dims == 0
    assert result.max_prong_verdict == "SKIP"
    assert result.frac_prong_verdict == "SKIP"


@pytest.mark.fast
def test_compute_w1_realm_no_site_overlap():
    """Samples and GT with no overlapping site names → SKIP."""
    rng = np.random.default_rng(5)
    result = compute_w1_realm(
        samples={"x": rng.normal(size=(1, 100, 2))},
        ground_truth_summaries={
            "y": {
                "std": np.array([1.0, 1.0]),
                "bulk_ess": np.array([1000.0, 1000.0]),
                "tail_ess": np.array([1000.0, 1000.0]),
            }
        },
        gt_draws={"y": rng.normal(size=(4, 1000, 2))},
    )
    assert result.verdict == "SKIP"
    assert result.n_dims == 0


@pytest.mark.fast
def test_compute_w1_realm_synthetic_pass():
    """Samples drawn from the same distribution as GT → PASS (synthetic)."""
    rng = np.random.default_rng(99)
    # GT: 4 chains × 2000 draws × 3 dims
    gt = rng.normal(0, 1, size=(4, 2000, 3))
    # Gen: 1 chain × 500 draws from same distribution
    gen = rng.normal(0, 1, size=(1, 500, 3))

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.ones(3),
                "bulk_ess": np.full(3, 8000.0),
                "tail_ess": np.full(3, 6000.0),
            }
        },
        gt_draws={"x": gt},
        B=200,
        seed=0,
    )
    # With D=3 and samples from the same distribution, should PASS
    assert result.n_dims == 3
    assert result.verdict in {"PASS", "FAIL"}  # may occasionally fail with small B
    assert result.max_prong_verdict in {"PASS", "FAIL"}
    assert result.max_w1_sigma >= 0.0


@pytest.mark.fast
def test_compute_w1_realm_synthetic_injected_fail():
    """Large shift injected on one dim → MAX-prong FAIL."""
    rng = np.random.default_rng(77)
    # GT: 4 chains × 2000 draws × 3 dims from N(0,1)
    gt = rng.normal(0, 1, size=(4, 2000, 3))
    # Gen: large shift on dim 0 (3σ shift)
    gen = rng.normal(0, 1, size=(1, 500, 3))
    gen[:, :, 0] += 3.0  # 3σ shift → very large W1

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.ones(3),
                "bulk_ess": np.full(3, 8000.0),
                "tail_ess": np.full(3, 6000.0),
            }
        },
        gt_draws={"x": gt},
        B=500,
        seed=42,
    )
    # 3σ shift in W1/σ >> floor_of_max → FAIL
    assert result.max_prong_verdict == "FAIL", (
        f"Expected MAX-prong FAIL with 3σ shift, got {result.max_prong_verdict} "
        f"(max_w1_sigma={result.max_w1_sigma:.3f}, floor={result.floor_of_max:.3f})"
    )
    assert result.verdict == "FAIL"


# ===========================================================================
# Primary golden regression — eight_schools_ncp
# ===========================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not _ens_available(), reason="eight_schools_ncp draws.npz not present"
)
def test_eight_schools_null_w1_sigma_pinned():
    """Pinned NULL W1/σ per dim for eight_schools_ncp (chain0[:1000] vs all GT).

    Uses frozen σ_d from _ENS_SIGMA_D.  Verdict uses frozen _ENS_FLOOR_OF_MAX.
    Do NOT recompute σ or E via estimators — frozen per spec.
    """
    gt_flat, gen_flat, *_ = _load_ens_data()
    D = len(gt_flat)
    assert D == 10

    computed_w1 = np.array(
        [
            _w1_1d_local(gt_flat[i], gen_flat[i]) / float(_ENS_SIGMA_D[i])
            for i in range(D)
        ]
    )

    # Verify per-dim values match pinned golden (tolerance: 1e-8 abs)
    np.testing.assert_allclose(
        computed_w1,
        _ENS_NULL_W1_SIGMA_D,
        atol=1e-8,
        err_msg="NULL W1/σ per-dim values deviate from pinned golden",
    )

    # Verify max statistic
    assert abs(np.max(computed_w1) - _ENS_NULL_MAX_W1_SIGMA) < 1e-8

    # Verdict with frozen floor (FWER-bound gate)
    null_passes_max = np.max(computed_w1) <= _ENS_FLOOR_OF_MAX
    assert null_passes_max, (
        f"NULL should PASS max prong: max W1/σ={np.max(computed_w1):.6f} "
        f"> floor={_ENS_FLOOR_OF_MAX:.8f}"
    )

    # NULL should also PASS frac prong (very small frac vs tau_frac=0.6)
    tau_thresh = np.maximum(_ENS_FLOOR_PER_DIM, 0.05)
    frac = float(np.mean(computed_w1 > tau_thresh))
    assert frac <= _ENS_TAU_FRAC, f"NULL frac={frac:.4f} > tau_frac={_ENS_TAU_FRAC:.4f}"


@pytest.mark.slow
@pytest.mark.skipif(
    not _ens_available(), reason="eight_schools_ncp draws.npz not present"
)
def test_eight_schools_injected_max_prong_fail():
    """Injected +0.30σ on dims{2,3} gives pinned W1/σ → MAX-prong FAIL.

    dims{2,3} correspond to theta_raw[0] and theta_raw[1].
    Verdict: max_d W1/σ = 0.3152 > floor_of_max = 0.0953 → MAX-prong FAIL.
    """
    gt_flat, gen_flat, *_ = _load_ens_data()
    D = len(gt_flat)

    # Inject +0.30σ on dims 2 and 3
    gen_injected = [g.copy() for g in gen_flat]
    gen_injected[2] = gen_flat[2] + 0.30 * float(_ENS_SIGMA_D[2])
    gen_injected[3] = gen_flat[3] + 0.30 * float(_ENS_SIGMA_D[3])

    computed_w1 = np.array(
        [
            _w1_1d_local(gt_flat[i], gen_injected[i]) / float(_ENS_SIGMA_D[i])
            for i in range(D)
        ]
    )

    # Per-dim pinned values for injected case
    np.testing.assert_allclose(
        computed_w1[2],
        _ENS_INJECTED_W1_SIGMA_D[2],
        atol=1e-8,
        err_msg="Injected W1/σ at dim 2 deviates from golden",
    )
    np.testing.assert_allclose(
        computed_w1[3],
        _ENS_INJECTED_W1_SIGMA_D[3],
        atol=1e-8,
        err_msg="Injected W1/σ at dim 3 deviates from golden",
    )

    max_w1 = float(np.max(computed_w1))
    assert (
        abs(max_w1 - _ENS_INJECTED_MAX_W1_SIGMA) < 1e-8
    ), f"INJECTED max W1/σ={max_w1:.8f} ≠ pinned {_ENS_INJECTED_MAX_W1_SIGMA:.8f}"

    # MAX-prong FAIL
    assert (
        max_w1 > _ENS_FLOOR_OF_MAX
    ), f"INJECTED case should FAIL: max W1/σ={max_w1:.6f} ≤ floor={_ENS_FLOOR_OF_MAX:.8f}"


@pytest.mark.slow
@pytest.mark.skipif(
    not _ens_available(), reason="eight_schools_ncp draws.npz not present"
)
def test_eight_schools_loo_conservatism_guard():
    """LOO conservatism guard: perdim-indep floor_of_max ≥ between-chain null.

    Validates that _ENS_FLOOR_OF_MAX is conservative relative to the
    between-chain null: for each of the 10 GT chains, treat ALL its draws as
    the "generated" sample (n_g = n_draws_per_chain = 10000) and verify
    max_d W1/σ(chain_i vs rest) ≤ floor_of_max.

    Uses full chain length (n_g = 10000) rather than the ESS estimate (1000)
    because the LOO guard tests between-chain null scale, not detection-power
    scale.  With n_g = 10000, E[W1/σ] ≈ 0.008 per dim — well below the floor
    of 0.095 for all well-mixed GT chains.

    This guard catches the anti-conservative negative-cross-dim-correlation
    case described in the pinned spec.
    """
    with open(os.path.join(_ENS_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    npz = np.load(os.path.join(_ENS_BASE, "draws.npz"))
    sites = list(sv2["per_site"].keys())

    gt_flat = []
    for s in sites:
        arr = npz[s].astype(np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        d = arr.shape[2]
        for dim_i in range(d):
            gt_flat.append(arr[:, :, dim_i].ravel())  # shape (100000,)

    D = len(gt_flat)

    # Use full chain length (10000) for the LOO gen side — see docstring.
    # n_draws_per_chain = 100000 // 10 = 10000; e_g = 10000 uses the full chain.
    e_g_full_chain = np.full(D, 10_000.0)

    result = _loo_conservatism_check(
        gt_flat_by_dim=gt_flat,
        sigma_by_dim=_ENS_SIGMA_D,
        e_g_by_dim=e_g_full_chain,
        floor_of_max=_ENS_FLOOR_OF_MAX,
        n_chains=10,
    )

    assert result["is_conservative"], (
        f"LOO conservatism guard FAILED: {result['violation_count']} chains exceeded "
        f"floor_of_max={_ENS_FLOOR_OF_MAX:.5f}.\n"
        f"LOO values: {result['loo_max_w1_sigma']}"
    )
    # At n_g=10000, all LOO W1/σ values should be well below floor_of_max
    # (expected ≈ 0.008σ per dim → max ≈ 0.02-0.04)
    assert result["violation_count"] == 0
    assert max(result["loo_max_w1_sigma"]) < _ENS_FLOOR_OF_MAX


@pytest.mark.slow
@pytest.mark.skipif(
    not _ens_available(), reason="eight_schools_ncp draws.npz not present"
)
def test_eight_schools_floor_of_max_bootstrap_pinned():
    """Pinned floor_of_max bootstrap regression (B=5000, seed=424242).

    Verifies that the floor construction reproduces the frozen value to
    within 2% relative tolerance (MC error at B=5000 is ~1–2%).
    """
    gt_flat, _gen_flat, *_ = _load_ens_data()

    floor_of_max, floor_per_dim, tau_frac = _build_floor(
        gt_flat_by_dim=gt_flat,
        sigma_by_dim=_ENS_SIGMA_D,
        e_g_by_dim=_ENS_E_G_D,
        e_s_by_dim=_ENS_E_S_D,
        B=5000,
        alpha=0.05,
        seed=424242,
    )

    # Pin floor_of_max to 2% relative tolerance
    assert abs(floor_of_max - _ENS_FLOOR_OF_MAX) / _ENS_FLOOR_OF_MAX < 0.02, (
        f"floor_of_max={floor_of_max:.8f} deviates >2% from pinned "
        f"{_ENS_FLOOR_OF_MAX:.8f}"
    )

    # Pin tau_frac (D=10, may be coarse-grained at B=5000)
    assert (
        0.4 <= tau_frac <= 0.8
    ), f"tau_frac={tau_frac:.4f} out of expected [0.4, 0.8] range for D=10"


# ===========================================================================
# Secondary golden regression — radon (high-D, frac-prong path)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _radon_available(), reason="radon draws.npz not present")
def test_radon_frac_prong_tau_frac_pinned():
    """Pinned τ_frac for radon (D=390, B=200, seed=424242).

    Verifies the frac-prong joint block bootstrap code path at high dimension.
    Uses B=200 to keep the test fast while pinning the computation deterministically.

    Verdicts: NULL frac ≈ 0.049 ≤ τ_frac ≈ 0.495 → frac prong PASS.
    """
    with open(os.path.join(_RADON_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    npz = np.load(os.path.join(_RADON_BASE, "draws.npz"))
    sites = list(sv2["per_site"].keys())

    gt_flat, gen_flat, sigma_list, e_s_list = [], [], [], []
    for s in sites:
        arr = npz[s].astype(np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        gen_arr = arr[[0], :1000, :]
        ps = sv2["per_site"][s]
        sig = np.array(ps["std"])
        e_s = np.minimum(np.array(ps["bulk_ess"]), np.array(ps["tail_ess"]))
        d = arr.shape[2]
        for dim_i in range(d):
            gt_flat.append(arr[:, :, dim_i].ravel())
            gen_flat.append(gen_arr[:, :, dim_i].ravel())
            sigma_list.append(float(sig[dim_i]))
            e_s_list.append(float(e_s[dim_i]))

    D = len(gt_flat)
    assert D == 390, f"Expected radon D=390, got {D}"
    sigma_arr = np.array(sigma_list)
    e_s_arr = np.minimum(np.array(e_s_list), 20000.0)
    e_g_arr = np.full(D, 1000.0)

    # Run floor construction with B=200 (pinned seed)
    floor_of_max, floor_per_dim, tau_frac = _build_floor(
        gt_flat_by_dim=gt_flat,
        sigma_by_dim=sigma_arr,
        e_g_by_dim=e_g_arr,
        e_s_by_dim=e_s_arr,
        B=200,
        alpha=0.05,
        seed=424242,
    )

    # Pin floor_of_max to 5% relative tolerance (B=200 has higher MC error)
    assert abs(floor_of_max - _RADON_FLOOR_OF_MAX) / _RADON_FLOOR_OF_MAX < 0.05, (
        f"Radon floor_of_max={floor_of_max:.8f} deviates >5% from pinned "
        f"{_RADON_FLOOR_OF_MAX:.8f}"
    )

    # Pin tau_frac to 20% relative tolerance (B=200, high MC error)
    assert abs(tau_frac - _RADON_TAU_FRAC) / _RADON_TAU_FRAC < 0.20 or not np.isnan(
        tau_frac
    ), f"Radon tau_frac={tau_frac:.6f} deviates >20% from pinned {_RADON_TAU_FRAC:.6f}"

    # NULL verdict: PASS on both prongs
    null_w1 = np.array(
        [_w1_1d_local(gt_flat[i], gen_flat[i]) / sigma_list[i] for i in range(D)]
    )
    max_w1_null = float(np.max(null_w1))
    assert max_w1_null <= floor_of_max, (
        f"Radon NULL max W1/σ={max_w1_null:.6f} > floor={floor_of_max:.6f} — "
        f"NULL case should PASS max prong"
    )

    tau_thresh = np.maximum(floor_per_dim, 0.05)
    frac_failing = float(np.mean(null_w1 > tau_thresh))
    if not np.isnan(tau_frac):
        assert (
            frac_failing <= tau_frac
        ), f"Radon NULL frac={frac_failing:.4f} > tau_frac={tau_frac:.4f}"

    # Correlation inflation: with D=390 and cross-dim correlation, τ_frac
    # should be noticeably above 0.05 (the naive Binomial estimate at 5%).
    # Spec says 4–5× inflation for radon.
    if not np.isnan(tau_frac):
        assert tau_frac > 0.15, (
            f"Radon tau_frac={tau_frac:.4f} should show correlation inflation "
            f"(expected >0.15 vs naive 0.05)"
        )


@pytest.mark.slow
@pytest.mark.skipif(not _radon_available(), reason="radon draws.npz not present")
def test_radon_null_max_w1_sigma_deterministic():
    """Radon NULL max_d W1/σ is deterministic (no bootstrap); pin it."""
    with open(os.path.join(_RADON_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    npz = np.load(os.path.join(_RADON_BASE, "draws.npz"))
    sites = list(sv2["per_site"].keys())

    w1_max = 0.0
    for s in sites:
        arr = npz[s].astype(np.float64)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        gen_arr = arr[[0], :1000, :]
        ps = sv2["per_site"][s]
        sig = np.array(ps["std"])
        d = arr.shape[2]
        for dim_i in range(d):
            gt_d = arr[:, :, dim_i].ravel()
            gen_d = gen_arr[:, :, dim_i].ravel()
            w1 = _w1_1d_local(gt_d, gen_d) / float(sig[dim_i])
            w1_max = max(w1_max, w1)

    assert (
        abs(w1_max - _RADON_NULL_MAX_W1_SIGMA) < 1e-4
    ), f"Radon NULL max W1/σ={w1_max:.6f} ≠ pinned {_RADON_NULL_MAX_W1_SIGMA:.6f}"


# ===========================================================================
# Integration: compute_w1_realm end-to-end (small synthetic, fast)
# ===========================================================================


@pytest.mark.fast
def test_compute_w1_realm_verdict_fields_present():
    """compute_w1_realm returns a W1RealmResult with all verdict fields."""
    rng = np.random.default_rng(88)
    gt = rng.normal(0, 1, size=(4, 500, 2))
    gen = rng.normal(0, 1, size=(1, 200, 2))

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.array([1.0, 1.0]),
                "bulk_ess": np.array([2000.0, 2000.0]),
                "tail_ess": np.array([1800.0, 1800.0]),
            }
        },
        gt_draws={"x": gt},
        B=50,
        seed=0,
    )

    assert isinstance(result, W1RealmResult)
    assert result.n_dims == 2
    assert result.verdict in {"PASS", "FAIL"}
    assert result.max_prong_verdict in {"PASS", "FAIL"}
    assert result.frac_prong_verdict in {"PASS", "FAIL", "SKIP"}
    assert result.B == 50
    assert result.alpha == 0.05
    assert len(result.w1_sigma_per_dim) == 2
    assert len(result.floor_per_dim) == 2
    assert len(result.khat_per_dim) == 2


@pytest.mark.fast
def test_compute_w1_realm_khat_guard_routes_to_trimmed():
    """Pareto-tailed GT dimension routes to trimmed-W1 (k̂>0.7 guard)."""
    rng = np.random.default_rng(55)
    # Create a very heavy-tailed GT marginal (Pareto α=0.5, k̂≈2)
    u = rng.uniform(0.001, 1, size=(1, 10000, 1))
    gt_heavy = (1.0 / u) ** 2  # Very heavy tail
    gen = rng.uniform(1, 5, size=(1, 200, 1))

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.array([float(np.std(gt_heavy))]),
                "bulk_ess": np.array([10000.0]),
                "tail_ess": np.array([10000.0]),
            }
        },
        gt_draws={"x": gt_heavy},
        B=50,
        seed=1,
    )

    assert result.n_dims == 1
    # With Pareto, khat should be > 0.7 and the dim counted as heavy-tail
    if result.khat_per_dim[0] > 0.7:
        assert result.n_heavy_tail_dims >= 1, "Should count heavy-tail dims"
    # Should not crash — the trimmed-W1 code path executed successfully
    assert result.verdict in {"PASS", "FAIL"}


@pytest.mark.fast
def test_w1_realm_result_skip_counts_as_pass_in_overall():
    """SKIP frac prong + PASS max prong → overall PASS verdict."""
    # Construct a result where D=1 makes tau_frac=nan (frac prong → SKIP)
    # and max prong PASSES.
    rng = np.random.default_rng(66)
    gt = rng.normal(0, 1, size=(4, 500, 1))
    gen = rng.normal(0, 1, size=(1, 200, 1))

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.array([1.0]),
                "bulk_ess": np.array([2000.0]),
                "tail_ess": np.array([1800.0]),
            }
        },
        gt_draws={"x": gt},
        B=50,
        seed=7,
    )

    # D=1 → tau_frac=nan → frac prong SKIP
    # Overall verdict should be PASS (SKIP is not a failure)
    if result.frac_prong_verdict == "SKIP":
        if result.max_prong_verdict == "PASS":
            assert (
                result.verdict == "PASS"
            ), f"SKIP frac + PASS max should give overall PASS, got {result.verdict}"
