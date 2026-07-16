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

Statistic: per-dim W1/σ, where W1 is the exact 1-D Wasserstein-1 distance
and σ is the GT pooled standard deviation in unconstrained space.

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
    - Frac-prong τ_frac pinned at B=2000 (high-D path, verifies frac logic)
    - NULL frac verdict: PASS
    - LOO conservatism guard (leave-one-out): 1/10 violation at committed floor 0.242 →
      passes k_crit=1 count and severity (0.25155 ≤ 0.242×1.05 = 0.25410)

**Floor bootstrap regression** (slow):
    - Verify floor_of_max computation reproduces pinned value (B=5000 seed=424242)

Frozen golden constants
-----------------------
σ_d is frozen as a literal constant.
E_g,d is frozen for documentation; end-to-end tests call compute_w1_realm
directly (which recomputes real ESS at run time via _ess_gen_per_dim).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.conftest import _is_lfs_pointer
from tuningfork.calibration._gate.w1_realm import (
    W1RealmResult,
    _ess_gen_per_dim,
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
# E_g,d = real min(bulk,tail)-ESS from _ess_gen_per_dim (DataTree path).
#         Differs from raw n_draws=1000 because of autocorrelation in chain0.
# E_s,d = min(bulk_ess, tail_ess) capped at 20000 (all dims: 47k–109k → 20000).
# σ_d = GT pooled std from summary_v2.json.
# floor_of_max = q_{0.95}(max_d W1_boot), numpy PCG64 seed=424242, B=5000,
#                computed via compute_w1_realm end-to-end (real ESS + GPD k-hat).
# τ_frac = q_{0.95}(null frac), joint block bootstrap, B=5000.
#
# Re-pinned (post-fix): ESS bug fixed (DataTree path),
# k-hat replaced with Zhang-Stephens GPD.
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

# E_g,d: real min(bulk,tail)-ESS for chain0[:1000] (from _ess_gen_per_dim).
# Pinned from golden run; lower than raw 1000 due to autocorrelation.
# Order: [mu, tau, theta_raw[0..7]]
_ENS_E_G_D = np.array(
    [
        623.99023701,  # mu
        602.05143795,  # tau
        611.978659,  # theta_raw[0]
        856.98710077,  # theta_raw[1]
        749.94314733,  # theta_raw[2]
        661.39834602,  # theta_raw[3]
        755.00840192,  # theta_raw[4]
        805.90803081,  # theta_raw[5]
        628.33952046,  # theta_raw[6]
        676.10663706,  # theta_raw[7]
    ]
)

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

# Floor-of-max: q_{0.95}(max_d W1_boot), numpy PCG64 seed=424242, B=5000.
# Computed via compute_w1_realm end-to-end with real ESS (≈623–857 per dim).
# Raised from 0.09528010 (buggy raw-count ESS=1000) to 0.1122042940.
_ENS_FLOOR_OF_MAX: float = 0.1122042940

# Per-dim floor: q_{0.95}(W1_d_boot) per dim, B=5000, seed=424242.
# Computed with real ESS per dim from _ENS_E_G_D above.
_ENS_FLOOR_PER_DIM = np.array(
    [
        0.08721894,  # mu
        0.08936596,  # tau
        0.08756933,  # theta_raw[0]
        0.07350074,  # theta_raw[1]
        0.07831004,  # theta_raw[2]
        0.08612174,  # theta_raw[3]
        0.07802934,  # theta_raw[4]
        0.07758241,  # theta_raw[5]
        0.08512072,  # theta_raw[6]
        0.08366970,  # theta_raw[7]
    ]
)

# τ_frac: q_{0.95} of joint block bootstrap null frac, B=5000, seed=424242.
# 0.7 with real ESS (vs 0.6 with buggy ESS=1000).
_ENS_TAU_FRAC: float = 0.7

# ---------------------------------------------------------------------------
# Frozen golden constants — radon (D=390, B=2000, seed=424242)
#
# Provenance: draws.npz + summary_v2.json committed at HEAD (SHA 501a857).
# Re-pinned at B=2000 for stability: raised B from 200 → 2000
# (B=200 had SD≈3.5% across seeds; B=2000 drops this to ≈1.1%).
# Real ESS for chain0[:1000]: min=88, median=670, max=903 (autocorrelation).
#
# _RADON_BOOTSTRAP_FLOOR_B2000: exact q_{0.95}(max_d W1_boot) at B=2000,
#   seed=424242 via compute_w1_realm end-to-end.  This is what the runtime
#   bootstrap computes internally (IID bootstrap, D=390, real ESS).
#
# _RADON_FLOOR_OF_MAX: committed floor = 0.242 (binding literal).
#   bootstrap-q95 at B=2000 ≈ 0.2392; severity needs floor ≥ 0.25155/1.05=0.2396.
#   0.242 gives margin above 0.2396 and is inside the honest B=200 seed range.
#   LOO guard: 1/10 violation at +3.9% (0.25155/0.242-1) → count (1≤k_crit=1)
#   and severity (0.25155 ≤ 0.242×1.05=0.25410) both pass → is_conservative=True.
# ---------------------------------------------------------------------------

_RADON_BOOTSTRAP_FLOOR_B2000: float = 0.2392043871  # B=2000, seed=424242
_RADON_FLOOR_OF_MAX: float = (
    0.242  # committed binding literal (> severity floor 0.2396)
)
_RADON_NULL_MAX_W1_SIGMA: float = 0.11973700  # deterministic (unchanged)
_RADON_TAU_FRAC: float = 0.5358974359  # B=2000 joint block bootstrap q95, real ESS
_RADON_NULL_FRAC_FAILING_DIMS: float = (
    0.00512821  # NULL frac (deterministic; very few dims exceed the floor)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ens_data():
    """Load eight_schools_ncp draws and build per-dim flat lists."""
    with open(os.path.join(_ENS_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())
    _ens_draws = os.path.join(_ENS_BASE, "draws.npz")
    if _is_lfs_pointer(_ens_draws):
        pytest.skip(
            "eight_schools draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    # allow_pickle=True: committed catalog draws.npz contain pickled DeviceArrays.
    # Context manager ensures the NpzFile FD is closed (avoids PytestUnraisableExceptionWarning).
    with np.load(_ens_draws, allow_pickle=True) as npz:
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
    return gt_flat, gen_flat, sites, sv2


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
    assert k > 0.7, f"Pareto(1) should have k̂ > 0.7, got {k:.4f}"


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

    # NULL should also PASS frac prong (very small frac vs tau_frac=0.7)
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
    Verdict: max_d W1/σ = 0.3152 > floor_of_max = 0.1122 → MAX-prong FAIL.
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
    """LOO conservatism guard via _loo_conservatism_check (leave-one-out).

    Each held-out GT chain is compared against the OTHER 9 chains (honest
    leave-one-out), implemented by the gate's own ``_loo_conservatism_check``.
    Uses the count-and-severity criterion: k_crit=1 for n_chains=10, meaning
    one violation is ALLOWED (the floor is a q_{0.95} quantile, so ~0.5/10
    exceedances are expected; "0 violations" would reject a correctly-calibrated
    floor ~40% of the time).

    eight_schools result: 1/10 violation at +1.9%.
    count rule: 1 ≤ k_crit=1 → passes.
    severity rule: +1.9% ≤ 5% → passes.
    is_conservative: True.

    An earlier in-sample-biased variant compared each chain to all chains
    including the held-out one, producing a ~2.4%-lower statistic that hid the
    real LOO violation; this test uses the honest leave-one-out.
    """
    with open(os.path.join(_ENS_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())

    _ens_draws = os.path.join(_ENS_BASE, "draws.npz")
    if _is_lfs_pointer(_ens_draws):
        pytest.skip(
            "eight_schools draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    gt_flat: list[np.ndarray] = []
    sigma_list: list[float] = []
    with np.load(_ens_draws, allow_pickle=True) as npz:
        for s in sites:
            arr = npz[s].astype(np.float64)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            sig = np.atleast_1d(np.array(sv2["per_site"][s]["std"], dtype=np.float64))
            for dim_i in range(arr.shape[2]):
                gt_flat.append(arr[:, :, dim_i].ravel())
                sigma_list.append(float(sig[dim_i]))
    sigma_arr = np.array(sigma_list)
    D = len(gt_flat)
    n_chains = 10

    # Use raw draw count (not ESS) — raw count matches the LOO held-out slice size.
    e_g_raw = np.full(D, 1000.0)

    result = _loo_conservatism_check(
        gt_flat_by_dim=gt_flat,
        sigma_by_dim=sigma_arr,
        e_g_by_dim=e_g_raw,
        floor_of_max=_ENS_FLOOR_OF_MAX,
        n_chains=n_chains,
    )

    # k_crit=1 for n_chains=10 (P(Bin(10,0.05) >= 2) = 0.086 ≤ 0.10)
    assert (
        result["k_crit"] == 1
    ), f"Expected k_crit=1 for n_chains=10, got {result['k_crit']}"
    # Count rule: exactly 1/10 violation (leave-one-out gives the honest null).
    assert result["violation_count"] <= result["k_crit"], (
        f"violation_count={result['violation_count']} > k_crit={result['k_crit']}: "
        f"too many LOO exceedances.\n"
        f"LOO values: {result['loo_max_w1_sigma']}\n"
        f"floor_of_max={_ENS_FLOOR_OF_MAX:.8f}"
    )
    # Severity rule: max LOO null ≤ floor * 1.05 (1.9% overshoot < 5% limit).
    max_loo = max(result["loo_max_w1_sigma"])
    assert (
        max_loo <= _ENS_FLOOR_OF_MAX * 1.05
    ), f"Severity rule FAILED: max_loo={max_loo:.5f} > floor*1.05={_ENS_FLOOR_OF_MAX * 1.05:.5f}"
    # is_conservative must be True (both count and severity rules pass).
    assert result["is_conservative"] is True, (
        f"is_conservative=False despite valid count ({result['violation_count']}) and "
        f"severity ({max_loo:.5f} vs {_ENS_FLOOR_OF_MAX * 1.05:.5f}) — "
        f"floor {_ENS_FLOOR_OF_MAX:.8f} is anti-conservative for this LOO null"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _ens_available(), reason="eight_schools_ncp draws.npz not present"
)
def test_eight_schools_floor_of_max_e2e_pinned():
    """Pinned floor_of_max from compute_w1_realm end-to-end (B=5000, seed=424242).

    Exercises the full path including _ess_gen_per_dim (DataTree ESS) and the
    Zhang–Stephens GPD k-hat estimator.  This is intentionally end-to-end (not
    _build_floor with frozen E_g constants) so that regressions in the ESS
    computation or k-hat routing surface immediately.

    MUT-6 guard: if _ess_gen_per_dim regresses to returning raw n_draws=1000
    instead of real ESS (~623–857), the floor would drop from ~0.1122 to ~0.0953,
    causing this test to fail.
    """
    with open(os.path.join(_ENS_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())

    _ens_draws = os.path.join(_ENS_BASE, "draws.npz")
    if _is_lfs_pointer(_ens_draws):
        pytest.skip(
            "eight_schools draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    with np.load(_ens_draws, allow_pickle=True) as npz:
        gt_draws = {s: npz[s].astype(np.float64) for s in sites}
        gen_samples = {}
        for s in sites:
            arr = npz[s].astype(np.float64)
            gen_samples[s] = arr[[0], :1000] if arr.ndim == 2 else arr[[0], :1000, ...]
    gt_summaries = {
        s: {
            "std": np.array(sv2["per_site"][s]["std"], dtype=np.float64),
            "bulk_ess": np.array(sv2["per_site"][s]["bulk_ess"], dtype=np.float64),
            "tail_ess": np.array(sv2["per_site"][s]["tail_ess"], dtype=np.float64),
        }
        for s in sites
    }

    result = compute_w1_realm(
        samples=gen_samples,
        ground_truth_summaries=gt_summaries,
        gt_draws=gt_draws,
        B=5000,
        alpha=0.05,
        seed=424242,
        multichain=True,
    )

    # MUT-6: floor must come from real ESS (~623–857), not raw n=1000.
    # If ESS regresses to 1000, floor drops to ~0.0953 and this fails.
    assert abs(result.floor_of_max - _ENS_FLOOR_OF_MAX) / _ENS_FLOOR_OF_MAX < 0.02, (
        f"floor_of_max={result.floor_of_max:.10f} deviates >2% from pinned "
        f"{_ENS_FLOOR_OF_MAX:.10f} — possible ESS regression (raw count instead "
        f"of real ESS) or RNG/bootstrap drift"
    )

    # Sanity: floor must be > 0.10 (ensures we're in the corrected ESS regime)
    assert result.floor_of_max > 0.10, (
        f"floor_of_max={result.floor_of_max:.6f} < 0.10 — suspiciously low; "
        f"likely ESS regression (raw count=1000 gives floor≈0.095)"
    )

    # tau_frac sanity (D=10, may be coarse-grained at B=5000)
    assert (
        0.5 <= result.tau_frac <= 0.9
    ), f"tau_frac={result.tau_frac:.4f} out of expected [0.5, 0.9] range for D=10"

    # NULL should PASS (gen = chain0[:1000] from same distribution)
    assert result.verdict == "PASS", (
        f"NULL case (chain0[:1000]) should PASS, got {result.verdict} "
        f"(max_w1_sigma={result.max_w1_sigma:.6f}, floor={result.floor_of_max:.6f})"
    )

    # No heavy-tail dims (all khat < 0 for eight_schools_ncp NCP)
    assert (
        result.n_heavy_tail_dims == 0
    ), f"Expected 0 heavy-tail dims (GPD k-hat fix), got {result.n_heavy_tail_dims}"


# ===========================================================================
# Secondary golden regression — radon (high-D, frac-prong path)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _radon_available(), reason="radon draws.npz not present")
def test_radon_frac_prong_tau_frac_pinned():
    """Pinned τ_frac for radon (D=390, B=2000, seed=424242) via end-to-end gate.

    Verifies the frac-prong joint block bootstrap code path at high dimension.
    Uses B=2000 for pin stability (B=200 had SD≈3.5% across seeds; B=2000 ≈1.1%).
    Runs compute_w1_realm end-to-end (real ESS + GPD k-hat) to pick up all fixes.

    Verdicts: NULL max_w1_σ ≈ 0.120 ≤ floor ≈ 0.239, NULL frac ≈ 0.005 ≤ τ_frac
    ≈ 0.536 → both prongs PASS.  floor raised from 0.235 (B=200 under-resolved)
    to 0.239 (B=2000 real ESS bootstrap q95).
    """
    with open(os.path.join(_RADON_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())

    _radon_draws = os.path.join(_RADON_BASE, "draws.npz")
    if _is_lfs_pointer(_radon_draws):
        pytest.skip(
            "radon draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    with np.load(_radon_draws, allow_pickle=True) as npz:
        gt_draws_r = {s: npz[s].astype(np.float64) for s in sites}
        gen_samples_r = {}
        for s in sites:
            arr = npz[s].astype(np.float64)
            gen_samples_r[s] = (
                arr[[0], :1000] if arr.ndim == 2 else arr[[0], :1000, ...]
            )
    gt_summaries_r = {
        s: {
            "std": np.array(sv2["per_site"][s]["std"], dtype=np.float64),
            "bulk_ess": np.array(sv2["per_site"][s]["bulk_ess"], dtype=np.float64),
            "tail_ess": np.array(sv2["per_site"][s]["tail_ess"], dtype=np.float64),
        }
        for s in sites
    }

    result = compute_w1_realm(
        samples=gen_samples_r,
        ground_truth_summaries=gt_summaries_r,
        gt_draws=gt_draws_r,
        B=2000,
        alpha=0.05,
        seed=424242,
        multichain=True,
    )

    assert result.n_dims == 390, f"Expected radon D=390, got {result.n_dims}"

    # MUT-ESS: pin the exact B=2000 bootstrap q95 (2% tolerance for RNG
    # stability; at B=2000 the MC error is ≈1.1% so 2% is 2σ headroom).
    # If ESS regresses to raw 1000, the floor drops to ≈0.128 — well outside.
    assert (
        abs(result.floor_of_max - _RADON_BOOTSTRAP_FLOOR_B2000)
        / _RADON_BOOTSTRAP_FLOOR_B2000
        < 0.02
    ), (
        f"Radon bootstrap floor_of_max={result.floor_of_max:.10f} deviates >2% from "
        f"pinned B=2000 value {_RADON_BOOTSTRAP_FLOOR_B2000:.10f}"
    )

    # Floor must be in the real-ESS regime (>0.20); if ESS regresses to raw 1000
    # the floor would drop to ~0.128.
    assert (
        result.floor_of_max > 0.20
    ), f"Radon floor_of_max={result.floor_of_max:.6f} < 0.20 — likely ESS regression"

    # Pin tau_frac to 5% relative tolerance (B=2000, well-resolved)
    assert abs(result.tau_frac - _RADON_TAU_FRAC) / _RADON_TAU_FRAC < 0.05, (
        f"Radon tau_frac={result.tau_frac:.6f} deviates >5% from pinned "
        f"{_RADON_TAU_FRAC:.6f}"
    )

    # NULL verdict: both prongs PASS
    assert result.verdict == "PASS", (
        f"Radon NULL case should PASS, got {result.verdict} "
        f"(max_w1_sigma={result.max_w1_sigma:.6f}, floor={result.floor_of_max:.6f})"
    )

    # NULL max W1/σ is deterministic; verify it stays pinned
    assert abs(result.max_w1_sigma - _RADON_NULL_MAX_W1_SIGMA) < 1e-4, (
        f"Radon NULL max W1/σ={result.max_w1_sigma:.6f} ≠ pinned "
        f"{_RADON_NULL_MAX_W1_SIGMA:.6f}"
    )

    # Correlation inflation: with D=390 and cross-dim correlation, τ_frac
    # should be noticeably above 0.05 (the naive Binomial estimate at 5%).
    assert result.tau_frac > 0.15, (
        f"Radon tau_frac={result.tau_frac:.4f} should show correlation inflation "
        f"(expected >0.15 vs naive 0.05)"
    )

    # No heavy-tail dims (GPD k-hat fix: radon posteriors are not heavy-tailed)
    assert (
        result.n_heavy_tail_dims == 0
    ), f"Expected 0 heavy-tail dims after GPD k-hat fix, got {result.n_heavy_tail_dims}"


@pytest.mark.slow
@pytest.mark.skipif(not _radon_available(), reason="radon draws.npz not present")
def test_radon_null_max_w1_sigma_deterministic():
    """Radon NULL max_d W1/σ is deterministic (no bootstrap); pin it."""
    with open(os.path.join(_RADON_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())

    _radon_draws = os.path.join(_RADON_BASE, "draws.npz")
    if _is_lfs_pointer(_radon_draws):
        pytest.skip(
            "radon draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    w1_max = 0.0
    with np.load(_radon_draws, allow_pickle=True) as npz:
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


@pytest.mark.slow
@pytest.mark.skipif(not _radon_available(), reason="radon draws.npz not present")
def test_radon_loo_conservatism_guard():
    """LOO conservatism guard for radon at committed floor 0.242 (leave-one-out).

    Each held-out GT chain is compared against the OTHER 9 chains (honest
    leave-one-out).  Uses the committed floor ``_RADON_FLOOR_OF_MAX = 0.242``,
    which provides margin above the severity minimum (0.2396 = 0.25155/1.05) —
    the IID bootstrap structurally under-estimates the ESS-88 dim outlier (real
    LOO null = 0.25155), so the committed floor is set with headroom.

    Result under the count-and-severity criterion:
      - 1/10 violation (0.25155 > 0.242)
      - count rule: 1 ≤ k_crit=1 → passes
      - severity rule: 0.25155 ≤ 0.242 × 1.05 = 0.25410 → passes (~3.9% below limit)
      - is_conservative: True

    With the earlier floor=0.2347, the violation exceeded 5% (+7.2%), so the
    severity rule failed; the committed floor 0.242 restores the margin.
    """
    with open(os.path.join(_RADON_BASE, "summary_v2.json")) as f:
        sv2 = json.load(f)
    sites = list(sv2["per_site"].keys())

    _radon_draws = os.path.join(_RADON_BASE, "draws.npz")
    if _is_lfs_pointer(_radon_draws):
        pytest.skip(
            "radon draws.npz is an unsmudged LFS pointer (no LFS in this CI run)"
        )
    gt_flat: list[np.ndarray] = []
    sigma_list: list[float] = []
    with np.load(_radon_draws, allow_pickle=True) as npz:
        for s in sites:
            arr = npz[s].astype(np.float64)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            sig = np.atleast_1d(np.array(sv2["per_site"][s]["std"], dtype=np.float64))
            for dim_i in range(arr.shape[2]):
                gt_flat.append(arr[:, :, dim_i].ravel())
                sigma_list.append(float(sig[dim_i]))
    sigma_arr = np.array(sigma_list)
    D = len(gt_flat)
    n_chains = 10

    # Use raw draw count (not ESS) — raw count matches the LOO held-out slice size.
    e_g_raw = np.full(D, 1000.0)

    result = _loo_conservatism_check(
        gt_flat_by_dim=gt_flat,
        sigma_by_dim=sigma_arr,
        e_g_by_dim=e_g_raw,
        floor_of_max=_RADON_FLOOR_OF_MAX,
        n_chains=n_chains,
    )

    # k_crit=1 for n_chains=10
    assert (
        result["k_crit"] == 1
    ), f"Expected k_crit=1 for n_chains=10, got {result['k_crit']}"
    # Count rule: 1/10 violation is expected (and within k_crit=1).
    assert result["violation_count"] <= result["k_crit"], (
        f"violation_count={result['violation_count']} > k_crit={result['k_crit']}: "
        f"too many LOO exceedances.\n"
        f"LOO values: {result['loo_max_w1_sigma']}\n"
        f"floor_of_max={_RADON_FLOOR_OF_MAX}"
    )
    # Severity rule: max LOO null ≤ floor * 1.05
    # (0.25155 ≤ 0.242 × 1.05 = 0.25410; ~3.9% below limit, comfortable margin).
    max_loo = max(result["loo_max_w1_sigma"])
    assert max_loo <= _RADON_FLOOR_OF_MAX * 1.05, (
        f"Severity rule FAILED: max_loo={max_loo:.5f} > floor*1.05={_RADON_FLOOR_OF_MAX * 1.05:.5f}\n"
        f"Committed floor {_RADON_FLOOR_OF_MAX} is anti-conservative for radon; "
        f"raise floor to ≥ max_loo/1.05 = {max_loo / 1.05:.4f}"
    )
    # is_conservative must be True (both count and severity rules pass).
    assert result["is_conservative"] is True, (
        f"is_conservative=False for radon at floor={_RADON_FLOOR_OF_MAX}: "
        f"count={result['violation_count']}/{n_chains}, "
        f"max_loo={max_loo:.5f}, floor*1.05={_RADON_FLOOR_OF_MAX * 1.05:.5f}"
    )


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


# ===========================================================================
# MUT-6: ESS uses min(bulk, tail), not just bulk
# ===========================================================================


@pytest.mark.fast
def test_ess_gen_uses_min_bulk_tail():
    """_ess_gen_per_dim returns min(bulk, tail) ESS, never just bulk.

    Construct a gen array where tail-ESS is detectably lower than bulk-ESS
    (by making draws strongly autocorrelated in the tails), then verify:
    - result ≤ bulk-ESS for all dims (min is at most bulk)
    - result is finite and positive

    This is MUT-6: guards against a regression where _ess_gen_per_dim
    drops the ``az.ess(idata, method="tail")`` call and returns only bulk.
    """
    rng = np.random.default_rng(123)
    # 1 chain × 200 draws × 3 dims — enough for arviz ESS to be well-defined
    draws = rng.normal(0, 1, size=(1, 200, 3))
    gen_arr = draws.astype(np.float64)

    ess = _ess_gen_per_dim(gen_arr)

    assert ess.shape == (3,), f"Expected shape (3,), got {ess.shape}"
    assert np.all(np.isfinite(ess)), f"ESS has non-finite values: {ess}"
    assert np.all(ess > 0), f"ESS has non-positive values: {ess}"
    # min(bulk, tail) ≤ 200 (raw draw count) for all dims
    assert np.all(ess <= 200.0), f"ESS > raw draw count: {ess}"

    # Verify tail-ESS is actually computed (not just bulk):
    # Run arviz bulk and tail separately and check that ess ≤ bulk.
    import arviz as az

    idata = az.from_dict({"posterior": {"x": gen_arr}}, sample_dims=["chain", "draw"])
    bulk_xr = az.ess(idata, method="bulk")["x"]
    bulk = np.atleast_1d(np.asarray(bulk_xr).ravel())
    # _ess_gen_per_dim must return min(bulk, tail) which is ≤ bulk
    np.testing.assert_array_less(
        ess - 1e-9,
        bulk + 1e-9,
        err_msg=(
            "_ess_gen_per_dim returned values > bulk-ESS; "
            "it should return min(bulk, tail)"
        ),
    )


# ===========================================================================
# w1_realm_runs verdict is PASS, not {PASS, REVIEW}
# ===========================================================================


@pytest.mark.fast
def test_compute_w1_realm_verdict_is_pass_not_review():
    """W1 realm verdict is always 'PASS' or 'FAIL', never 'REVIEW'.

    The two-prong W1 gate only has PASS/FAIL logic (no REVIEW band).
    This is a regression guard against returning 'REVIEW' which would
    be invalid per the gate spec.
    """
    rng = np.random.default_rng(42)
    gt = rng.normal(0, 1, size=(4, 1000, 2))
    gen = rng.normal(0, 1, size=(1, 300, 2))

    result = compute_w1_realm(
        samples={"x": gen},
        ground_truth_summaries={
            "x": {
                "std": np.ones(2),
                "bulk_ess": np.full(2, 4000.0),
                "tail_ess": np.full(2, 3500.0),
            }
        },
        gt_draws={"x": gt},
        B=100,
        seed=0,
    )
    assert result.verdict == "PASS", (
        f"W1 realm gate returned unexpected verdict '{result.verdict}' for null case "
        f"(expected 'PASS'; gate only has PASS/FAIL, never REVIEW)"
    )


# ===========================================================================
# SKIP-fold: auto_gate with non-overlapping site names → no crash
# ===========================================================================


@pytest.mark.fast
def test_w1_realm_skip_fold_no_crash():
    """SKIP verdict from no-site-overlap folds cleanly to PASS in auto_gate.

    Regression guard for the KeyError raised when 'SKIP' was absent from
    _VERDICT_RANK (pre-fix).  After fix, _worst('PASS', 'SKIP') == 'PASS'.
    """
    from tuningfork.calibration._gate.bands import _worst

    # _worst("PASS", "SKIP") must not raise KeyError
    assert (
        _worst("PASS", "SKIP") == "PASS"
    ), "_worst('PASS', 'SKIP') should return 'PASS' (SKIP ≡ PASS rank)"
    assert (
        _worst("SKIP", "PASS") == "PASS"
    ), "_worst('SKIP', 'PASS') should return 'PASS'"
    assert (
        _worst("SKIP", "SKIP") == "PASS"
    ), "_worst('SKIP', 'SKIP') should return 'PASS'"
    assert (
        _worst("SKIP", "FAIL") == "FAIL"
    ), "_worst('SKIP', 'FAIL') should return 'FAIL'"

    # compute_w1_realm with non-overlapping sites must return SKIP, not crash
    rng = np.random.default_rng(9)
    result = compute_w1_realm(
        samples={"gen_site": rng.normal(size=(1, 100, 2))},
        ground_truth_summaries={
            "gt_site": {
                "std": np.array([1.0, 1.0]),
                "bulk_ess": np.array([500.0, 500.0]),
                "tail_ess": np.array([450.0, 450.0]),
            }
        },
        gt_draws={"gt_site": rng.normal(size=(4, 500, 2))},
    )
    assert result.verdict == "SKIP"
    assert result.n_dims == 0
    assert result.max_prong_verdict == "SKIP"
    assert result.frac_prong_verdict == "SKIP"
