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
"""Tests for the reference-certification analytic certifier (Path A).

Checks:
- certify_reference_analytic returns (draws, Summaries) with correct shapes.
- Empirical moments match analytic moments within 4-sigma Monte Carlo SE for
  MVN-10 (mean=0, std=1 per dim) and Neal's Funnel (v: mean=0, std=3).
- Raises ValueError for a NUTS-path entry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tuningfork.calibration._summary import Summaries
from tuningfork.calibration.certify_reference_analytic import certify_reference_analytic
from tuningfork.model import MODELS

pytestmark = pytest.mark.slow

MVN_ENTRY = MODELS["mvn_10"]
FUNNEL_ENTRY = MODELS["neals_funnel"]
NUTS_ENTRY = MODELS["eight_schools_ncp"]

N = 50_000  # large enough for tight 4-sigma bounds


class TestCertifyAnalyticInterface:
    """Basic interface and type checks."""

    def test_returns_tuple(self) -> None:
        key = jax.random.key(0)
        result = certify_reference_analytic(MVN_ENTRY, 100, key)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_draws_dict(self) -> None:
        key = jax.random.key(1)
        draws, _ = certify_reference_analytic(MVN_ENTRY, 100, key)
        assert isinstance(draws, dict)
        assert "x" in draws

    def test_returns_summaries_instance(self) -> None:
        key = jax.random.key(2)
        _, summaries = certify_reference_analytic(MVN_ENTRY, 100, key)
        assert isinstance(summaries, Summaries)

    def test_draws_shape_mvn(self) -> None:
        key = jax.random.key(3)
        draws, _ = certify_reference_analytic(MVN_ENTRY, 200, key)
        assert draws["x"].shape == (200, 10)

    def test_summaries_n_samples(self) -> None:
        key = jax.random.key(4)
        _, summaries = certify_reference_analytic(MVN_ENTRY, 300, key)
        assert summaries.n_samples == 300

    def test_raises_for_nuts_entry(self) -> None:
        key = jax.random.key(5)
        with pytest.raises(ValueError, match="NUTS path"):
            certify_reference_analytic(NUTS_ENTRY, 100, key)


class TestMVN10Moments:
    """Empirical moments of MVN-10 match analytic (mean=0, std=1) at 4-sigma."""

    def test_mean_near_zero(self) -> None:
        key = jax.random.key(42)
        draws, _ = certify_reference_analytic(MVN_ENTRY, N, key)
        x = np.asarray(draws["x"])  # (N, 10)
        # MC SE of mean estimator = std / sqrt(n) = 1 / sqrt(N)
        tol = 4.0 / np.sqrt(N)
        empirical_mean = x.mean(axis=0)
        np.testing.assert_allclose(
            empirical_mean,
            np.zeros(10),
            atol=tol,
            err_msg="MVN-10 empirical mean not within 4-sigma of 0",
        )

    def test_std_near_one(self) -> None:
        key = jax.random.key(43)
        draws, _ = certify_reference_analytic(MVN_ENTRY, N, key)
        x = np.asarray(draws["x"])  # (N, 10)
        # MC SE of std estimator ≈ std / sqrt(2(n-1)) ≈ 1 / sqrt(2N)
        tol = 4.0 / np.sqrt(2 * N)
        empirical_std = x.std(axis=0)
        np.testing.assert_allclose(
            empirical_std,
            np.ones(10),
            atol=tol,
            err_msg="MVN-10 empirical std not within 4-sigma of 1",
        )

    def test_summaries_mean_near_zero(self) -> None:
        key = jax.random.key(44)
        _, summaries = certify_reference_analytic(MVN_ENTRY, N, key)
        tol = 4.0 / np.sqrt(N)
        np.testing.assert_allclose(
            np.asarray(summaries.mean["x"]),
            np.zeros(10),
            atol=tol,
            err_msg="Summaries.mean['x'] not within 4-sigma of 0",
        )

    def test_summaries_std_near_one(self) -> None:
        key = jax.random.key(45)
        _, summaries = certify_reference_analytic(MVN_ENTRY, N, key)
        tol = 4.0 / np.sqrt(2 * N)
        np.testing.assert_allclose(
            np.asarray(summaries.std["x"]),
            np.ones(10),
            atol=tol,
            err_msg="Summaries.std['x'] not within 4-sigma of 1",
        )


class TestNealsFunnelMoments:
    """Empirical moments of Neal's Funnel match analytic values at 4-sigma.

    Analytic marginals:
        v ~ Normal(0, 3)       → mean=0, std=3
        theta_i | v ~ Normal(0, exp(v/2))
        Marginal theta_i has very heavy tails (infinite variance),
        so we check v only with tight bounds; theta tolerances are generous.
    """

    def test_v_mean_near_zero(self) -> None:
        key = jax.random.key(100)
        draws, _ = certify_reference_analytic(FUNNEL_ENTRY, N, key)
        v = np.asarray(draws["v"])  # (N,)
        # std(v) = 3, so MC SE = 3 / sqrt(N)
        tol = 4.0 * 3.0 / np.sqrt(N)
        assert (
            abs(v.mean()) < tol
        ), f"Funnel v mean={v.mean():.6f} not within 4-sigma tol={tol:.6f}"

    def test_v_std_near_three(self) -> None:
        key = jax.random.key(101)
        draws, _ = certify_reference_analytic(FUNNEL_ENTRY, N, key)
        v = np.asarray(draws["v"])  # (N,)
        # MC SE of std estimator ≈ 3 / sqrt(2N)
        tol = 4.0 * 3.0 / np.sqrt(2 * N)
        assert (
            abs(v.std() - 3.0) < tol
        ), f"Funnel v std={v.std():.6f} not within 4-sigma of 3 (tol={tol:.6f})"

    def test_theta_mean_near_zero(self) -> None:
        # Generous tolerance: theta marginal std is large (heavy tails).
        # Use 5 SE per-dim with empirical std as estimate.
        # np.testing.assert_allclose requires scalar atol, so check per dim.
        key = jax.random.key(102)
        draws, _ = certify_reference_analytic(FUNNEL_ENTRY, N, key)
        theta = np.asarray(draws["theta"])  # (N, 9)
        empirical_std = theta.std(axis=0)
        empirical_mean = theta.mean(axis=0)
        per_dim_tol = 5.0 * empirical_std / np.sqrt(N)
        # Check each dim individually so atol is always scalar
        for i in range(theta.shape[1]):
            assert abs(empirical_mean[i]) < per_dim_tol[i], (
                f"Funnel theta[{i}] mean={empirical_mean[i]:.6f} not within "
                f"5-sigma SE tol={per_dim_tol[i]:.6f}"
            )

    def test_summaries_v_mean(self) -> None:
        key = jax.random.key(103)
        _, summaries = certify_reference_analytic(FUNNEL_ENTRY, N, key)
        v_mean = float(jnp.asarray(summaries.mean["v"]))
        tol = 4.0 * 3.0 / np.sqrt(N)
        assert (
            abs(v_mean) < tol
        ), f"Summaries v mean={v_mean:.6f} not within tolerance {tol:.6f}"

    def test_draws_shape_funnel(self) -> None:
        key = jax.random.key(104)
        draws, _ = certify_reference_analytic(FUNNEL_ENTRY, 500, key)
        assert draws["v"].shape == (500,)
        assert draws["theta"].shape == (500, 9)
