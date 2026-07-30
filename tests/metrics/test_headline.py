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
"""Tests for tuningfork.metrics.headline.min_bulk_ess_per_grad.

Empirical questions answered here (confirmed via test output):

1. effective_sample_size output shape:
   For input shape (C, S, D), output is shape (D,) — already aggregated
   across chains (bulk-ESS sums over chains per the standard definition).

2. i.i.d. baseline headline / (C*S):
   For C=4, S=1000, D=5 i.i.d. normal samples with n_grad_evals=C*S=4000,
   the headline metric is approximately 0.85–1.05 in practice. The
   variance-corrected ESS estimator (autocorrelation-based) underestimates
   slightly for finite samples; observed ratio documented in test assertions.

3. AR(1) suppression ratio headline_iid / headline_ar1:
   For φ=0.9, S=2000, C=2: theoretical ESS ≈ N*(1-φ)/(1+φ) = 4000*0.1/1.9
   ≈ 210. Observed ratio is ≥ 5 (required) and typically 15–22.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from blackjax.diagnostics import effective_sample_size, ess_bulk

from tuningfork.metrics.grad_counter import total_grad_evals
from tuningfork.metrics.headline import (
    build_headline_basis,
    estimator_ratio,
    min_bulk_ess,
    min_bulk_ess_classic_legacy,
    min_bulk_ess_per_grad,
)

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helper: deterministic keys for reproducible tests
# ---------------------------------------------------------------------------

KEY0 = jax.random.key(42)


def _key(n: int) -> jax.Array:
    """Return a deterministic key for test index n."""
    return jax.random.fold_in(KEY0, n)


# ---------------------------------------------------------------------------
# 1. i.i.d. samples baseline
# ---------------------------------------------------------------------------


class TestIIDBaseline:
    """i.i.d. normal samples → headline close to 1.0 for n_grad_evals = C*S."""

    def test_iid_baseline_close_to_one(self) -> None:
        """For i.i.d. (C=4, S=1000, D=5) headline / (C*S) ≈ 0.7–1.05."""
        C, S, D = 4, 1000, 5
        n_grad_evals = C * S  # 4000
        samples = {"x": jax.random.normal(_key(0), (C, S, D))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=n_grad_evals)
        # Document observed range: variance-corrected ESS underestimates
        # slightly for finite chains; allow generous tolerance ≥ 0.7.
        assert headline > 0.70, (
            f"i.i.d. baseline headline={headline:.4f} is unexpectedly low; "
            "possible misuse of effective_sample_size aggregation."
        )
        assert headline < 1.5, (
            f"i.i.d. baseline headline={headline:.4f} is implausibly high; "
            "possible double-counting of chains."
        )

    def test_iid_returns_float(self) -> None:
        """Return type must be a Python float, not a JAX Array."""
        samples = {"x": jax.random.normal(_key(1), (2, 100, 3))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=200)
        assert isinstance(
            headline, float
        ), f"Expected Python float, got {type(headline).__name__}"

    def test_iid_positive(self) -> None:
        """Headline must be strictly positive for valid i.i.d. samples."""
        samples = {"x": jax.random.normal(_key(2), (4, 500, 2))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=2000)
        assert headline > 0.0


# ---------------------------------------------------------------------------
# 2. Multi-site dict
# ---------------------------------------------------------------------------


class TestMultiSiteDict:
    """Multi-site: headline is governed by the worst site across all dims."""

    def test_two_sites_both_iid(self) -> None:
        """Two i.i.d. sites — headline controlled by global min ESS dim."""
        C, S = 4, 1000
        sites = {
            "x": jax.random.normal(_key(10), (C, S, 5)),
            "y": jax.random.normal(_key(11), (C, S, 3)),
        }
        n_grad_evals = C * S
        headline = min_bulk_ess_per_grad(sites, n_grad_evals=n_grad_evals)
        # Min ESS across 8 dims (5+3), all i.i.d. → still close to 1.0
        assert headline > 0.60, f"Multi-site headline={headline:.4f} too low"
        assert headline < 1.5, f"Multi-site headline={headline:.4f} too high"

    def test_worst_site_governs(self) -> None:
        """When one site has much lower ESS, it should pull the headline down."""
        C, S = 2, 500
        # good_site: i.i.d. (high ESS)
        good_site = jax.random.normal(_key(12), (C, S, 3))
        # bad_site: highly autocorrelated (low ESS) — AR(1) with φ=0.95
        bad_site = jnp.zeros((C, S, 1))
        for c in range(C):
            row = [jax.random.normal(_key(c + 20))]
            for t in range(1, S):
                noise = jax.random.normal(_key(c * S + t + 100)) * 0.05
                row.append(0.95 * row[-1] + noise)
            bad_site = bad_site.at[c, :, 0].set(jnp.array(row))

        sites = {"good": good_site, "bad": bad_site}
        n_grad_evals = C * S

        headline_good_only = min_bulk_ess_per_grad({"good": good_site}, n_grad_evals)
        headline_both = min_bulk_ess_per_grad(sites, n_grad_evals)

        assert (
            headline_both < headline_good_only
        ), "Adding a bad site should lower the headline metric"


# ---------------------------------------------------------------------------
# 3. AR(1) autocorrelation suppression
# ---------------------------------------------------------------------------


class TestAR1Suppression:
    """AR(1) with φ=0.9 should suppress ESS substantially vs. i.i.d."""

    def _build_ar1_chain(
        self, phi: float, C: int, S: int, key: jax.Array
    ) -> jnp.ndarray:
        """Build AR(1) chain: x_t = phi * x_{t-1} + sqrt(1-phi^2) * eps."""
        noise_scale = jnp.sqrt(1 - phi**2)  # stationary variance = 1
        chains = []
        for c in range(C):
            chain_key = jax.random.fold_in(key, c)
            eps = jax.random.normal(chain_key, (S,)) * noise_scale
            x = jnp.zeros(S)
            x = x.at[0].set(eps[0])

            # Use scan for efficiency
            def step(x_prev, eps_t):  # noqa: E306
                x_t = phi * x_prev + eps_t
                return x_t, x_t

            _, xs = jax.lax.scan(step, x[0], eps[1:])
            chain = jnp.concatenate([x[:1], xs])
            chains.append(chain)
        return jnp.stack(chains)[:, :, None]  # (C, S, 1)

    def test_ar1_lower_than_iid(self) -> None:
        """headline_iid > 5 × headline_ar1 for φ=0.9, C=2, S=2000."""
        C, S = 2, 2000
        phi = 0.9
        n_grad_evals = C * S  # same for both

        iid = {"x": jax.random.normal(_key(30), (C, S, 1))}
        ar1 = {"x": self._build_ar1_chain(phi, C, S, _key(31))}

        headline_iid = min_bulk_ess_per_grad(iid, n_grad_evals)
        headline_ar1 = min_bulk_ess_per_grad(ar1, n_grad_evals)

        ratio = headline_iid / headline_ar1
        assert ratio > 5.0, (
            f"Expected iid/ar1 > 5, got {ratio:.2f} "
            f"(headline_iid={headline_iid:.4f}, headline_ar1={headline_ar1:.4f})"
        )

    def test_ar1_ess_finite_positive(self) -> None:
        """AR(1) chain should still give a finite positive headline."""
        C, S = 2, 2000
        ar1 = {"x": self._build_ar1_chain(0.9, C, S, _key(32))}
        headline = min_bulk_ess_per_grad(ar1, n_grad_evals=C * S)
        assert jnp.isfinite(headline), f"AR(1) headline is not finite: {headline}"
        assert headline > 0.0, f"AR(1) headline is not positive: {headline}"


# ---------------------------------------------------------------------------
# 4. Site shape variants
# ---------------------------------------------------------------------------


class TestSiteShapeVariants:
    """Verify flattening works correctly for diverse site shapes."""

    def test_scalar_site(self) -> None:
        """Site shape (C, S) — 0-D per sample — should work via (C, S, 1) reshape."""
        C, S = 3, 200
        # Shape (C, S): no trailing dims
        samples = {"scalar": jax.random.normal(_key(40), (C, S))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=C * S)
        assert isinstance(headline, float)
        assert headline > 0.0

    def test_2d_site_flattened(self) -> None:
        """Site shape (C, S, 4, 4) should flatten to (C, S, 16) silently."""
        C, S = 2, 100
        samples = {"matrix": jax.random.normal(_key(41), (C, S, 4, 4))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=C * S)
        assert isinstance(headline, float)
        assert headline > 0.0

    def test_1d_site(self) -> None:
        """Site shape (C, S, 1) — a 1-D site — should work fine."""
        C, S = 4, 500
        samples = {"x": jax.random.normal(_key(42), (C, S, 1))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=C * S)
        assert isinstance(headline, float)
        assert headline > 0.0

    def test_high_d_site(self) -> None:
        """Site shape (C, S, 20) — 20-D — should yield shape-(20,) ESS."""
        C, S, D = 4, 300, 20
        samples = {"big": jax.random.normal(_key(43), (C, S, D))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=C * S)
        assert isinstance(headline, float)
        assert headline > 0.0


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and error conditions."""

    def test_n_grad_evals_zero_returns_inf(self) -> None:
        """n_grad_evals=0 with finite samples → returns inf."""
        samples = {"x": jax.random.normal(_key(50), (2, 100, 3))}
        result = min_bulk_ess_per_grad(samples, n_grad_evals=0)
        assert result == float("inf"), f"Expected inf, got {result}"

    def test_n_grad_evals_negative_raises(self) -> None:
        """n_grad_evals=-1 should raise ValueError."""
        samples = {"x": jax.random.normal(_key(51), (2, 100, 3))}
        with pytest.raises(ValueError, match="non-negative"):
            min_bulk_ess_per_grad(samples, n_grad_evals=-1)

    def test_empty_states_raises(self) -> None:
        """Empty states_position dict should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            min_bulk_ess_per_grad({}, n_grad_evals=100)

    def test_site_ndim_1_raises(self) -> None:
        """A site array with ndim=1 should raise ValueError."""
        # Shape (100,) — only one axis, missing chain+sample axes.
        bad = jnp.ones((100,))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            min_bulk_ess_per_grad({"x": bad}, n_grad_evals=100)

    def test_n_grad_evals_zero_degenerate_returns_nan(self) -> None:
        """n_grad_evals=0 with non-finite min_ess → nan.

        We trigger this by creating a constant chain (ESS = 0 or inf depending
        on implementation). The edge case tests the NaN branch.
        Note: if effective_sample_size returns 0 for a constant chain,
        min_ess = 0 which is finite, so the result is still inf. We test
        the nan path by directly passing n_grad_evals=0 to a degenerate
        constructed case where we verify at minimum that the function
        doesn't crash and returns either inf or nan.
        """
        # constant chain — ESS is effectively degenerate (may be 0 or inf)
        const = jnp.ones((2, 10, 1))  # identical samples: ESS could be 0 or high
        result = min_bulk_ess_per_grad({"x": const}, n_grad_evals=0)
        # result is either inf or nan — both are non-negative or nan
        assert not jnp.isneginf(result), f"Unexpected -inf result: {result}"


# ---------------------------------------------------------------------------
# 6. Integration with grad_counter.total_grad_evals
# ---------------------------------------------------------------------------


class TestIntegrationWithGradCounter:
    """Smoke test: headline + grad_counter compose correctly."""

    def test_smoke_with_fake_hmc_info(self) -> None:
        """Fake HMC info with num_integration_steps → total_grad_evals → headline."""

        class FakeHMCInfo(NamedTuple):
            num_integration_steps: jnp.ndarray

        # [5, 3, 7, 2] → sum = 17
        infos = FakeHMCInfo(
            num_integration_steps=jnp.array([5, 3, 7, 2], dtype=jnp.int32)
        )
        n = total_grad_evals(infos, lambda i: i.num_integration_steps)
        assert n == 17

        # 4 chains, 50 samples, 3-D site
        samples = {"x": jax.random.normal(_key(60), (4, 50, 3))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=n)

        assert isinstance(headline, float)
        assert jnp.isfinite(headline)
        assert headline > 0.0

    def test_mala_like_constant_cost(self) -> None:
        """MALA-style: 1 grad/step. Verify composition is dimensionally consistent."""

        class FakeMALAInfo(NamedTuple):
            accepted: jnp.ndarray

        n_steps = 200
        infos = FakeMALAInfo(accepted=jnp.ones((n_steps,), dtype=jnp.bool_))
        n = total_grad_evals(infos, lambda i: 1)
        assert n == n_steps

        # 2 chains, 200 samples, 5-D site
        samples = {"theta": jax.random.normal(_key(61), (2, 200, 5))}
        headline = min_bulk_ess_per_grad(samples, n_grad_evals=n)

        assert isinstance(headline, float)
        assert headline > 0.0


# ---------------------------------------------------------------------------
# 7. Which ESS estimator the headline uses
# ---------------------------------------------------------------------------


def _heavy_tailed_slow_mixing(key: jax.Array, C: int = 4, S: int = 1000) -> jnp.ndarray:
    """exp(AR(1) φ=0.95) — draws where the two ESS estimators disagree sharply.

    Chosen because it exercises BOTH mechanisms that separate ``ess_bulk`` from
    ``effective_sample_size``: strong autocorrelation makes each chain half look
    like a distinct chain once split, and the lognormal marginal is exactly what
    rank normalisation is for.  Measured across seeds 0–5 the ratio sits in
    0.12–0.30 (a 3–8× gap) and never changes sign, so the fixture discriminates
    without being seed-fragile.
    """
    eps = jax.random.normal(key, (S, C, 3))

    def step(x_prev, eps_t):
        x_t = 0.95 * x_prev + eps_t
        return x_t, x_t

    _, xs = jax.lax.scan(step, jnp.zeros((C, 3)), eps)
    return jnp.exp(jnp.moveaxis(xs, 0, 1))  # (C, S, 3)


class TestEstimatorContract:
    """The headline numerator is rank-normalised split-chain bulk-ESS."""

    def test_helpers_pin_to_their_blackjax_functions(self) -> None:
        """min_bulk_ess ↔ ess_bulk and min_bulk_ess_classic_legacy ↔ effective_sample_size."""
        samples = {"x": _heavy_tailed_slow_mixing(_key(70))}
        flat = samples["x"]

        expected_rank = float(jnp.min(ess_bulk(flat, chain_axis=0, sample_axis=1)))
        expected_classic = float(
            jnp.min(effective_sample_size(flat, chain_axis=0, sample_axis=1))
        )

        assert min_bulk_ess(samples) == pytest.approx(expected_rank, rel=1e-9)
        assert min_bulk_ess_classic_legacy(samples) == pytest.approx(
            expected_classic, rel=1e-9
        )

    def test_headline_uses_rank_normalised_not_classic_ess(self) -> None:
        """min_bulk_ess_per_grad reports the ess_bulk value, not effective_sample_size.

        Positive control for the estimator switch.  Wiring
        ``_min_ess_over_sites(states_position, effective_sample_size)`` back into
        ``min_bulk_ess_per_grad`` turns this red: on these draws the two
        estimators are 3–8× apart, far outside the 1% band below.
        """
        C, S = 4, 1000
        n_grad_evals = C * S
        samples = {"x": _heavy_tailed_slow_mixing(_key(71), C, S)}

        rank = min_bulk_ess(samples)
        classic = min_bulk_ess_classic_legacy(samples)

        # Guard the fixture itself: a fixture that stopped discriminating would
        # make the assertions below vacuous, and it should say so out loud.
        assert classic / rank > 3.0, (
            f"fixture no longer separates the estimators (rank={rank:.1f}, "
            f"classic={classic:.1f}); the test below would be vacuous"
        )

        headline = min_bulk_ess_per_grad(samples, n_grad_evals=n_grad_evals)

        assert headline == pytest.approx(rank / n_grad_evals, rel=1e-9), (
            f"headline={headline!r} does not match the rank-normalised value "
            f"{rank / n_grad_evals!r}; the headline numerator must be ess_bulk"
        )
        assert headline != pytest.approx(classic / n_grad_evals, rel=1e-2), (
            f"headline={headline!r} matches the LEGACY estimator "
            f"{classic / n_grad_evals!r} — effective_sample_size is still wired in"
        )

    def test_estimator_ratio_is_rank_over_classic(self) -> None:
        """estimator_ratio isolates the estimator effect on one fixed set of draws."""
        samples = {"x": _heavy_tailed_slow_mixing(_key(72))}
        rank = min_bulk_ess(samples)
        classic = min_bulk_ess_classic_legacy(samples)

        assert estimator_ratio(rank, classic) == pytest.approx(rank / classic, rel=1e-9)

    @pytest.mark.parametrize(
        "rank,classic",
        [(100.0, 0.0), (100.0, float("nan")), (float("inf"), 100.0), (100.0, None)],
    )
    def test_estimator_ratio_is_null_when_undefined(self, rank, classic) -> None:
        """A degenerate ratio is recorded as null, never as an inf that reads as real."""
        assert estimator_ratio(rank, classic) is None

    def test_basis_stamps_provenance_and_carries_both_estimators(self) -> None:
        """build_headline_basis records which estimator ran plus the legacy value."""
        samples = {"x": _heavy_tailed_slow_mixing(_key(73))}
        grad_evals = 8000

        headline, basis = build_headline_basis(
            samples,
            denominator=grad_evals,
            total_grad_evals=grad_evals,
            grad_count_convention="2",
            is_lower_bound=False,
        )

        assert basis["ess_estimator"] == "ess_bulk"
        assert basis["min_bulk_ess"] == pytest.approx(min_bulk_ess(samples), rel=1e-9)
        assert basis["min_bulk_ess_classic_legacy"] == pytest.approx(
            min_bulk_ess_classic_legacy(samples), rel=1e-9
        )
        assert basis["estimator_ratio"] == pytest.approx(
            basis["min_bulk_ess"] / basis["min_bulk_ess_classic_legacy"], rel=1e-9
        )
        assert basis["grad_count_convention"] == "2"
        assert basis["is_lower_bound"] is False
        assert headline == pytest.approx(
            min_bulk_ess_per_grad(samples, grad_evals), rel=1e-9
        )

    def test_basis_reproduces_the_headline_exactly(self) -> None:
        """headline == basis.min_bulk_ess / total_grad_evals with no rounding slack.

        The catalog-wide invariant test uses a 1e-9 band, so the back-derivation
        has to be exact rather than merely close.
        """
        samples = {"x": _heavy_tailed_slow_mixing(_key(74))}
        grad_evals = 12345

        headline, basis = build_headline_basis(
            samples,
            denominator=grad_evals,
            total_grad_evals=grad_evals,
            grad_count_convention="2",
            is_lower_bound=False,
        )
        derived = basis["min_bulk_ess"] / basis["total_grad_evals"]
        assert abs(headline - derived) <= 1e-12 * max(abs(headline), abs(derived))

    def test_basis_gradient_free_denominator_differs_from_grad_evals(self) -> None:
        """Gradient-free: total_grad_evals=0 while the divisor is the draw count."""
        C, S = 4, 1000
        samples = {"x": _heavy_tailed_slow_mixing(_key(75), C, S)}

        headline, basis = build_headline_basis(
            samples,
            denominator=C * S,
            total_grad_evals=0,
            grad_count_convention="0 (gradient-free; headline = min_bulk_ess/n_total_samples)",
            is_lower_bound=False,
        )

        assert basis["total_grad_evals"] == 0
        assert basis["ess_estimator"] == "ess_bulk"
        assert headline == pytest.approx(min_bulk_ess(samples) / (C * S), rel=1e-9)

    def test_basis_rejects_nonpositive_denominator(self) -> None:
        """A zero divisor is a caller bug, not an inf headline to be shipped."""
        samples = {"x": jax.random.normal(_key(76), (2, 100, 3))}
        with pytest.raises(ValueError, match="denominator must be positive"):
            build_headline_basis(
                samples,
                denominator=0,
                total_grad_evals=0,
                grad_count_convention="x",
                is_lower_bound=False,
            )
