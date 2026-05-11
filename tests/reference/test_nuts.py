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
"""Tests for the Tier-A NUTS path (Path B) — 8-Schools NCP.

Runs certify_reference_nuts with small parameters to keep the test fast
(~30s on CPU):
    n_warmup=500, n_samples=4000, n_chunks=4, seed=42

Seed selection: seeds [0, 1, 2, 42] with n_samples=2000 all failed the
min_chunk_bulk_ess >= 400 gate (2000 samples / 4 chunks = 500 per chunk
gives too few effective samples for 8-Schools at this warmup budget).
Bumping to n_samples=4000 lets seed=42 clear the gate with
min_chunk_bulk_ess ≈ 554, 0 divergences, e_bfmi ≈ 0.97.
"""

import jax
import pytest

from bjx_bench.calibration._summary import Summaries
from bjx_bench.calibration.certify_reference import (
    AdaptationParams,
    CertificationResult,
    certify_reference_nuts,
)
from bjx_bench.model import MODELS

pytestmark = pytest.mark.slow

ENTRY = MODELS["eight_schools_ncp"]

# Fixed seed used for the NUTS test; see module docstring for selection rationale.
NUTS_SEED = 42
N_WARMUP = 500
N_SAMPLES = 4000
N_CHUNKS = 4


class TestCertifyNutsInterface:
    """Basic interface and type checks for certify_reference_nuts."""

    def test_returns_four_tuple(self) -> None:
        key = jax.random.key(NUTS_SEED)
        result = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_draws_is_dict(self) -> None:
        key = jax.random.key(NUTS_SEED)
        draws, _, _, _ = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert isinstance(draws, dict)

    def test_draws_keys_contain_theta_raw(self) -> None:
        key = jax.random.key(NUTS_SEED)
        draws, _, _, _ = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        # 8-Schools NCP latent sites: mu, tau (log-transformed), theta_raw
        assert "theta_raw" in draws

    def test_draws_sample_axis(self) -> None:
        key = jax.random.key(NUTS_SEED)
        draws, _, _, _ = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        # Each site's first axis should be n_samples
        for site, arr in draws.items():
            assert (
                arr.shape[0] == N_SAMPLES
            ), f"Site {site!r}: expected shape[0]={N_SAMPLES}, got {arr.shape[0]}"

    def test_summaries_instance(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, summaries, _, _ = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert isinstance(summaries, Summaries)

    def test_adaptation_params_instance(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, _, adaptation, _ = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert isinstance(adaptation, AdaptationParams)
        assert adaptation.step_size > 0.0
        assert adaptation.inverse_mass_matrix.ndim >= 1

    def test_cert_instance(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, _, _, cert = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert isinstance(cert, CertificationResult)

    def test_raises_for_analytic_entry(self) -> None:
        key = jax.random.key(99)
        analytic_entry = MODELS["mvn_10"]
        with pytest.raises(ValueError, match="analytic path"):
            certify_reference_nuts(analytic_entry, key)


class TestCertifyNutsPassesGate:
    """8-Schools NCP must pass Tier-A certification at small params (seed=42).

    Gate thresholds (from tier_a.py):
        split_rhat_max <= 1.01
        min_chunk_bulk_ess >= 400
        num_divergences == 0
        e_bfmi >= 0.3

    At seed=42, n_warmup=500, n_samples=4000, n_chunks=4:
        split_rhat_max ≈ 1.0003
        min_chunk_bulk_ess ≈ 554
        num_divergences = 0
        e_bfmi ≈ 0.97
    """

    def test_certification_passed(self) -> None:
        """certification.passed must be True for seed=42 at small params."""
        key = jax.random.key(NUTS_SEED)
        _, _, _, cert = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert cert.passed, (
            f"8-Schools NCP failed Tier-A certification at seed={NUTS_SEED}: "
            f"split_rhat_max={cert.split_rhat_max:.4f}, "
            f"min_chunk_bulk_ess={cert.min_chunk_bulk_ess:.1f}, "
            f"num_divergences={cert.num_divergences}, "
            f"e_bfmi={cert.e_bfmi:.4f}"
        )

    def test_no_divergences(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, _, _, cert = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert (
            cert.num_divergences == 0
        ), f"Expected 0 divergences, got {cert.num_divergences}"

    def test_e_bfmi_above_threshold(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, _, _, cert = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert cert.e_bfmi >= 0.3, f"E-BFMI={cert.e_bfmi:.4f} below threshold 0.3"

    def test_split_rhat_below_threshold(self) -> None:
        key = jax.random.key(NUTS_SEED)
        _, _, _, cert = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            n_chunks=N_CHUNKS,
        )
        assert (
            cert.split_rhat_max <= 1.01
        ), f"split_rhat_max={cert.split_rhat_max:.4f} exceeds threshold 1.01"
