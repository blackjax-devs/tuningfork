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
"""Tests for the long-NUTS reference-certification path (Path B).

Design rationale (refactored 2026-05-17 — see PR #5):

The original test suite ran 15 independent ``certify_reference_nuts`` calls on
8-Schools NCP with downscaled config (n=4000) and asserted both structural
return shapes AND the gate verdict on each. Two problems:

1. The gate threshold (``min_chunk_bulk_ess >= 400``) is *absolute*, so to
   assert PASS verdict on the downscaled config we needed a sample size large
   enough to drive ESS over 400. That made each test slow (~14 s wall on
   16-core), and the seed=42 / n=4000 baseline was fragile against PRNG-path
   variation across runners — GH-Actions run 25983264151 returned
   min_chunk_bulk_ess=353.8 on the same seed (local: ~554, no chain pathology,
   pure platform numerics).

2. Most tests don't actually care about the verdict — they assert ``isinstance``
   and ``arr.shape[0] == N_SAMPLES``. The cert function raises
   ``CertificationError`` on FAIL, so the structural assertions never ran when
   the gate didn't pass on a given runner.

The fix splits concerns:

* ``TestCertifyNutsInterface`` / ``TestChainStats`` use a *module-scoped fixture*
  that runs ``certify_reference_nuts`` ONCE on 8-Schools NCP at a tiny config
  (n=500, n_warmup=200) and yields the verdict result regardless of PASS/FAIL.
  The fixture catches ``CertificationError`` and rebuilds the structural payload
  from the exception's attached ``cert`` / ``chain_stats`` / ``adaptation`` /
  ``draws`` fields. Tests assert structure only; they are PRNG-stable.

* ``TestCertifyNutsGateLogic`` exercises ``compute_certification_verdict``
  directly with synthetic inputs (no JAX trace, no NUTS run, runs in ms).
  This is where we assert PASS-on-clean, FAIL-on-high-rhat,
  FAIL-on-low-ess, FAIL-on-too-many-divergences, FAIL-on-low-ebfmi.

The original ``TestCertifyNutsPassesGate`` (4 tests that asserted real-cert
verdict on 8-Schools at n=4000) is removed. The verdict-PASS check on real data
now lives in the production recipe-generation pipeline, not in unit tests —
per CLAUDE.md "do NOT brute-force the gate by inflating n_samples" + the
2026-05-17 user direction "tests should not run real certification".
"""

import dataclasses
from typing import Any

import jax
import pytest

from tuningfork.calibration._summary import Summaries
from tuningfork.calibration.certify_reference import (
    AdaptationParams,
    CertificationError,
    CertificationResult,
    certify_reference_nuts,
    compute_certification_verdict,
)
from tuningfork.model import MODELS

ENTRY = MODELS["eight_schools_ncp"]

# Tiny config for the structural smoke fixture. n=500 / n_warmup=200 is far below
# what the cert gate requires (min_chunk_bulk_ess >= 400 needs ~2800+ samples
# split across 4 chunks) — the fixture EXPECTS the gate to fail and inspects
# the structural payload regardless. Pure-logic gate tests live below in
# TestCertifyNutsGateLogic.
SMOKE_SEED = 42
SMOKE_N_WARMUP = 200
SMOKE_N_SAMPLES = 500
SMOKE_N_CHUNKS = 4


@pytest.fixture(scope="module")
def smoke_cert_result():
    """Run certify_reference_nuts ONCE at a tiny config; return its 5-tuple
    payload regardless of PASS/FAIL verdict.

    When the gate FAILs (expected at this sample size), the function raises
    CertificationError — we catch it and reconstruct the structural payload
    from the exception's attached fields. The Summaries object is not on the
    exception, so we substitute None and the one test that wants it skips
    accordingly.

    Returns
    -------
    tuple
        ``(draws, summaries_or_none, adaptation, cert, chain_stats)``. The
        Summaries entry is ``None`` iff the gate raised (no summaries are
        computed on the failure path).
    """
    key = jax.random.key(SMOKE_SEED)
    try:
        draws, summaries, adaptation, cert, chain_stats = certify_reference_nuts(
            ENTRY,
            key,
            n_warmup=SMOKE_N_WARMUP,
            n_samples=SMOKE_N_SAMPLES,
            n_chunks=SMOKE_N_CHUNKS,
        )
        return (draws, summaries, adaptation, cert, chain_stats)
    except CertificationError as exc:
        # Expected on the tiny config — gate threshold absolute, sample size below.
        # The exception carries the structural payload for downstream inspection.
        return (exc.draws, None, exc.adaptation, exc.cert, exc.chain_stats)


@pytest.mark.slow
class TestCertifyNutsInterface:
    """Structural return-shape checks — assertions independent of gate verdict."""

    def test_returns_five_payload(self, smoke_cert_result) -> None:
        assert isinstance(smoke_cert_result, tuple)
        assert len(smoke_cert_result) == 5

    def test_draws_is_dict(self, smoke_cert_result) -> None:
        draws, _, _, _, _ = smoke_cert_result
        assert isinstance(draws, dict)

    def test_draws_keys_contain_theta_raw(self, smoke_cert_result) -> None:
        draws, _, _, _, _ = smoke_cert_result
        # 8-Schools NCP latent sites: mu, tau (log-transformed), theta_raw
        assert "theta_raw" in draws

    def test_draws_sample_axis(self, smoke_cert_result) -> None:
        draws, _, _, _, _ = smoke_cert_result
        for site, arr in draws.items():
            assert (
                arr.shape[0] == SMOKE_N_SAMPLES
            ), f"Site {site!r}: expected shape[0]={SMOKE_N_SAMPLES}, got {arr.shape[0]}"

    def test_summaries_instance(self, smoke_cert_result) -> None:
        _, summaries, _, _, _ = smoke_cert_result
        # Summaries are not computed on the FAIL path; skip the type check when
        # the gate (expected to) raised. The smoke fixture sets summaries=None
        # in that case.
        if summaries is None:
            pytest.skip("smoke fixture hit FAIL path; Summaries not computed")
        assert isinstance(summaries, Summaries)

    def test_adaptation_params_instance(self, smoke_cert_result) -> None:
        _, _, adaptation, _, _ = smoke_cert_result
        assert isinstance(adaptation, AdaptationParams)
        assert adaptation.step_size > 0.0
        assert adaptation.inverse_mass_matrix.ndim >= 1

    def test_cert_instance(self, smoke_cert_result) -> None:
        _, _, _, cert, _ = smoke_cert_result
        assert isinstance(cert, CertificationResult)

    def test_raises_for_analytic_entry(self) -> None:
        key = jax.random.key(99)
        analytic_entry = MODELS["mvn_10"]
        with pytest.raises(ValueError, match="analytic path"):
            certify_reference_nuts(analytic_entry, key)


@pytest.mark.slow
class TestChainStats:
    """chain_stats extraction — structural checks independent of gate verdict."""

    def test_chain_stats_returned(self, smoke_cert_result) -> None:
        _, _, _, _, chain_stats = smoke_cert_result
        assert isinstance(chain_stats, dict)

    def test_chain_stats_has_required_fields(self, smoke_cert_result) -> None:
        _, _, _, _, chain_stats = smoke_cert_result
        required_fields = {
            "num_integration_steps",
            "energy",
            "is_divergent",
            "acceptance_rate",
        }
        missing = required_fields - set(chain_stats.keys())
        assert not missing, f"chain_stats missing fields: {missing}"

    def test_chain_stats_field_shapes(self, smoke_cert_result) -> None:
        _, _, _, _, chain_stats = smoke_cert_result
        for field_name, arr in chain_stats.items():
            assert (
                arr.shape[0] == SMOKE_N_SAMPLES
            ), f"Field {field_name!r}: expected shape[0]={SMOKE_N_SAMPLES}, got {arr.shape[0]}"


# Default-clean baseline used by gate-logic tests. Mutated per-test by
# overlay kwargs to flip one metric at a time across its threshold. The
# Any annotation lets us mix int/float values for unpacking into
# compute_certification_verdict (which has int-typed num_divergences/n_samples
# alongside float-typed thresholds).
_CLEAN_KW: dict[str, Any] = dict(
    split_rhat_max=1.001,  # well below 1.01
    min_chunk_bulk_ess=800.0,  # well above 400
    num_divergences=0,
    e_bfmi=0.95,  # well above 0.3
    n_samples=10_000,  # divergence allowance: 0.1% × 10_000 = 10
)


@pytest.mark.fast
class TestCertifyNutsGateLogic:
    """Unit tests of compute_certification_verdict — synthetic inputs, no NUTS."""

    def test_clean_inputs_pass(self) -> None:
        cert = compute_certification_verdict(**_CLEAN_KW)
        assert cert.passed is True

    def test_rhat_at_threshold_passes(self) -> None:
        """split_rhat_max == 1.01 (exactly the threshold) should PASS — gate is <= not <."""
        kw = {**_CLEAN_KW, "split_rhat_max": 1.01}
        assert compute_certification_verdict(**kw).passed is True

    def test_rhat_above_threshold_fails(self) -> None:
        kw = {**_CLEAN_KW, "split_rhat_max": 1.011}
        cert = compute_certification_verdict(**kw)
        assert cert.passed is False
        assert cert.split_rhat_max == 1.011  # echoed back unchanged

    def test_ess_at_threshold_passes(self) -> None:
        """min_chunk_bulk_ess == 400 (exactly the threshold) should PASS — gate is >= not >."""
        kw = {**_CLEAN_KW, "min_chunk_bulk_ess": 400.0}
        assert compute_certification_verdict(**kw).passed is True

    def test_ess_below_threshold_fails(self) -> None:
        kw = {**_CLEAN_KW, "min_chunk_bulk_ess": 399.9}
        cert = compute_certification_verdict(**kw)
        assert cert.passed is False
        assert cert.min_chunk_bulk_ess == 399.9

    def test_divergences_at_allowance_passes(self) -> None:
        """num_divergences == max_allowed (10 at default tolerance 0.001 × 10_000) passes."""
        kw = {**_CLEAN_KW, "num_divergences": 10}
        assert compute_certification_verdict(**kw).passed is True

    def test_divergences_above_allowance_fails(self) -> None:
        kw = {**_CLEAN_KW, "num_divergences": 11}
        cert = compute_certification_verdict(**kw)
        assert cert.passed is False
        assert cert.num_divergences == 11

    def test_divergence_rate_tolerance_override(self) -> None:
        """Per-model tolerance override widens the allowance (e.g. stoch_vol uses 0.005)."""
        # At n=10_000 + tolerance=0.005, allowance is 50 — so 11 divs passes here.
        kw = {**_CLEAN_KW, "num_divergences": 11, "divergence_rate_tolerance": 0.005}
        assert compute_certification_verdict(**kw).passed is True

    def test_ebfmi_at_threshold_passes(self) -> None:
        """e_bfmi == 0.3 (exactly the threshold) should PASS — gate is >= not >."""
        kw = {**_CLEAN_KW, "e_bfmi": 0.3}
        assert compute_certification_verdict(**kw).passed is True

    def test_ebfmi_below_threshold_fails(self) -> None:
        kw = {**_CLEAN_KW, "e_bfmi": 0.29}
        cert = compute_certification_verdict(**kw)
        assert cert.passed is False
        assert cert.e_bfmi == 0.29

    def test_returns_frozen_dataclass(self) -> None:
        """CertificationResult is frozen — mutation must raise."""
        cert = compute_certification_verdict(**_CLEAN_KW)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cert.passed = False  # type: ignore[misc]

    def test_threshold_overrides_loosen_or_tighten_gate(self) -> None:
        """Per-call threshold overrides exist for edge-case testing only."""
        # Tighten rhat threshold to 1.005; clean inputs (rhat=1.001) still pass.
        assert (
            compute_certification_verdict(**_CLEAN_KW, rhat_threshold=1.005).passed
            is True
        )
        # Same but with rhat=1.006 → now fails under tightened gate.
        kw_high_rhat = {**_CLEAN_KW, "split_rhat_max": 1.006}
        assert (
            compute_certification_verdict(**kw_high_rhat, rhat_threshold=1.005).passed
            is False
        )
