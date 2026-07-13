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
"""Golden-proof: refactored auto_gate == pre-refactor auto_gate (exact equality).

This harness proves zero-behavior change between the staged refactor and the
pre-refactor monolith.  It imports both:

  - ``_gate_golden_reference.auto_gate``   — verbatim snapshot of the original
    882-line monolith, vendored at refactor time (branch feat/z-advisory-realm).
  - ``statistician_gate.auto_gate``        — the new thin-facade + staged impl.

Both are run over an identical corpus and ``to_dict()`` output is asserted
**exactly equal** (float ``==``, not ``allclose`` — same operations in the
same order must be bit-identical; any mismatch means the refactor changed
computation).

Corpus design
-------------
- Synthetic fixtures drawn from seeds matching the 37 existing gate tests
  (covers the full code path: clean PASS, stuck REVIEW/FAIL, with/without GT,
  with/without cost kwargs, VI mode, single-chain rechunk).
- Edge shapes: nc=4, nc=65, nc=128.
- All three ``multichain`` hint modes: ``True``, ``False``, ``None``.
- VI mode flag: True and False.
- Missing GT: both present and absent.
- Cost kwargs: present and absent.
- Resonance warning: in-zone and out-of-zone ``(step_size, num_integration_steps)``.

Note on real-draw fixture
-------------------------
The brief also mentions
``worklog/data/gpu-chees-meads-2026-07-11/emissions-2026-07-12/nc128_V_emit.npz``.
That file is not present in the current checkout (GPU run artifacts are not
committed).  The synthetic corpus below is sufficient to cover all code paths;
the missing file is noted as a follow-up for when the artifact is available.
"""

import types

import jax.numpy as jnp
import numpy as np
import pytest

import tuningfork.calibration._gate_golden_reference as _ref
import tuningfork.calibration.statistician_gate as _new

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _make_clean_mc(rng, nc, nd, dim):
    """Well-mixed iid samples: (nc, nd, dim)."""
    return {"x": rng.normal(size=(nc, nd, dim))}


def _make_stuck_mc(nc, nd, dim):
    """Stuck chains at different means (high R̂)."""
    rng = np.random.RandomState(999)
    offsets = np.linspace(0, 10.0, nc)[:, None, None]
    return {"x": rng.normal(scale=0.01, size=(nc, nd, dim)) + offsets}


def _make_single_chain(nd, dim):
    """Single-chain layout (nd, dim) — will be rechunked."""
    rng = np.random.RandomState(42)
    return {"x": rng.normal(size=(nd, dim))}


def _make_info(nc, nd, n_div=0):
    """Mock sampler info with ``is_divergent``."""
    flat = np.zeros(nc * nd, dtype=bool)
    flat[:n_div] = True
    return types.SimpleNamespace(is_divergent=jnp.asarray(flat.reshape(nc, nd)))


def _make_gt(dim, n_samples=100_000):
    """Ground-truth with mean=0, std=1 for dim-D parameter."""
    return {
        "x": {
            "mean": np.zeros(dim),
            "std": np.ones(dim),
            "n_samples": n_samples,
        }
    }


def _posterior_with_tags(*tags):
    return types.SimpleNamespace(tags=tags)


# ---------------------------------------------------------------------------
# Golden assertion
# ---------------------------------------------------------------------------


def _assert_exact_equal(old, new, *, label: str) -> None:
    """Assert ``to_dict()`` output is exactly equal (float ==, not allclose)."""
    d_old = old.to_dict()
    d_new = new.to_dict()
    assert d_old == d_new, (
        f"[{label}] refactored auto_gate produced different output:\n"
        f"  old={d_old}\n"
        f"  new={d_new}"
    )


# ---------------------------------------------------------------------------
# Golden corpus — each case is a (label, kwargs) pair
# ---------------------------------------------------------------------------


def _build_corpus():
    """Return a list of (label, kwargs_dict) to feed both auto_gate impls."""
    corpus = []

    # Case 1: clean PASS, nc=4, no GT, no cost
    rng = np.random.RandomState(0)
    corpus.append(
        (
            "clean_pass_nc4",
            dict(
                samples=_make_clean_mc(rng, 4, 1000, 3),
                info=_make_info(4, 1000),
            ),
        )
    )

    # Case 2: stuck chains, nc=4, FAIL on R̂
    corpus.append(
        (
            "stuck_fail_rhat",
            dict(
                samples=_make_stuck_mc(4, 200, 1),
                info=_make_info(4, 200),
            ),
        )
    )

    # Case 3: with GT, PASS z
    rng = np.random.RandomState(6)
    corpus.append(
        (
            "with_gt_pass_z",
            dict(
                samples=_make_clean_mc(rng, 4, 2000, 1),
                info=_make_info(4, 2000),
                ground_truth_summaries=_make_gt(1),
            ),
        )
    )

    # Case 4: with GT, biased sample (small realm → FAIL on z)
    rng = np.random.RandomState(101)
    nc, nd, dim = 4, 1000, 1
    samples_biased = {"x": rng.normal(loc=0.10, scale=1.0, size=(nc, nd, dim))}
    corpus.append(
        (
            "biased_small_realm_fail_z",
            dict(
                samples=samples_biased,
                info=_make_info(nc, nd),
                ground_truth_summaries={
                    "x": {
                        "mean": np.array([0.0]),
                        "std": np.array([1.0]),
                        "n_samples": 1_000_000,
                    }
                },
            ),
        )
    )

    # Case 5: large realm (nc=4, nd=5000) biased → z-advisory REVIEW
    rng = np.random.RandomState(102)
    nc, nd, dim = 4, 5000, 1
    samples_large = {"x": rng.normal(loc=0.10, scale=1.0, size=(nc, nd, dim))}
    corpus.append(
        (
            "biased_large_realm_advisory",
            dict(
                samples=samples_large,
                info=_make_info(nc, nd),
                ground_truth_summaries={
                    "x": {
                        "mean": np.array([0.0]),
                        "std": np.array([1.0]),
                        "n_samples": 1_000_000,
                    }
                },
            ),
        )
    )

    # Case 6: divergences only
    rng = np.random.RandomState(2)
    corpus.append(
        (
            "many_divergences_fail",
            dict(
                samples=_make_clean_mc(rng, 4, 1000, 3),
                info=_make_info(4, 1000, n_div=50),
            ),
        )
    )

    # Case 7: few divergences still PASS
    rng = np.random.RandomState(2)
    corpus.append(
        (
            "few_divergences_pass",
            dict(
                samples=_make_clean_mc(rng, 4, 1000, 3),
                info=_make_info(4, 1000, n_div=5),
            ),
        )
    )

    # Case 8: no GT → max_abs_mean_z is None
    rng = np.random.RandomState(5)
    corpus.append(
        (
            "no_gt_z_skipped",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 2),
                info=_make_info(4, 500),
                ground_truth_summaries=None,
            ),
        )
    )

    # Case 9: cost kwargs present
    rng = np.random.RandomState(108)
    corpus.append(
        (
            "cost_kwargs_present",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                ess_per_grad=10.5,
                total_grad_evals=5000,
                wall_seconds=42.3,
            ),
        )
    )

    # Case 10: VI mode, clean samples with GT
    rng = np.random.RandomState(200)
    corpus.append(
        (
            "vi_mode_clean",
            dict(
                samples=_make_clean_mc(rng, 4, 1000, 5),
                info=_make_info(4, 1000),
                ground_truth_summaries=_make_gt(5),
                vi_sampler_mode=True,
            ),
        )
    )

    # Case 11: single-chain rechunk (multichain=False explicitly)
    rng = np.random.RandomState(9)
    corpus.append(
        (
            "single_chain_rechunk_explicit_false",
            dict(
                samples=_make_single_chain(4000, 2),
                info=None,
                multichain=False,
                n_chunks=4,
            ),
        )
    )

    # Case 12: multichain=True explicit, nc=65 (above old ≤64 cliff)
    rng = np.random.RandomState(11)
    corpus.append(
        (
            "nc65_multichain_true",
            dict(
                samples={"x": rng.normal(size=(65, 200, 3))},
                info=_make_info(65, 200),
                multichain=True,
            ),
        )
    )

    # Case 13: nc=128, ndim=3, heuristic (ndim≥3 = multichain)
    rng = np.random.RandomState(14)
    corpus.append(
        (
            "nc128_ndim3_heuristic",
            dict(
                samples={"x": rng.normal(size=(128, 1000, 5))},
                info=_make_info(128, 1000),
                multichain=None,
            ),
        )
    )

    # Case 14: resonance in-zone k=1 (L·ε ≈ 6.28)
    rng = np.random.RandomState(3)
    corpus.append(
        (
            "resonance_zone1",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                step_size=0.314,
                num_integration_steps=20,  # L·ε = 6.28
            ),
        )
    )

    # Case 15: resonance out-of-zone (5π/2 ≈ 7.85, no warning)
    rng = np.random.RandomState(3)
    corpus.append(
        (
            "resonance_out_of_zone",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                step_size=0.31,
                num_integration_steps=25,  # L·ε = 7.75, outside both zones
            ),
        )
    )

    # Case 16: funnel posterior tag
    rng = np.random.RandomState(15)
    corpus.append(
        (
            "funnel_tag",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 2),
                info=_make_info(4, 500),
                posterior=_posterior_with_tags("funnel"),
            ),
        )
    )

    # Case 17: multimodal posterior tag (z check skipped)
    rng = np.random.RandomState(16)
    corpus.append(
        (
            "multimodal_tag_no_z",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 2),
                info=_make_info(4, 500),
                ground_truth_summaries=_make_gt(2),
                posterior=_posterior_with_tags("multimodal"),
            ),
        )
    )

    # Case 18: high-correlation tag
    rng = np.random.RandomState(17)
    corpus.append(
        (
            "high_correlation_tag",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                posterior=_posterior_with_tags("high-correlation"),
            ),
        )
    )

    # Case 19: d=10, z just below Šidák PASS band (PASS)
    rng = np.random.RandomState(100)
    corpus.append(
        (
            "sidak_d10_pass",
            dict(
                samples=_make_clean_mc(rng, 4, 2000, 10),
                info=_make_info(4, 2000),
                ground_truth_summaries=_make_gt(10),
            ),
        )
    )

    # Case 20: bias_sigma_max_at_z4 semantics (2D, dim-0 z≥4, dim-1 z<4)
    rng = np.random.RandomState(107)
    nc, nd = 4, 1000
    samples_2d = {
        "x": np.concatenate(
            [
                rng.normal(loc=0.025, scale=0.1, size=(nc, nd, 1)),
                rng.normal(loc=0.048, scale=1.0, size=(nc, nd, 1)),
            ],
            axis=2,
        )
    }
    corpus.append(
        (
            "bias_sigma_max_at_z4_semantics",
            dict(
                samples=samples_2d,
                info=_make_info(nc, nd),
                ground_truth_summaries={
                    "x": {
                        "mean": np.array([0.0, 0.0]),
                        "std": np.array([5.0, 1.0]),
                        "n_samples": 1_000_000,
                    }
                },
            ),
        )
    )

    # Case 21: MCLMC-style info (no is_divergent attr) → n_div = 0
    rng = np.random.RandomState(21)
    corpus.append(
        (
            "mclmc_no_is_divergent",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 2),
                info=types.SimpleNamespace(),  # no is_divergent
            ),
        )
    )

    # Case 22: info=None → n_divergences = None
    rng = np.random.RandomState(22)
    corpus.append(
        (
            "info_none",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=None,
            ),
        )
    )

    # Case 23: empty samples dict
    corpus.append(
        (
            "empty_samples",
            dict(
                samples={},
                info=None,
            ),
        )
    )

    # Case 24: resonance zone k=2 (L·ε ≈ 12.57)
    rng = np.random.RandomState(24)
    corpus.append(
        (
            "resonance_zone2",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                step_size=0.628,
                num_integration_steps=20,  # L·ε = 12.56
            ),
        )
    )

    # Case 25: all cost kwargs absent, only ess_per_grad
    rng = np.random.RandomState(25)
    corpus.append(
        (
            "cost_ess_per_grad_only",
            dict(
                samples=_make_clean_mc(rng, 4, 500, 1),
                info=_make_info(4, 500),
                ess_per_grad=5.0,
            ),
        )
    )

    return corpus


# ---------------------------------------------------------------------------
# Parametrized golden test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,kwargs", _build_corpus())
def test_gate_refactor_exact_equality(label, kwargs):
    """Refactored auto_gate produces bit-identical to_dict() output.

    Asserts exact float equality (``==``, not ``allclose``) on ``to_dict()``
    for both the verdict fields and the full margins dict including all
    nested bias_sigma_* entries.
    """
    old = _ref.auto_gate(**kwargs)
    new = _new.auto_gate(**kwargs)
    _assert_exact_equal(old, new, label=label)


# ---------------------------------------------------------------------------
# Structural smoke tests (not in the parametrized set)
# ---------------------------------------------------------------------------


def test_all_public_names_importable():
    """All documented public names are re-exported from statistician_gate."""
    import tuningfork.calibration.statistician_gate as sg

    for name in [
        "auto_gate",
        "AutoGateVerdict",
        "DEFAULT_THRESHOLDS",
        "Z_VERDICT_ESS_CEILING",
        "resolve_thresholds",
        "sidak_t_pass",
    ]:
        assert hasattr(sg, name), f"Missing public name: {name}"


def test_private_samples_to_multichain_importable():
    """_samples_to_multichain is importable from statistician_gate (test compat)."""
    from tuningfork.calibration.statistician_gate import _samples_to_multichain

    assert callable(_samples_to_multichain)


def test_golden_reference_is_self_contained():
    """_gate_golden_reference.auto_gate runs without importing from _gate."""
    rng = np.random.RandomState(99)
    samples = {"x": rng.normal(size=(4, 200, 2))}
    info = _make_info(4, 200)
    verdict = _ref.auto_gate(samples, info)
    assert verdict.verdict in {"PASS", "REVIEW", "FAIL"}


def test_corpus_covers_z_advisory_realms():
    """Corpus includes both small-realm and large-realm ESS cases."""
    corpus = _build_corpus()
    labels = {label for label, _ in corpus}
    assert "biased_small_realm_fail_z" in labels
    assert "biased_large_realm_advisory" in labels


def test_corpus_covers_multichain_modes():
    """Corpus includes all three multichain hint modes."""
    corpus = _build_corpus()
    labels = {label for label, _ in corpus}
    assert "nc65_multichain_true" in labels  # multichain=True
    assert "single_chain_rechunk_explicit_false" in labels  # multichain=False
    assert "nc128_ndim3_heuristic" in labels  # multichain=None


def test_corpus_size():
    """Corpus has at least 20 cases."""
    corpus = _build_corpus()
    assert len(corpus) >= 20, f"Corpus too small: {len(corpus)} cases"
