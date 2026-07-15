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
"""Tests for the closed-form GP marginal generation path.

Fast tests: structural dispatch helpers (_load_explicit_positions,
x64-check error, n_chains validation error).

Slow tests: smoke run at 2×50×100 scale — skipped unless
``JAX_ENABLE_X64=1`` is set at process start.

Note: gp_regression requires 64-bit floats.  In CI (JAX_PLATFORM_NAME=cpu
without x64), the slow smoke test is skipped rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary
from tuningfork.groundtruth._gp_marginal import (
    _load_explicit_positions,
    generate_gp_marginal,
)

_X64_ENABLED = bool(jax.config.read("jax_enable_x64"))

# --------------------------------------------------------------------------- #
# fast: structural checks that do not require x64
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_gp_marginal_requires_x64_raises_without_it() -> None:
    """generate_gp_marginal raises RuntimeError if JAX_ENABLE_X64 is not set."""
    if _X64_ENABLED:
        pytest.skip("x64 is enabled; cannot test the x64-check error path")
    committed = load_committed_summary("gp_regression")
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        generate_gp_marginal("gp_regression", committed, Path("/tmp"))


@pytest.mark.fast
def test_gp_marginal_wrong_model_raises() -> None:
    """generate_gp_marginal rejects model names other than 'gp_regression'."""
    committed = load_committed_summary("radon")
    with pytest.raises(ValueError, match="gp_regression"):
        generate_gp_marginal("radon", committed, Path("/tmp"))


@pytest.mark.fast
def test_gp_marginal_committed_has_init_positions() -> None:
    """Committed gp_regression summary has the init_positions block."""
    committed = load_committed_summary("gp_regression")
    pos = _load_explicit_positions(committed)
    sites = list(pos.keys())
    # Should contain the three hyperparameter sites
    expected_sites = {"log_lengthscale", "log_kernel_scale", "log_noise_scale"}
    assert expected_sites.issubset(
        set(sites)
    ), f"Missing sites: {expected_sites - set(sites)}"
    n_chains = len(pos[sites[0]])
    assert n_chains > 0


@pytest.mark.fast
def test_gp_marginal_n_chains_exceeds_committed_raises() -> None:
    """Requesting more chains than committed init_positions raises ValueError."""
    if not _X64_ENABLED:
        pytest.skip("x64 not enabled; generate_gp_marginal would raise earlier")
    committed = load_committed_summary("gp_regression")
    pos = _load_explicit_positions(committed)
    n_committed = len(pos[list(pos.keys())[0]])
    with pytest.raises(ValueError, match="n_chains"):
        generate_gp_marginal(
            "gp_regression", committed, Path("/tmp"), n_chains=n_committed + 99
        )


# --------------------------------------------------------------------------- #
# slow: smoke run (requires x64)
# --------------------------------------------------------------------------- #

_SMOKE_N_CHAINS = 2
_SMOKE_N_DRAWS = 50
_SMOKE_N_WARMUP = 100


@pytest.mark.slow
@pytest.mark.skipif(not _X64_ENABLED, reason="gp_regression requires JAX_ENABLE_X64=1")
def test_gp_marginal_smoke(tmp_path: Path) -> None:
    """Smoke: gp_regression generates correct schema and f_raw shape at 2×50."""
    model_name = "gp_regression"
    committed = load_committed_summary(model_name)
    committed_gt = committed_gt_dir(model_name)

    result = generate_gp_marginal(
        model_name,
        committed,
        tmp_path,
        smoke=True,
    )

    # --- schema ---
    assert result["schema_version"] == "gt_v2_multichain"
    assert result["generator"] == (
        "nuts_on_closed_form_gp_marginal_plus_conditional_f_reconstruction"
    )
    assert result["model_name"] == model_name
    assert result["n_chains"] == _SMOKE_N_CHAINS
    assert result["n_draws_per_chain"] == _SMOKE_N_DRAWS

    # --- gate fields (not gate pass at smoke scale) ---
    gate = result["quality_gate"]
    assert "max_rhat" in gate
    assert gate["max_rhat"] < 2.0, f"max_rhat={gate['max_rhat']:.4f} >= 2.0"

    # --- draws.npz shape: hyperparameters + f_raw ---
    draws_path = tmp_path / "draws.npz"
    assert draws_path.exists()
    draws = np.load(str(draws_path))

    # Hyperparameter sites must have shape (nc, nd)
    for hp_site in ("log_lengthscale", "log_kernel_scale", "log_noise_scale"):
        assert hp_site in draws.files, f"{hp_site} missing from draws.npz"
        shape = draws[hp_site].shape
        assert shape == (
            _SMOKE_N_CHAINS,
            _SMOKE_N_DRAWS,
        ), f"{hp_site}: expected ({_SMOKE_N_CHAINS}, {_SMOKE_N_DRAWS}), got {shape}"

    # f_raw must have shape (nc, nd, 200)
    assert "f_raw" in draws.files, "f_raw missing from draws.npz"
    f_raw_shape = draws["f_raw"].shape
    assert f_raw_shape == (
        _SMOKE_N_CHAINS,
        _SMOKE_N_DRAWS,
        200,
    ), f"f_raw: expected ({_SMOKE_N_CHAINS}, {_SMOKE_N_DRAWS}, 200), got {f_raw_shape}"

    # f_raw must be finite
    assert np.all(np.isfinite(draws["f_raw"])), "f_raw contains non-finite values"

    # --- crude mean coherence for hyperparameters vs committed GT ---
    committed_draws = np.load(str(committed_gt / "draws.npz"))
    per_site = result["per_site"]
    for site in ("log_lengthscale", "log_kernel_scale", "log_noise_scale"):
        if site not in per_site or site not in committed_draws.files:
            continue
        new_mean = float(np.asarray(per_site[site]["mean"]))
        comm_arr = committed_draws[site]
        comm_mean = float(comm_arr.mean())
        comm_std = float(comm_arr.std(ddof=1))
        if comm_std < 1e-12:
            continue
        dev = abs(new_mean - comm_mean) / comm_std
        assert dev < 10.0, (
            f"gp_regression.{site}: new mean deviates {dev:.2f}× posterior std "
            f"(new={new_mean:.4f}, committed={comm_mean:.4f}±{comm_std:.4f})"
        )
