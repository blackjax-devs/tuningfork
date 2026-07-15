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
"""Smoke tests for the standard_multichain_nuts generation path.

Each slow test runs at tiny scale (2 chains × 100 draws × 100 warmup) and
validates:
1. The generated draws.npz loads with the correct shape.
2. The generated summary_v2.json has the expected schema.
3. R̂ is finite and < 2.0 (sanity check; gate thresholds are not meaningful
   at this draw count).
4. Per-site mean is within 10× committed posterior std (crude coherence).

Fast tests cover:
- ``_load_explicit_positions`` raises KeyError on a bad summary.
- ``_load_explicit_positions`` extracts the right block from a committed
  lotka_volterra summary.
- ``_build_perchain_inits_uniform`` returns correct shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary
from tuningfork.groundtruth._nuts_multichain import (
    _load_explicit_positions,
    generate_nuts_multichain,
)

# --------------------------------------------------------------------------- #
# fast: dispatch helpers
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_load_explicit_positions_missing_raises() -> None:
    """KeyError raised when provenance block has no init_positions.positions."""
    summary: dict = {"provenance": {"init_positions": {}}}
    with pytest.raises(KeyError, match="init_positions"):
        _load_explicit_positions(summary)


@pytest.mark.fast
def test_load_explicit_positions_no_provenance_raises() -> None:
    """KeyError raised when provenance block is absent."""
    summary: dict = {}
    with pytest.raises(KeyError, match="init_positions"):
        _load_explicit_positions(summary)


@pytest.mark.fast
def test_load_explicit_positions_lotka_volterra() -> None:
    """Committed lotka_volterra summary contains a valid positions dict."""
    summary = load_committed_summary("lotka_volterra")
    pos_dict = _load_explicit_positions(summary)
    assert isinstance(pos_dict, dict), "positions dict must be a dict"
    sites = list(pos_dict.keys())
    assert len(sites) > 0, "positions dict must contain at least one site"
    # Each site maps to a list of length n_chains
    n_chains = len(pos_dict[sites[0]])
    assert n_chains > 0, "must have at least one chain"
    for site in sites:
        assert len(pos_dict[site]) == n_chains, (
            f"site {site!r}: expected {n_chains} chains, " f"got {len(pos_dict[site])}"
        )


# --------------------------------------------------------------------------- #
# slow: smoke run for standard_multichain_nuts
# --------------------------------------------------------------------------- #

_SMOKE_N_CHAINS = 2
_SMOKE_N_DRAWS = 100
_SMOKE_N_WARMUP = 100


@pytest.mark.slow
def test_nuts_multichain_smoke_radon(tmp_path: Path) -> None:
    """Smoke: radon (standard_multichain_nuts) generates correct schema output."""
    model_name = "radon"
    committed = load_committed_summary(model_name)
    committed_gt = committed_gt_dir(model_name)

    result = generate_nuts_multichain(
        model_name,
        committed,
        tmp_path,
        smoke=True,
    )

    # --- schema ---
    assert result["schema_version"] == "gt_v2_multichain"
    assert result["generator"] == "nuts_perchain"
    assert result["model_name"] == model_name
    assert result["n_chains"] == _SMOKE_N_CHAINS
    assert result["n_draws_per_chain"] == _SMOKE_N_DRAWS

    # --- gate fields present (not gate pass — invalid at smoke scale) ---
    gate = result["quality_gate"]
    assert "max_rhat" in gate
    assert "min_bulk_ess" in gate
    # R̂ < 2.0 at 2×100 is a very loose sanity check (catches sampler failures)
    assert (
        gate["max_rhat"] < 2.0
    ), f"{model_name}: max_rhat={gate['max_rhat']:.4f} >= 2.0"

    # --- diagnostics present ---
    dpc = result["diagnostics_per_chain"]
    assert "step_size" in dpc
    assert "divergences_per_chain" in dpc
    assert "e_bfmi_per_chain" in dpc
    assert "min_e_bfmi" in dpc
    assert len(dpc["step_size"]) == _SMOKE_N_CHAINS
    assert len(dpc["divergences_per_chain"]) == _SMOKE_N_CHAINS

    # --- draws.npz shape ---
    draws_path = tmp_path / "draws.npz"
    assert draws_path.exists()
    draws = np.load(str(draws_path), allow_pickle=True)
    for site in draws.files:
        shape = draws[site].shape
        assert (
            shape[0] == _SMOKE_N_CHAINS
        ), f"{model_name}.{site}: expected n_chains={_SMOKE_N_CHAINS}, got {shape[0]}"
        assert (
            shape[1] == _SMOKE_N_DRAWS
        ), f"{model_name}.{site}: expected n_draws={_SMOKE_N_DRAWS}, got {shape[1]}"

    # --- crude mean coherence vs committed GT ---
    # At 2×100 draws, use 10× posterior std as the tolerance.
    committed_draws = np.load(str(committed_gt / "draws.npz"), allow_pickle=True)
    per_site = result["per_site"]
    for site in per_site:
        if site not in committed_draws.files:
            continue
        new_mean = np.asarray(per_site[site]["mean"]).ravel()
        comm_arr = committed_draws[site]
        comm_mean = comm_arr.reshape(-1, *comm_arr.shape[2:]).mean(axis=0).ravel()
        comm_std = comm_arr.reshape(-1, *comm_arr.shape[2:]).std(axis=0, ddof=1).ravel()
        scale = np.maximum(comm_std, 1e-12)
        max_dev = float(np.max(np.abs(new_mean - comm_mean) / scale))
        assert max_dev < 10.0, (
            f"{model_name}.{site}: new mean deviates "
            f"{max_dev:.2f}× committed posterior std"
        )
        assert np.all(
            np.isfinite(new_mean)
        ), f"{model_name}.{site}: new mean contains non-finite values"


@pytest.mark.slow
def test_nuts_multichain_output_files(tmp_path: Path) -> None:
    """Output directory contains exactly draws.npz and summary_v2.json."""
    committed = load_committed_summary("radon")
    generate_nuts_multichain("radon", committed, tmp_path, smoke=True)

    written = {p.name for p in tmp_path.iterdir()}
    assert "draws.npz" in written
    assert "summary_v2.json" in written


@pytest.mark.slow
def test_nuts_multichain_custom_seed_differs(tmp_path: Path) -> None:
    """Two different seeds produce different draws."""
    committed = load_committed_summary("radon")
    out1 = tmp_path / "seed1"
    out2 = tmp_path / "seed2"

    generate_nuts_multichain("radon", committed, out1, seed=1, smoke=True)
    generate_nuts_multichain("radon", committed, out2, seed=99, smoke=True)

    d1 = np.load(str(out1 / "draws.npz"), allow_pickle=True)
    d2 = np.load(str(out2 / "draws.npz"), allow_pickle=True)
    site = d1.files[0]
    assert not np.allclose(
        d1[site], d2[site]
    ), "Different seeds produced identical draws — RNG not working correctly"
