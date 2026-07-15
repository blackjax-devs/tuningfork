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
"""Tests for catalog/render.py — load_idata multichain detection and shape correctness.

Covers the bug where multichain GT draws (n_chains, n_draws, *event) were
treated as single-chain draws, garbling the posterior group via the
cert-protocol reshape path.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.fast


def _make_recipe(model_name: str, n_chunks: int = 4) -> MagicMock:
    """Build a minimal mock Recipe for render.load_idata."""
    recipe = MagicMock()
    recipe.model_name = model_name
    recipe.warmup_params = {"n_chunks": n_chunks}
    # Effort.GROUNDTRUTH — make Effort comparison work
    try:
        from tuningfork.recipes._base import Effort

        recipe.effort = Effort.GROUNDTRUTH
    except ImportError:
        recipe.effort = "groundtruth"
    return recipe


class TestLoadIdataMultichainDetection:
    """load_idata must handle multichain draws (post-GT-migration) correctly."""

    def test_multichain_draws_produce_correct_posterior_shape(self) -> None:
        """When summary_v2.json reports n_chains=10, draws (10,10000) → posterior (10,10000).

        This is the core regression: without multichain detection, the old code
        treated (10, 10000) as a single chain with 10 draws, then with n_chunks=4
        tried arr[:8].reshape(4, 2, 10000) → garbled (4, 2, 10000) posterior.
        With the fix, is_multichain=True → posterior correctly has 10 chains × 10000 draws.
        """
        from tuningfork.catalog.render import load_idata

        _N_CHAINS = 10
        _N_DRAWS = 10000
        multichain_samples = {
            "mu": np.zeros((_N_CHAINS, _N_DRAWS)),
            "tau": np.ones((_N_CHAINS, _N_DRAWS)),
        }
        recipe = _make_recipe("eight_schools_ncp", n_chunks=4)
        mock_posterior = MagicMock()

        with (
            patch(
                "tuningfork.catalog.render.load_samples",
                return_value=multichain_samples,
            ),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains",
                return_value=_N_CHAINS,
            ),
            patch.dict(
                "tuningfork.catalog.render.MODELS",
                {"eight_schools_ncp": mock_posterior},
            ),
        ):
            idata = load_idata(recipe)

        # Posterior must have 10 chains × 10000 draws — not garbled
        mu_vals = idata.posterior["mu"].values
        assert mu_vals.shape == (_N_CHAINS, _N_DRAWS), (
            f"Expected posterior['mu'].shape == ({_N_CHAINS}, {_N_DRAWS}), "
            f"got {mu_vals.shape}. "
            "The multichain detection fix may be broken — check is_multichain logic."
        )
        tau_vals = idata.posterior["tau"].values
        assert tau_vals.shape == (_N_CHAINS, _N_DRAWS)

    def test_multichain_draws_with_event_dim(self) -> None:
        """Vector params (n_chains, n_draws, dim) are passed through unchanged."""
        from tuningfork.catalog.render import load_idata

        _N_CHAINS, _N_DRAWS, _DIM = 10, 10000, 8
        multichain_samples = {
            "theta": np.zeros((_N_CHAINS, _N_DRAWS, _DIM)),
        }
        recipe = _make_recipe("eight_schools_ncp", n_chunks=4)
        mock_posterior = MagicMock()

        with (
            patch(
                "tuningfork.catalog.render.load_samples",
                return_value=multichain_samples,
            ),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains",
                return_value=_N_CHAINS,
            ),
            patch.dict(
                "tuningfork.catalog.render.MODELS",
                {"eight_schools_ncp": mock_posterior},
            ),
        ):
            idata = load_idata(recipe)

        assert idata.posterior["theta"].values.shape == (_N_CHAINS, _N_DRAWS, _DIM)

    def test_legacy_single_chain_unchanged(self) -> None:
        """Without summary_v2 (n_chains=None), single-chain draws keep existing reshape."""
        from tuningfork.catalog.render import load_idata

        _N_DRAWS = 40000
        _N_CHUNKS = 4
        single_chain_samples = {
            "mu": np.zeros((_N_DRAWS,)),
        }
        recipe = _make_recipe("gp_regression", n_chunks=_N_CHUNKS)
        mock_posterior = MagicMock()

        with (
            patch(
                "tuningfork.catalog.render.load_samples",
                return_value=single_chain_samples,
            ),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            # No summary_v2 → get_groundtruth_n_chains returns None
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains", return_value=None
            ),
            patch.dict(
                "tuningfork.catalog.render.MODELS", {"gp_regression": mock_posterior}
            ),
            # Suppress the n_chunks reshape warning (expected for legacy path)
            pytest.warns(UserWarning, match="cert-protocol reshape"),
        ):
            idata = load_idata(recipe)

        # Single-chain 40000 draws split into 4 chunks of 10000
        expected_shape = (_N_CHUNKS, _N_DRAWS // _N_CHUNKS)
        assert idata.posterior["mu"].values.shape == expected_shape, (
            f"Legacy single-chain reshape should give {expected_shape}, "
            f"got {idata.posterior['mu'].values.shape}"
        )

    def test_multichain_suppresses_n_chunks_warning(self) -> None:
        """No reshape warning for multichain draws — n_chunks is inapplicable."""
        import warnings

        from tuningfork.catalog.render import load_idata

        multichain_samples = {"mu": np.zeros((10, 10000))}
        recipe = _make_recipe("lotka_volterra", n_chunks=4)
        mock_posterior = MagicMock()

        with (
            patch(
                "tuningfork.catalog.render.load_samples",
                return_value=multichain_samples,
            ),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains", return_value=10
            ),
            patch.dict(
                "tuningfork.catalog.render.MODELS", {"lotka_volterra": mock_posterior}
            ),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # any warning → test fails
                load_idata(recipe)  # must not warn


class TestGetGroundtruthNChains:
    """Unit tests for _cache_io.get_groundtruth_n_chains."""

    def test_returns_n_chains_when_summary_v2_present(self, tmp_path: Path) -> None:
        """Reads n_chains from summary_v2.json when present."""
        from tuningfork._cache_io import get_groundtruth_n_chains

        model = "eight_schools_ncp"
        sv2_dir = tmp_path / model / "groundtruth_samples" / "blackjax"
        sv2_dir.mkdir(parents=True)
        sv2_data = {"n_chains": 10, "n_draws_per_chain": 10000, "n_total": 100000}
        (sv2_dir / "summary_v2.json").write_text(json.dumps(sv2_data))

        mock_entry = MagicMock()
        mock_entry.name = model

        result = get_groundtruth_n_chains(mock_entry, cache_dir=tmp_path)
        assert result == 10

    def test_returns_none_when_summary_v2_absent(self, tmp_path: Path) -> None:
        """Returns None when summary_v2.json does not exist (legacy model)."""
        from tuningfork._cache_io import get_groundtruth_n_chains

        mock_entry = MagicMock()
        mock_entry.name = "gp_regression"

        result = get_groundtruth_n_chains(mock_entry, cache_dir=tmp_path)
        assert result is None

    def test_returns_none_on_malformed_json(self, tmp_path: Path) -> None:
        """Returns None gracefully when summary_v2.json is malformed."""
        from tuningfork._cache_io import get_groundtruth_n_chains

        model = "bad_model"
        sv2_dir = tmp_path / model / "groundtruth_samples" / "blackjax"
        sv2_dir.mkdir(parents=True)
        (sv2_dir / "summary_v2.json").write_text("not valid json {{{")

        mock_entry = MagicMock()
        mock_entry.name = model

        result = get_groundtruth_n_chains(mock_entry, cache_dir=tmp_path)
        assert result is None

    def test_returns_none_when_n_chains_missing(self, tmp_path: Path) -> None:
        """Returns None when summary_v2.json lacks n_chains field."""
        from tuningfork._cache_io import get_groundtruth_n_chains

        model = "no_n_chains"
        sv2_dir = tmp_path / model / "groundtruth_samples" / "blackjax"
        sv2_dir.mkdir(parents=True)
        (sv2_dir / "summary_v2.json").write_text(json.dumps({"n_total": 100000}))

        mock_entry = MagicMock()
        mock_entry.name = model

        result = get_groundtruth_n_chains(mock_entry, cache_dir=tmp_path)
        assert result is None
