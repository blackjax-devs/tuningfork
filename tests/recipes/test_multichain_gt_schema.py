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
"""Tests for the multichain GT recipe schema extension (feat/multichain-gt-recipe).

Covers:
- Recipe.load correctly reads gt_schema_version and summary_v2_path for migrated models
- Legacy models (gp_regression) load with gt_schema_version=None (backward-compat)
- load_imm_sidecar(catalog_root) returns None for multichain GT recipes (null path)
- list_recipes still enumerates groundtruth.json for migrated models
- load_idata returns correctly-shaped multichain posterior on new schema (mock-based)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.fast

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


class TestMultichainGTSchemaLoad:
    """Recipe.load reads gt_schema_version + summary_v2_path for migrated models."""

    def test_radon_gt_schema_version(self) -> None:
        """radon groundtruth.json loads with gt_schema_version='gt_v2_multichain'."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert recipe.gt_schema_version == "gt_v2_multichain"

    def test_radon_summary_v2_path(self) -> None:
        """radon groundtruth.json has a non-null summary_v2_path."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert (
            recipe.summary_v2_path
            == "radon/groundtruth_samples/blackjax/summary_v2.json"
        )

    def test_radon_no_inverse_mass_matrix(self) -> None:
        """Multichain GT recipe has no inverse_mass_matrix in base_method_params."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert "inverse_mass_matrix" not in recipe.base_method_params

    def test_radon_step_size_null(self) -> None:
        """Multichain GT has step_size=null (per-chain adapted at runtime)."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert recipe.base_method_params.get("step_size") is None

    def test_radon_warmup_name_valid_registry_key(self) -> None:
        """warmup_name is the registered WARMUPS key (not a per-chain variant)."""
        from tuningfork.recipes._base import Recipe
        from tuningfork.warmup import WARMUPS

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert recipe.warmup_name in WARMUPS, (
            f"warmup_name {recipe.warmup_name!r} not in WARMUPS registry. "
            "Multichain GT recipe must use the registered warmup key."
        )

    def test_radon_load_imm_sidecar_returns_none(self) -> None:
        """load_imm_sidecar returns None (no crash) when inverse_mass_matrix_path=null."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        assert recipe.inverse_mass_matrix_path is None
        result = recipe.load_imm_sidecar(_CATALOG_ROOT)
        assert result is None


class TestLegacyGTSchemaUnchanged:
    """Legacy single-chain GT models load with gt_schema_version=None."""

    def test_gp_regression_gt_schema_version_none(self) -> None:
        """gp_regression groundtruth.json has gt_schema_version=None (legacy)."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "gp_regression" / "groundtruth.json")
        assert recipe.gt_schema_version is None

    def test_gp_regression_summary_v2_path_none(self) -> None:
        """gp_regression has no summary_v2_path (no multichain migration yet)."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "gp_regression" / "groundtruth.json")
        assert recipe.summary_v2_path is None

    def test_gp_regression_imm_sidecar_still_loads(self) -> None:
        """Legacy gp_regression imm sidecar is unchanged and loads correctly."""
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "gp_regression" / "groundtruth.json")
        assert recipe.inverse_mass_matrix_path is not None
        assert recipe.base_method_params.get("inverse_mass_matrix") == "sidecar"


class TestListRecipesEnumeration:
    """list_recipes must still enumerate groundtruth.json for migrated models."""

    def test_radon_gt_in_list_recipes(self) -> None:
        """list_recipes('radon') includes groundtruth.json."""
        from tuningfork.catalog.inspect import list_recipes

        paths = list_recipes("radon")
        filenames = [p.name for p in paths]
        assert (
            "groundtruth.json" in filenames
        ), f"groundtruth.json missing from list_recipes('radon'). Got: {filenames}"

    def test_gp_regression_gt_in_list_recipes(self) -> None:
        """list_recipes('gp_regression') includes groundtruth.json (legacy model)."""
        from tuningfork.catalog.inspect import list_recipes

        paths = list_recipes("gp_regression")
        filenames = [p.name for p in paths]
        assert (
            "groundtruth.json" in filenames
        ), f"groundtruth.json missing from list_recipes('gp_regression'). Got: {filenames}"


class TestLoadIdataMultichainNewSchema:
    """load_idata returns correctly-shaped multichain posterior on new GT schema."""

    def test_load_idata_on_new_radon_schema_returns_multichain_shape(self) -> None:
        """load_idata with new radon groundtruth recipe returns (10, 10000) posterior.

        The key regression: the new schema has no n_chunks in warmup_params and
        base_method_params.step_size=null. load_idata must still route through the
        multichain detection path (get_groundtruth_n_chains → 10) and return
        correct (10, 10000) shape without reshaping as a single chain.
        """
        from tuningfork.catalog.render import load_idata
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        _N_CHAINS, _N_DRAWS = 10, 10000
        mock_samples = {
            "mu": np.zeros((_N_CHAINS, _N_DRAWS)),
            "sigma": np.ones((_N_CHAINS, _N_DRAWS)),
        }
        mock_posterior = MagicMock()

        with (
            patch(
                "tuningfork.catalog.render.load_samples",
                return_value=mock_samples,
            ),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains",
                return_value=_N_CHAINS,
            ),
            patch.dict(
                "tuningfork.catalog.render.MODELS",
                {"radon": mock_posterior},
            ),
        ):
            idata = load_idata(recipe)

        mu_shape = idata.posterior["mu"].values.shape
        assert mu_shape == (_N_CHAINS, _N_DRAWS), (
            f"Expected posterior['mu'].shape == ({_N_CHAINS}, {_N_DRAWS}), "
            f"got {mu_shape}. The new multichain GT schema may have broken "
            "load_idata's multichain detection path."
        )

    def test_load_idata_new_schema_no_n_chunks_warning(self) -> None:
        """No reshape warning when loading radon GT (multichain, no n_chunks)."""
        import warnings

        from tuningfork.catalog.render import load_idata
        from tuningfork.recipes._base import Recipe

        recipe = Recipe.load(_CATALOG_ROOT / "radon" / "groundtruth.json")
        mock_samples = {"mu": np.zeros((10, 10000))}
        mock_posterior = MagicMock()

        with (
            patch("tuningfork.catalog.render.load_samples", return_value=mock_samples),
            patch("tuningfork.catalog.render.load_chain_stats", return_value=None),
            patch(
                "tuningfork.catalog.render.get_groundtruth_n_chains", return_value=10
            ),
            patch.dict("tuningfork.catalog.render.MODELS", {"radon": mock_posterior}),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                load_idata(recipe)
