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
"""Fast tests for tuningfork.notebooks.render (load_samples, samples_to_idata).

All tests are pure logic / schema — no JAX trace, no real chain runs.
samples_to_idata is tested with numpy arrays (no JAX compilation).
load_samples is tested with a mocked recipe (cache hit and miss paths).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# samples_to_idata tests
# ---------------------------------------------------------------------------


def test_samples_to_idata_single_chain_default() -> None:
    """samples_to_idata promotes (n_draws, *event) to (1, n_draws, *event)."""
    from tuningfork.notebooks import samples_to_idata

    rng = np.random.default_rng(0)
    samples = {
        "mu": rng.standard_normal((100, 3)),
        "sigma": rng.standard_normal((100,)),
    }
    idata = samples_to_idata(samples)

    # ArviZ >= 0.20 returns xarray.DataTree; access posterior group via idata["posterior"]
    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    assert "mu" in posterior
    assert "sigma" in posterior
    # Default is_multichain=False → single chain promoted to (1, n_draws, ...)
    assert posterior["mu"].shape[0] == 1  # n_chains = 1
    assert posterior["mu"].shape[1] == 100  # n_draws


def test_samples_to_idata_multichain() -> None:
    """samples_to_idata with is_multichain=True preserves (n_chains, n_draws, *event)."""
    from tuningfork.notebooks import samples_to_idata

    rng = np.random.default_rng(1)
    samples = {
        "mu": rng.standard_normal((4, 50, 2)),
        "tau": rng.standard_normal((4, 50)),
    }
    idata = samples_to_idata(samples, is_multichain=True)

    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    assert posterior["mu"].shape == (4, 50, 2)
    assert posterior["tau"].shape == (4, 50)


def test_samples_to_idata_dimension_labels() -> None:
    """samples_to_idata produces an idata object with chain and draw dimensions."""
    from tuningfork.notebooks import samples_to_idata

    rng = np.random.default_rng(42)
    samples = {"x": rng.standard_normal((200, 5))}
    idata = samples_to_idata(samples, is_multichain=False)

    # ArviZ uses 'chain' and 'draw' as coordinate names (both legacy and DataTree API)
    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    coords = posterior.coords if hasattr(posterior, "coords") else posterior.ds.coords
    assert "chain" in coords
    assert "draw" in coords


# ---------------------------------------------------------------------------
# load_samples tests — cache miss and cache hit with mock
# ---------------------------------------------------------------------------


def test_load_samples_cache_miss_raises_file_not_found() -> None:
    """load_samples raises FileNotFoundError with a clear message on cache miss."""
    from tuningfork.notebooks import load_samples

    mock_recipe = MagicMock()
    mock_recipe.model_name = "mvn_10"
    mock_recipe.effort.value = "low"
    mock_recipe.base_method_name = "nuts"
    mock_recipe.load_cached_samples.return_value = None

    with pytest.raises(FileNotFoundError, match="No cached samples found"):
        load_samples(mock_recipe)


def test_load_samples_cache_miss_message_points_to_docs() -> None:
    """load_samples error message mentions recipe_diagnostics.md or Phase 0."""
    from tuningfork.notebooks import load_samples

    mock_recipe = MagicMock()
    mock_recipe.model_name = "some_model"
    mock_recipe.effort.value = "groundtruth"
    mock_recipe.base_method_name = "nuts"
    mock_recipe.load_cached_samples.return_value = None

    with pytest.raises(FileNotFoundError) as exc_info:
        load_samples(mock_recipe)

    msg = str(exc_info.value)
    # Should mention useful context: the model name and guidance
    assert "some_model" in msg
    assert "cache" in msg.lower() or "Phase 0" in msg or "recipe_diagnostics" in msg


def test_load_samples_cache_hit_returns_dict() -> None:
    """load_samples returns the dict from recipe.load_cached_samples on cache hit."""
    from tuningfork.notebooks import load_samples

    fake_draws = {
        "mu": np.ones((1000, 8)),
        "tau": np.ones((1000,)),
    }

    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_draws

    result = load_samples(mock_recipe)
    assert result is fake_draws
    mock_recipe.load_cached_samples.assert_called_once_with(cache_dir=None)


def test_load_samples_passes_cache_dir_override() -> None:
    """load_samples forwards cache_dir kwarg to recipe.load_cached_samples."""
    from pathlib import Path

    from tuningfork.notebooks import load_samples

    fake_draws = {"x": np.zeros((100,))}
    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_draws

    custom_dir = Path("/tmp/custom_cache")
    load_samples(mock_recipe, cache_dir=custom_dir)

    mock_recipe.load_cached_samples.assert_called_once_with(cache_dir=custom_dir)
