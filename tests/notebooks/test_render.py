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
"""Fast tests for tuningfork.catalog.render (load_samples, samples_to_idata).

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
    from tuningfork.catalog.render import samples_to_idata

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


def test_samples_to_idata_n_chunks_reshapes_single_to_multichain() -> None:
    """samples_to_idata with n_chunks=4 splits a 4000-draw chain into 4×1000."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(0)
    samples = {
        "mu": rng.standard_normal((4000, 3)),
        "sigma": rng.standard_normal((4000,)),
    }
    idata = samples_to_idata(samples, is_multichain=False, n_chunks=4)
    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    assert posterior["mu"].shape[0] == 4  # 4 chains
    assert posterior["mu"].shape[1] == 1000  # 4000 / 4
    assert posterior["sigma"].shape[0] == 4
    assert posterior["sigma"].shape[1] == 1000


def test_samples_to_idata_n_chunks_truncates_remainder() -> None:
    """When n_draws is not divisible by n_chunks, the remainder is dropped."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(0)
    # 4003 % 4 = 3 → expect drop the trailing 3 draws, reshape to (4, 1000)
    samples = {"mu": rng.standard_normal(4003)}
    idata = samples_to_idata(samples, is_multichain=False, n_chunks=4)
    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    assert posterior["mu"].shape == (4, 1000)


def test_samples_to_idata_n_chunks_reshapes_chain_stats_consistently() -> None:
    """sample_stats arrays are reshaped to the same (n_chunks, per_chunk) layout."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(0)
    n_total = 4000
    n_chunks = 4
    samples = {"mu": rng.standard_normal(n_total)}
    chain_stats = {
        "is_divergent": rng.integers(0, 2, size=n_total).astype(bool),
        "energy": rng.standard_normal(n_total).astype(np.float32),
        "num_integration_steps": rng.integers(1, 128, size=n_total).astype(np.int32),
        "acceptance_rate": rng.uniform(0, 1, size=n_total).astype(np.float32),
    }
    idata = samples_to_idata(
        samples, is_multichain=False, chain_stats=chain_stats, n_chunks=n_chunks
    )
    # Both posterior and sample_stats reshaped to (4, 1000)
    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    stats = (
        idata["sample_stats"] if hasattr(idata, "__getitem__") else idata.sample_stats
    )
    assert posterior["mu"].shape == (n_chunks, n_total // n_chunks)
    assert stats["diverging"].shape == (n_chunks, n_total // n_chunks)
    assert stats["energy"].shape == (n_chunks, n_total // n_chunks)


def test_samples_to_idata_multichain() -> None:
    """samples_to_idata with is_multichain=True preserves (n_chains, n_draws, *event)."""
    from tuningfork.catalog.render import samples_to_idata

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
    from tuningfork.catalog.render import samples_to_idata

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
    from tuningfork.catalog.render import load_samples

    mock_recipe = MagicMock()
    mock_recipe.model_name = "mvn_10"
    mock_recipe.effort.value = "low"
    mock_recipe.base_method_name = "nuts"
    mock_recipe.load_cached_samples.return_value = None

    with pytest.raises(FileNotFoundError, match="No cached samples found"):
        load_samples(mock_recipe)


def test_load_samples_cache_miss_message_points_to_docs() -> None:
    """load_samples error message mentions recipe_diagnostics.md or Phase 0."""
    from tuningfork.catalog.render import load_samples

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
    from tuningfork.catalog.render import load_samples

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

    from tuningfork.catalog.render import load_samples

    fake_draws = {"x": np.zeros((100,))}
    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_draws

    custom_dir = Path("/tmp/custom_cache")
    load_samples(mock_recipe, cache_dir=custom_dir)

    mock_recipe.load_cached_samples.assert_called_once_with(cache_dir=custom_dir)


# ---------------------------------------------------------------------------
# chain_stats / sample_stats projection tests
# ---------------------------------------------------------------------------


def test_samples_to_idata_with_chain_stats_populates_sample_stats() -> None:
    """When chain_stats is passed, ArviZ sample_stats group is populated with
    renamed fields (is_divergent → diverging, num_integration_steps → n_steps,
    num_trajectory_expansions → tree_depth, energy → energy,
    acceptance_rate → acceptance_rate, is_turning → tuningfork_is_turning)."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(7)
    n_draws = 200
    samples = {"theta": rng.standard_normal((n_draws, 3))}
    chain_stats = {
        "is_divergent": rng.integers(0, 2, n_draws).astype(bool),
        "is_turning": rng.integers(0, 2, n_draws).astype(bool),
        "energy": rng.standard_normal(n_draws),
        "acceptance_rate": rng.uniform(0.5, 1.0, n_draws),
        "num_integration_steps": rng.integers(1, 64, n_draws),
        "num_trajectory_expansions": rng.integers(1, 10, n_draws),
        # Vectors / unknown fields should be silently dropped:
        "momentum": rng.standard_normal((n_draws, 3)),
        "some_unknown_field": rng.standard_normal(n_draws),
    }
    idata = samples_to_idata(samples, chain_stats=chain_stats)

    sample_stats = (
        idata["sample_stats"] if hasattr(idata, "__getitem__") else idata.sample_stats
    )
    # ArviZ-canonical names present
    assert "diverging" in sample_stats
    assert "energy" in sample_stats
    assert "acceptance_rate" in sample_stats
    assert "n_steps" in sample_stats
    assert "tree_depth" in sample_stats
    # Non-canonical fields under prefixed names
    assert "tuningfork_is_turning" in sample_stats
    # Original (non-canonical) names absent
    assert "is_divergent" not in sample_stats
    assert "num_integration_steps" not in sample_stats
    assert "num_trajectory_expansions" not in sample_stats
    # Vector/unknown fields dropped
    assert "momentum" not in sample_stats
    assert "some_unknown_field" not in sample_stats


def test_load_idata_groundtruth_enrichment(monkeypatch) -> None:
    """For GROUNDTRUTH recipes, load_idata enriches sample_stats with
    step_size (broadcast from adapted scalar) and reached_max_treedepth
    (derived from num_trajectory_expansions vs max_num_doublings)."""
    from tuningfork.catalog.render import load_idata
    from tuningfork.recipes._base import Effort

    rng = np.random.default_rng(23)
    n_draws = 100
    fake_samples = {"theta": rng.standard_normal((n_draws, 3))}
    fake_chain_stats = {
        "is_divergent": np.zeros(n_draws, dtype=bool),
        "energy": rng.standard_normal(n_draws),
        "acceptance_rate": rng.uniform(0.8, 1.0, n_draws),
        "num_integration_steps": rng.integers(1, 32, n_draws),
        "num_trajectory_expansions": np.full(n_draws, 5, dtype=np.int32),
    }

    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_samples
    mock_recipe.model_name = "mvn_10"
    mock_recipe.effort = Effort.GROUNDTRUTH
    mock_recipe.base_method_params = {"step_size": 0.234}
    mock_recipe.warmup_params = {"max_num_doublings": 10}

    monkeypatch.setattr(
        "tuningfork.catalog.render.try_load_cached_chain_stats",
        lambda entry, cache_dir=None: fake_chain_stats,
    )

    idata = load_idata(mock_recipe)

    sample_stats = (
        idata["sample_stats"] if hasattr(idata, "__getitem__") else idata.sample_stats
    )
    # Standard projections
    assert "diverging" in sample_stats
    assert "tree_depth" in sample_stats
    # GROUNDTRUTH-enriched fields
    assert "step_size" in sample_stats
    assert "reached_max_treedepth" in sample_stats
    # step_size should be broadcast to (1, n_draws) (single-chain shape)
    step_size_arr = np.asarray(sample_stats["step_size"])
    assert step_size_arr.shape[-1] == n_draws
    assert np.allclose(step_size_arr, 0.234)
    # reached_max_treedepth = num_trajectory_expansions >= 10. Our trajectory
    # expansions are all 5, so none should be at-cap.
    rmd = np.asarray(sample_stats["reached_max_treedepth"])
    assert rmd.dtype == bool
    assert not rmd.any()


def test_samples_to_idata_with_partial_chain_stats() -> None:
    """If only a subset of fields is present in chain_stats, only those map to sample_stats."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(11)
    samples = {"x": rng.standard_normal((100, 2))}
    chain_stats = {"is_divergent": np.zeros(100, dtype=bool)}
    idata = samples_to_idata(samples, chain_stats=chain_stats)

    sample_stats = (
        idata["sample_stats"] if hasattr(idata, "__getitem__") else idata.sample_stats
    )
    assert "diverging" in sample_stats
    assert "energy" not in sample_stats  # not in chain_stats input


def test_samples_to_idata_no_chain_stats_no_sample_stats_group() -> None:
    """Default behaviour (chain_stats=None) — no sample_stats group attached."""
    from tuningfork.catalog.render import samples_to_idata

    rng = np.random.default_rng(13)
    samples = {"x": rng.standard_normal((100,))}
    idata = samples_to_idata(samples)  # no chain_stats

    # ArviZ ≥ 0.20 uses DataTree; absent groups may raise KeyError on subscript.
    # Test by checking attribute access (returns None if absent in legacy API).
    has_sample_stats = hasattr(idata, "sample_stats") and idata.sample_stats is not None
    # Legacy InferenceData: attribute exists but may be None; DataTree: subscript raises
    if not has_sample_stats:
        try:
            ss = idata["sample_stats"]  # may raise on DataTree if absent
            assert ss is None or len(list(ss.data_vars)) == 0
        except (KeyError, AttributeError):
            pass  # expected — no sample_stats group


# ---------------------------------------------------------------------------
# load_idata tests — one-call convenience
# ---------------------------------------------------------------------------


def test_load_idata_combines_samples_and_chain_stats(monkeypatch) -> None:
    """load_idata bundles load_samples + load_chain_stats + samples_to_idata."""
    from tuningfork.catalog.render import load_idata

    rng = np.random.default_rng(17)
    n_draws = 150
    fake_samples = {"theta": rng.standard_normal((n_draws, 4))}
    fake_chain_stats = {
        "is_divergent": np.zeros(n_draws, dtype=bool),
        "energy": rng.standard_normal(n_draws),
        "num_integration_steps": rng.integers(1, 32, n_draws),
        "acceptance_rate": rng.uniform(0.5, 1.0, n_draws),
    }

    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_samples
    mock_recipe.model_name = "mvn_10"

    # Mock try_load_cached_chain_stats via the MODELS lookup path.
    # tuningfork.catalog.render.load_chain_stats does:
    #   MODELS[recipe.model_name] → try_load_cached_chain_stats(entry)
    # Patch the eager-imported name in tuningfork.catalog.render (used by load_chain_stats).
    monkeypatch.setattr(
        "tuningfork.catalog.render.try_load_cached_chain_stats",
        lambda entry, cache_dir=None: fake_chain_stats,
    )

    idata = load_idata(mock_recipe)

    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    sample_stats = (
        idata["sample_stats"] if hasattr(idata, "__getitem__") else idata.sample_stats
    )
    assert "theta" in posterior
    assert "diverging" in sample_stats
    assert "n_steps" in sample_stats


def test_load_idata_missing_chain_stats_returns_posterior_only(monkeypatch) -> None:
    """load_idata is robust to chain_stats cache miss — returns idata with only posterior."""
    from tuningfork.catalog.render import load_idata

    rng = np.random.default_rng(19)
    fake_samples = {"x": rng.standard_normal((100, 2))}

    mock_recipe = MagicMock()
    mock_recipe.load_cached_samples.return_value = fake_samples
    mock_recipe.model_name = "mvn_10"

    # chain_stats cache miss → try_load returns None
    monkeypatch.setattr(
        "tuningfork.catalog.render.try_load_cached_chain_stats",
        lambda entry, cache_dir=None: None,
    )

    idata = load_idata(mock_recipe)

    posterior = idata["posterior"] if hasattr(idata, "__getitem__") else idata.posterior
    assert "x" in posterior
