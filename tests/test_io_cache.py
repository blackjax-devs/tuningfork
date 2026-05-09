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
"""Tests for the reference cache I/O layer (bjx_bench.reference._io).

Tests:
- Round-trip write/load returns same draws.
- SHA mismatch triggers regeneration.
- Smaller num_samples than requested triggers regeneration.
- BJX_BENCH_REFERENCE_DIR env override works.
- get_reference_summaries loads from JSON.
- get_adaptation_params raises ValueError for analytic models.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bjx_bench.model import MODELS
from bjx_bench.reference._io import (
    _atomic_write_json,
    _atomic_write_npz,
    _metadata_path,
    _summaries_path,
    get_adaptation_params,
    get_reference_draws,
    get_reference_summaries,
)

pytestmark = pytest.mark.fast

MVN_ENTRY = MODELS["mvn_10"]
N_SMALL = 50  # small n for fast tests


class TestRoundTrip:
    """Cache round-trip: write then load returns the same draws."""

    def test_analytic_round_trip(self, tmp_path: Path) -> None:
        key = jax.random.key(42)
        draws = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path
        )
        assert "x" in draws
        assert draws["x"].shape == (N_SMALL, 10)

        # Second call: cache hit
        draws2 = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path
        )
        np.testing.assert_allclose(
            np.asarray(draws["x"]), np.asarray(draws2["x"]), rtol=1e-6
        )

    def test_draws_written_to_npz(self, tmp_path: Path) -> None:
        key = jax.random.key(1)
        get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path)
        npz_path = tmp_path / "draws" / "mvn_10.npz"
        assert npz_path.exists(), "draws npz not written"

    def test_metadata_written(self, tmp_path: Path) -> None:
        key = jax.random.key(2)
        get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path)
        meta_path = _metadata_path("mvn_10", tmp_path)
        assert meta_path.exists(), "metadata json not written"
        with meta_path.open() as fh:
            meta = json.load(fh)
        assert meta["generator"] == "analytic"
        assert meta["num_samples"] == N_SMALL
        assert meta["certification"]["passed"] is True

    def test_summaries_written(self, tmp_path: Path) -> None:
        key = jax.random.key(3)
        get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path)
        assert _summaries_path("mvn_10", tmp_path).exists()


class TestCacheInvalidation:
    """Cache hit/miss logic."""

    def test_sha_mismatch_triggers_regeneration(self, tmp_path: Path) -> None:
        key = jax.random.key(10)
        # Populate cache
        draws1 = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path
        )

        # Tamper with SHA in metadata → cache miss → regenerate with new key
        meta_path = _metadata_path("mvn_10", tmp_path)
        with meta_path.open() as fh:
            meta = json.load(fh)
        meta["code_sha"] = "deadbeef000"
        _atomic_write_json(meta_path, meta)

        key2 = jax.random.key(99)
        draws2 = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key2, cache_dir=tmp_path
        )
        # Draws should differ because a different key was used
        assert not np.allclose(
            np.asarray(draws1["x"]), np.asarray(draws2["x"])
        ), "Expected regeneration with different key"

    def test_smaller_n_triggers_regeneration(self, tmp_path: Path) -> None:
        key = jax.random.key(20)
        # Populate with 30 samples
        get_reference_draws(MVN_ENTRY, n=30, rng_key=key, cache_dir=tmp_path)

        # Tamper: set num_samples to 30 in metadata, then request 50
        meta_path = _metadata_path("mvn_10", tmp_path)
        with meta_path.open() as fh:
            meta = json.load(fh)
        assert meta["num_samples"] == 30

        # Request more than cached → regeneration
        draws2 = get_reference_draws(
            MVN_ENTRY, n=50, rng_key=jax.random.key(21), cache_dir=tmp_path
        )
        assert draws2["x"].shape == (50, 10)

    def test_force_regenerate_ignores_valid_cache(self, tmp_path: Path) -> None:
        key1 = jax.random.key(30)
        draws1 = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key1, cache_dir=tmp_path
        )

        key2 = jax.random.key(31)
        draws2 = get_reference_draws(
            MVN_ENTRY,
            n=N_SMALL,
            rng_key=key2,
            force_regenerate=True,
            cache_dir=tmp_path,
        )
        # Different keys → different draws
        assert not np.allclose(np.asarray(draws1["x"]), np.asarray(draws2["x"]))

    def test_version_mismatch_triggers_regeneration(self, tmp_path: Path) -> None:
        key = jax.random.key(40)
        get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path)

        # Tamper with version
        meta_path = _metadata_path("mvn_10", tmp_path)
        with meta_path.open() as fh:
            meta = json.load(fh)
        meta["bjx_bench_version"] = "99.99.99"
        _atomic_write_json(meta_path, meta)

        # Should regenerate
        draws2 = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=jax.random.key(41), cache_dir=tmp_path
        )
        assert draws2["x"].shape == (N_SMALL, 10)


class TestEnvOverride:
    """BJX_BENCH_REFERENCE_DIR env variable override."""

    def test_env_override_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BJX_BENCH_REFERENCE_DIR", str(tmp_path))
        key = jax.random.key(50)
        # cache_dir=None → should read from env
        draws = get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key)
        assert draws["x"].shape == (N_SMALL, 10)
        assert (tmp_path / "draws" / "mvn_10.npz").exists()
        monkeypatch.delenv("BJX_BENCH_REFERENCE_DIR")

    def test_explicit_cache_dir_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alt_dir = tmp_path / "alt"
        alt_dir.mkdir()
        # Set env to something different
        monkeypatch.setenv("BJX_BENCH_REFERENCE_DIR", str(tmp_path / "from_env"))
        key = jax.random.key(51)
        draws = get_reference_draws(
            MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=alt_dir
        )
        assert draws["x"].shape == (N_SMALL, 10)
        # artifact written in alt_dir, not env dir
        assert (alt_dir / "draws" / "mvn_10.npz").exists()
        monkeypatch.delenv("BJX_BENCH_REFERENCE_DIR")


class TestSummariesAndAdaptation:
    """get_reference_summaries and get_adaptation_params."""

    def test_get_reference_summaries(self, tmp_path: Path) -> None:
        key = jax.random.key(60)
        get_reference_draws(MVN_ENTRY, n=N_SMALL, rng_key=key, cache_dir=tmp_path)
        summaries = get_reference_summaries(MVN_ENTRY, cache_dir=tmp_path)
        assert "x" in summaries.mean
        assert summaries.n_samples == N_SMALL

    def test_get_summaries_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_reference_summaries(
                MVN_ENTRY, cache_dir=tmp_path, auto_regenerate=False
            )

    def test_get_adaptation_params_raises_for_analytic(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="analytic path"):
            get_adaptation_params(MVN_ENTRY, cache_dir=tmp_path)

    def test_get_adaptation_params_raises_missing_cache(self, tmp_path: Path) -> None:
        eight_schools = MODELS["eight_schools_ncp"]
        with pytest.raises(FileNotFoundError):
            get_adaptation_params(eight_schools, cache_dir=tmp_path)


class TestAtomicHelpers:
    """Unit tests for atomic write helpers."""

    def test_atomic_write_json(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "test.json"
        _atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert data == {"a": 1, "b": [1, 2, 3]}

    def test_atomic_write_npz(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "test.npz"
        arr = np.random.randn(5, 3)
        _atomic_write_npz(path, {"x": arr})
        assert path.exists()
        loaded = np.load(str(path))
        np.testing.assert_allclose(loaded["x"], arr)
