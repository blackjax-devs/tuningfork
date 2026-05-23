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
"""Fast tests for tuningfork.catalog._rerun_inference cache round-trip.

Regression guard for the 2026-05-23 bug where the cache stored canonical
ArviZ sample_stats names (``diverging``, ``n_steps``) but the loader
re-passed them through ``samples_to_idata``'s rename projection — which
expects RAW blackjax names on the LHS (``is_divergent``,
``num_integration_steps``) and silently dropped the already-canonical
keys. Result: every cache-hit idata was missing ``diverging`` and
``n_steps`` across all sampler families (nuts/hmc/dynamic_hmc).

Test isolates ``_load_from_cache`` (no JAX, no sampler invocation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fast


def _write_canonical_cache(tmp_path: Path) -> tuple[Path, Path]:
    """Write a fake cache file pair using ArviZ canonical sample_stats names.

    Mimics what ``_save_to_cache`` produces after a fresh sampler run:
    posterior keys are param names, sample_stats keys are
    ``diverging``/``energy``/``acceptance_rate``/``n_steps``.
    """
    n_chains, n_draws = 4, 100
    draws_path = tmp_path / "stem.draws.npz"
    stats_path = tmp_path / "stem.chain_stats.npz"

    rng = np.random.default_rng(0)
    np.savez_compressed(
        str(draws_path),
        mu=rng.standard_normal((n_chains, n_draws)),
        tau=np.exp(rng.standard_normal((n_chains, n_draws))),
    )
    np.savez_compressed(
        str(stats_path),
        diverging=rng.integers(0, 2, (n_chains, n_draws)).astype(bool),
        energy=rng.standard_normal((n_chains, n_draws)),
        acceptance_rate=rng.uniform(0.0, 1.0, (n_chains, n_draws)),
        n_steps=rng.integers(1, 20, (n_chains, n_draws)).astype(np.int32),
    )
    return draws_path, stats_path


def test_load_from_cache_preserves_canonical_sample_stats(tmp_path: Path) -> None:
    """Cache files use canonical names; load_from_cache must round-trip them.

    Regression guard for the rename-asymmetry bug fixed 2026-05-23: the
    cache writer saves canonical names but the cache reader was passing
    them through a rename map that expects raw blackjax names, silently
    dropping ``diverging`` and ``n_steps``.
    """
    from tuningfork.catalog._rerun_inference import _load_from_cache

    draws_path, stats_path = _write_canonical_cache(tmp_path)
    idata = _load_from_cache(draws_path, stats_path)

    # Posterior round-trips
    assert set(idata.posterior.data_vars) == {"mu", "tau"}

    # All four canonical sample_stats keys survive
    stat_vars = set(idata.sample_stats.data_vars)
    expected = {"diverging", "energy", "acceptance_rate", "n_steps"}
    missing = expected - stat_vars
    assert not missing, (
        f"Cache load dropped canonical sample_stats keys: {missing}. "
        f"Got: {sorted(stat_vars)}"
    )


def test_load_from_cache_passes_unknown_keys_through(tmp_path: Path) -> None:
    """Unknown sample_stats keys (future schema additions) shouldn't be dropped.

    ``samples_to_idata``'s rename projection drops keys not in the map.
    The reverse-map in ``_load_from_cache`` should fall back to identity
    on unknown keys so they reach ``samples_to_idata`` under their cached
    name; if ``samples_to_idata`` drops them, that's an out-of-scope concern
    for this round-trip guard. This test asserts the reverse-map does the
    right thing for canonical keys (the original bug) and doesn't crash
    on unknown ones.
    """
    from tuningfork.catalog._rerun_inference import _load_from_cache

    draws_path, stats_path = _write_canonical_cache(tmp_path)
    # Add an unknown key to the stats cache
    stats_data = dict(np.load(str(stats_path)))
    stats_data["future_field_not_in_map"] = np.zeros((4, 100))
    np.savez_compressed(str(stats_path), **stats_data)

    # Must not raise
    idata = _load_from_cache(draws_path, stats_path)
    # The canonical keys still survive
    assert "diverging" in idata.sample_stats.data_vars
    assert "n_steps" in idata.sample_stats.data_vars
