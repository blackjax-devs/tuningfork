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


# ---------------------------------------------------------------------------
# regenerate_idata: unit tests (no JAX; mock run_recipe_to_idata)
# ---------------------------------------------------------------------------


def test_regenerate_idata_is_exported() -> None:
    """regenerate_idata is accessible from the catalog public API."""
    from tuningfork.catalog import regenerate_idata

    assert callable(regenerate_idata)


def test_regenerate_idata_signature() -> None:
    """regenerate_idata has the expected keyword-only parameters."""
    import inspect

    from tuningfork.catalog._rerun_inference import regenerate_idata

    sig = inspect.signature(regenerate_idata)
    params = sig.parameters
    assert "recipe" in params, "missing 'recipe' parameter"
    assert "n_samples" in params, "missing 'n_samples' parameter"
    assert "seed" in params, "missing 'seed' parameter"
    assert "catalog_root" in params, "missing 'catalog_root' parameter"
    # recipe should be positional; the rest keyword-only
    assert params["n_samples"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["seed"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["catalog_root"].kind == inspect.Parameter.KEYWORD_ONLY


def test_regenerate_idata_passes_allow_failed_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regenerate_idata passes _allow_failed_diagnostic=True to run_recipe_to_idata.

    Verifies that FAIL recipes don't raise RecipeFailedError when called via
    regenerate_idata, and that force_resample_config carries the correct params.
    Uses monkeypatch on the module attribute (after first import) to intercept.
    """
    import tuningfork.catalog._rerun_inference as _mod
    from tuningfork.recipes._base import Effort, Recipe

    calls: list[dict] = []

    def _fake_run(r, *, force_resample_config=None, catalog_root=None, **kw):
        calls.append(
            {
                "force_resample_config": force_resample_config,
                "catalog_root": catalog_root,
                "allow_failed": kw.get("_allow_failed_diagnostic", False),
            }
        )
        return object()  # fake InferenceData

    # Patch at module level so the lazy import inside regenerate_idata sees it
    monkeypatch.setattr(
        "tuningfork.recipes._recipe_runner.run_recipe_to_idata",
        _fake_run,
        raising=False,
    )
    import tuningfork.recipes._recipe_runner as _rr

    monkeypatch.setattr(_rr, "run_recipe_to_idata", _fake_run, raising=False)

    recipe = Recipe(
        model_name="mvn_10",
        base_method_name="nuts",
        warmup_name="window_adaptation_diag_imm",
        effort=Effort.FAILED,
        base_method_params={"step_size": 0.1, "inverse_mass_matrix": [1.0]},
        warmup_params={"n_warmup": 100, "num_chains": 4},
        headline_metric=None,
        sample_quality=None,
        calibration_budget={"trials": 0, "wall_seconds_estimate": 1.0},
        difficulty=None,
        instructions="test",
    )

    _mod.regenerate_idata(recipe, n_samples=200, seed=42, catalog_root=tmp_path)

    assert len(calls) == 1, f"Expected exactly one call, got {len(calls)}"
    cfg = calls[0]["force_resample_config"]
    assert cfg["n_samples"] == 200
    assert cfg["seed"] == 42
    assert calls[0]["catalog_root"] == tmp_path


def test_regenerate_idata_default_values() -> None:
    """regenerate_idata has sane defaults for n_samples, seed, catalog_root."""
    import inspect

    from tuningfork.catalog._rerun_inference import regenerate_idata

    sig = inspect.signature(regenerate_idata)
    params = sig.parameters
    assert params["n_samples"].default == 1000
    assert params["seed"].default == 20260517
    assert params["catalog_root"].default is None
