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

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
    """Unknown sample_stats keys survive the cache round-trip unchanged."""
    from tuningfork.catalog._rerun_inference import _load_from_cache

    draws_path, stats_path = _write_canonical_cache(tmp_path)
    # Add an unknown key to the stats cache
    stats_data = dict(np.load(str(stats_path)))
    stats_data["future_field_not_in_map"] = np.zeros((4, 100))
    np.savez_compressed(str(stats_path), **stats_data)

    idata = _load_from_cache(draws_path, stats_path)
    assert "diverging" in idata.sample_stats.data_vars
    assert "n_steps" in idata.sample_stats.data_vars
    assert "future_field_not_in_map" in idata.sample_stats.data_vars


def test_load_from_cache_rejects_corrupt_stats_instead_of_dropping_them(
    tmp_path: Path,
) -> None:
    from tuningfork.catalog._rerun_inference import _load_from_cache

    draws_path, stats_path = _write_canonical_cache(tmp_path)
    stats_path.write_bytes(b"not an npz archive")
    with pytest.raises(ValueError, match="Could not load chain-stats cache"):
        _load_from_cache(draws_path, stats_path)


def test_load_from_cache_rejects_stat_name_collisions(tmp_path: Path) -> None:
    from tuningfork.catalog._rerun_inference import _load_from_cache

    draws_path, stats_path = _write_canonical_cache(tmp_path)
    np.savez(
        stats_path,
        diverging=np.zeros((4, 100), dtype=bool),
        is_divergent=np.ones((4, 100), dtype=bool),
    )
    with pytest.raises(ValueError, match="collide"):
        _load_from_cache(draws_path, stats_path)


# ---------------------------------------------------------------------------
# regenerate_idata: unit tests (no JAX; mock generated execution)
# ---------------------------------------------------------------------------


def _failed_recipe():
    from tuningfork.recipes._base import Effort, Recipe

    return Recipe(
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


def test_regenerate_idata_is_exported() -> None:
    """regenerate_idata is accessible from the catalog public API."""
    from tuningfork.catalog import load_generated_idata, regenerate_idata

    assert callable(regenerate_idata)
    assert callable(load_generated_idata)


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


def test_regenerate_idata_executes_generated_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execution receives a copied recipe and a durable generated-run root."""
    import tuningfork.catalog._rerun_inference as _mod

    calls: list[tuple[object, Path, dict]] = []
    artifact = tmp_path / "artifact.npz"
    artifact.write_bytes(b"placeholder")
    expected = object()

    def _fake_execute(r, run_root, **kw):
        calls.append((r, run_root, kw))
        return SimpleNamespace(artifact_path=artifact)

    monkeypatch.setattr("tuningfork.catalog.emit.execute_recipe", _fake_execute)
    monkeypatch.setattr(_mod, "load_generated_idata", lambda path: expected)

    recipe = _failed_recipe()

    result = _mod.regenerate_idata(
        recipe, n_samples=200, seed=42, catalog_root=tmp_path
    )

    assert len(calls) == 1, f"Expected exactly one call, got {len(calls)}"
    copied, run_root, kwargs = calls[0]
    assert result is expected
    assert getattr(copied, "tuning_seed") == 42
    assert recipe.tuning_seed != 42
    assert kwargs == {"num_samples": 200}
    assert run_root == tmp_path / recipe.model_name / "_cache" / "generated_runs"


def test_regenerate_idata_propagates_generated_program_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuningfork.catalog._rerun_inference as _mod
    from tuningfork.recipes._launcher import GeneratedProgramError

    recipe = _failed_recipe()
    receipt = object()
    result_obj = cast(Any, SimpleNamespace(receipt_path=receipt))
    error = GeneratedProgramError("failed", result_obj)

    def fail_execution(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "tuningfork.catalog.emit.execute_recipe",
        fail_execution,
    )
    with pytest.raises(GeneratedProgramError) as caught:
        _mod.regenerate_idata(recipe, catalog_root=tmp_path)
    assert caught.value is error
    assert caught.value.result is error.result
    assert caught.value.receipt_path is receipt


def test_regenerate_idata_rejects_missing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuningfork.catalog._rerun_inference as _mod

    recipe = _failed_recipe()
    monkeypatch.setattr(
        "tuningfork.catalog.emit.execute_recipe",
        lambda *args, **kwargs: SimpleNamespace(artifact_path=None),
    )
    with pytest.raises(RuntimeError, match="without a verified artifact"):
        _mod.regenerate_idata(recipe, catalog_root=tmp_path)


def test_regenerate_idata_keeps_groundtruth_as_an_lfs_backed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tuningfork.catalog._rerun_inference as _mod

    expected = object()
    recipe = SimpleNamespace(
        effort=SimpleNamespace(value="groundtruth"),
        model_name="mvn_10",
    )
    calls = []

    def fake_load(value, *, cache_dir):
        calls.append((value, cache_dir))
        return expected

    monkeypatch.setattr("tuningfork.catalog.render.load_idata", fake_load)
    monkeypatch.setattr(
        "tuningfork.catalog.emit.execute_recipe",
        lambda *args, **kwargs: pytest.fail("groundtruth load must not launch"),
    )

    assert _mod.regenerate_idata(recipe, catalog_root=tmp_path) is expected
    assert calls == [(recipe, tmp_path)]


def test_cached_force_regeneration_uses_generated_execution_and_recipe_sample_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced cache population goes through codegen with the pinned draw count."""
    import tuningfork.catalog._rerun_inference as _mod
    from tuningfork.recipes._base import Effort

    recipe = _failed_recipe()
    recipe = replace(
        recipe,
        effort=Effort.LOW,
        calibration_budget={"n_samples": 37, "num_chains": 3},
        tuning_seed=73,
    )
    calls: list[tuple[object, Path, dict[str, object]]] = []
    expected = object()

    def _execute(r, run_root, **kwargs):
        calls.append((r, run_root, kwargs))
        return SimpleNamespace(artifact_path=tmp_path / "generated.npz")

    monkeypatch.setattr(
        "tuningfork.catalog.emit.execute_recipe",
        _execute,
    )
    (tmp_path / "generated.npz").write_bytes(b"placeholder")
    monkeypatch.setattr(_mod, "load_generated_idata", lambda path: expected)
    monkeypatch.setattr(_mod, "_save_to_cache", lambda *args: None)
    monkeypatch.setattr(_mod, "_write_cache_params_sidecar", lambda *args: None)

    assert (
        _mod.cached_idata_for_recipe(
            recipe, catalog_root=tmp_path, force_regenerate=True
        )
        is expected
    )
    assert len(calls) == 1
    copied, run_root, kwargs = calls[0]
    assert getattr(copied, "tuning_seed") == recipe.tuning_seed
    assert run_root == tmp_path / recipe.model_name / "_cache" / "generated_runs"
    assert kwargs == {"num_samples": 37}


def test_cached_force_regeneration_preserves_zero_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed zero is an executable value, not an absent configuration."""
    import tuningfork.catalog._rerun_inference as _mod
    from tuningfork.recipes._base import Effort

    recipe = replace(_failed_recipe(), effort=Effort.LOW, tuning_seed=0)
    calls: list[dict[str, object]] = []

    def _regenerate(_recipe, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        _mod,
        "regenerate_idata",
        _regenerate,
    )
    monkeypatch.setattr(_mod, "_save_to_cache", lambda *args: None)
    monkeypatch.setattr(_mod, "_write_cache_params_sidecar", lambda *args: None)

    _mod.cached_idata_for_recipe(recipe, catalog_root=tmp_path, force_regenerate=True)

    assert calls[0]["seed"] == 0


def test_cached_groundtruth_force_is_load_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Groundtruth always loads its LFS artifact, ignoring cache regeneration."""
    import tuningfork.catalog._rerun_inference as _mod

    expected = object()
    recipe = SimpleNamespace(
        effort=SimpleNamespace(value="groundtruth"), model_name="mvn_10"
    )
    calls: list[tuple[object, Path]] = []

    def _load(value, *, cache_dir):
        calls.append((value, cache_dir))
        return expected

    monkeypatch.setattr(
        "tuningfork.catalog.render.load_idata",
        _load,
    )
    monkeypatch.setattr(
        "tuningfork.catalog.emit.execute_recipe",
        lambda *args, **kwargs: pytest.fail(
            "groundtruth must not execute generated code"
        ),
    )

    assert (
        _mod.cached_idata_for_recipe(
            recipe, catalog_root=tmp_path, force_regenerate=True
        )
        is expected
    )
    assert calls == [(recipe, tmp_path)]


def test_artifact_to_idata_preserves_posterior_and_stats(tmp_path: Path) -> None:
    from tuningfork.catalog._rerun_inference import _artifact_to_idata

    path = tmp_path / "draws.npz"
    np.savez(
        path,
        position=np.zeros((2, 3, 1)),
        _ss_is_divergent=np.array([[False, True, False], [False, False, True]]),
        _ss_num_integration_steps=np.arange(6).reshape(2, 3),
        _ss_energy=np.ones((2, 3)),
        _ss_future=np.ones((2, 3, 2)),
        _ss_negative=np.full((2, 3), -1),
    )
    idata = _artifact_to_idata(path)
    assert "position" in idata.posterior
    assert {"diverging", "n_steps", "energy", "future", "negative"} <= set(
        idata.sample_stats.data_vars
    )
    assert np.asarray(idata.sample_stats["future"]).shape == (2, 3, 2)
    assert np.all(np.asarray(idata.sample_stats["negative"]) == -1)


def test_artifact_to_idata_supports_posterior_without_stats(tmp_path: Path) -> None:
    from tuningfork.catalog._rerun_inference import _artifact_to_idata

    path = tmp_path / "draws.npz"
    np.savez(path, position=np.zeros((1, 2, 1)))
    idata = _artifact_to_idata(path)
    assert "position" in idata.posterior
    assert not hasattr(idata, "sample_stats")


def test_artifact_to_idata_rejects_canonical_stat_collision(tmp_path: Path) -> None:
    from tuningfork.catalog._rerun_inference import _artifact_to_idata

    path = tmp_path / "draws.npz"
    np.savez(
        path,
        position=np.zeros((1, 2, 1)),
        _ss_is_divergent=np.zeros((1, 2), dtype=bool),
        _ss_diverging=np.ones((1, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="both map.*diverging"):
        _artifact_to_idata(path)


@pytest.mark.parametrize(
    "arrays, message",
    [
        ({"_ss_energy": np.zeros((1, 2))}, "no posterior"),
        ({"position": np.zeros((1, 2)), "_ss_": np.zeros((1, 2))}, "empty statistic"),
    ],
)
def test_artifact_to_idata_rejects_invalid_artifacts(
    tmp_path: Path, arrays: dict[str, np.ndarray], message: str
) -> None:
    from tuningfork.catalog._rerun_inference import _artifact_to_idata

    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        _artifact_to_idata(path)


@pytest.mark.parametrize(
    "arrays, message",
    [
        ({"position": np.zeros(())}, "posterior variable"),
        (
            {"position": np.zeros((1, 2, 1)), "other": np.zeros((2, 2, 1))},
            "inconsistent leading shapes",
        ),
        (
            {"position": np.zeros((1, 2, 1)), "_ss_energy": np.zeros(())},
            "statistic",
        ),
        (
            {"position": np.zeros((1, 2, 1)), "_ss_energy": np.zeros((1, 3))},
            "expected",
        ),
    ],
)
def test_artifact_to_idata_rejects_invalid_leading_shapes(
    tmp_path: Path, arrays: dict[str, np.ndarray], message: str
) -> None:
    from tuningfork.catalog._rerun_inference import _artifact_to_idata

    path = tmp_path / "bad-shape.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message):
        _artifact_to_idata(path)


def test_regenerate_idata_default_values() -> None:
    """regenerate_idata has sane defaults for n_samples, seed, catalog_root."""
    import inspect

    from tuningfork.catalog._rerun_inference import regenerate_idata

    sig = inspect.signature(regenerate_idata)
    params = sig.parameters
    assert params["n_samples"].default == 1000
    assert params["seed"].default == 20260517
    assert params["catalog_root"].default is None


def test_cache_params_sidecar_roundtrips_to_path_a(tmp_path: Path) -> None:
    """Fix 2 ↔ Fix 1 round-trip: sidecar written at cache-time → reader keeps path-A.

    Regression guard for issue #244 write side.  ``_write_cache_params_sidecar``
    derives the sidecar from the same on-disk recipe JSON that
    ``classify_recipe_path`` inspects, so a freshly cached recipe must classify
    as "A" (trusted cache), and a subsequent param re-emit must degrade it.
    """
    import json

    from tuningfork.calibration.revalidation import classify_recipe_path
    from tuningfork.catalog._rerun_inference import _write_cache_params_sidecar

    model = "german_credit"
    stem = "medium__hmc__window_adaptation_diag_imm"
    recipe: dict = {
        "gate_evidence": {"auto": {"verdict": "PASS"}},
        "base_method_name": "hmc",
        "base_method_params": {"step_size": 0.2915, "num_integration_steps": 8},
        "warmup_params": {"target_acceptance": 0.8},
        "calibration_budget": {"num_chains": 4},
    }
    # Catalog layout: <root>/<model>/{recipes,_cache,groundtruth_samples/blackjax}
    recipe_path = tmp_path / model / "recipes" / f"{stem}.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(recipe))

    gt = tmp_path / model / "groundtruth_samples" / "blackjax"
    gt.mkdir(parents=True)
    (gt / "draws.npz").write_bytes(b"")
    (gt / "summary_v2.json").write_text("{}")

    cache_dir = tmp_path / model / "_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{stem}.draws.npz").write_bytes(b"")  # freshly written draws

    # Fix 2: co-write the params sidecar from the on-disk recipe JSON.
    _write_cache_params_sidecar(cache_dir, tmp_path, model, stem)
    sidecar = cache_dir / f"{stem}.params_hash.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == {
        "step_size": 0.2915,
        "num_integration_steps": 8,
        "target_acceptance": 0.8,
        "imm_l2_norm": None,
    }

    # Fix 1 reader agrees: trusted cache → path A.
    assert classify_recipe_path(recipe_path) == "A"

    # Re-emit with new params (L=8→L=12): sidecar now stale → degrade to B.
    recipe["base_method_params"]["num_integration_steps"] = 12
    recipe_path.write_text(json.dumps(recipe))
    assert classify_recipe_path(recipe_path) == "B"


def test_cache_params_sidecar_noop_when_recipe_missing(tmp_path: Path) -> None:
    """No recipe JSON on disk → sidecar write is a silent no-op (safe degrade)."""
    from tuningfork.catalog._rerun_inference import _write_cache_params_sidecar

    model = "mvn_10"
    stem = "low__nuts__window_adaptation_diag_imm"
    cache_dir = tmp_path / model / "_cache"
    cache_dir.mkdir(parents=True)

    _write_cache_params_sidecar(cache_dir, tmp_path, model, stem)
    assert not (cache_dir / f"{stem}.params_hash.json").exists()
