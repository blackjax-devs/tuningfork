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

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_NUTS_RECIPE = (
    _REPO_ROOT
    / "tuningfork/catalog/mvn_10/recipes/low__nuts__window_adaptation_diag_imm.json"
)
_DYNAMIC_HMC_RECIPE = (
    _REPO_ROOT / "tuningfork/catalog/mvn_10/recipes/"
    "low__dynamic_hmc__window_adaptation_diag_imm.json"
)
_HMC_RECIPE = (
    _REPO_ROOT
    / "tuningfork/catalog/eight_schools_ncp/recipes/low__hmc__window_adaptation_diag_imm.json"
)


def _load(path: Path):
    from tuningfork.catalog.inspect import load_recipe

    if not path.exists():
        pytest.skip(f"Recipe not found on disk: {path}")
    return load_recipe(path)


def _tap_jsonl(root: Path) -> list[Path]:
    return list((root / "tuningfork-tap-diagnostics").glob("*.jsonl"))


@pytest.mark.e2e
def test_generated_execute_recipe_diagnostics_toggle(tmp_path):
    """Tap changes instrumentation, never the generated sampling program."""
    from tuningfork.catalog import execute_recipe

    recipe = _load(_HMC_RECIPE)
    off_root = tmp_path / "off"
    on_root = tmp_path / "on"
    off = execute_recipe(
        recipe, off_root, num_samples=3, num_chains=1, diagnostics=False, timeout=120
    )
    on = execute_recipe(
        recipe,
        on_root,
        num_samples=3,
        num_chains=1,
        diagnostics=True,
        env={"TMPDIR": str(on_root)},
        timeout=120,
    )
    assert off.artifact_path is not None and on.artifact_path is not None
    assert not _tap_jsonl(off_root)
    jsonl = _tap_jsonl(on_root)
    assert jsonl, "diagnostics=True generated run produced no JSONL evidence"
    assert on.timings is not None
    assert off.source_sha256 == on.source_sha256
    assert off.manifest == on.manifest
    assert off.receipt.status == on.receipt.status == "success"
    with (
        np.load(off.artifact_path, allow_pickle=False) as off_draws,
        np.load(on.artifact_path, allow_pickle=False) as on_draws,
    ):
        assert set(off_draws.files) == set(on_draws.files)
        for name in off_draws.files:
            # jaxtap rewrites JAX loops with an extra carry and callbacks.
            # HMC trajectories are chaotic enough that the changed floating-point
            # schedule need not remain bitwise-identical across instrumentation
            # modes.  The executable plan and artifact contract must remain exact.
            assert off_draws[name].shape == on_draws[name].shape
            assert off_draws[name].dtype == on_draws[name].dtype
            assert np.all(np.isfinite(off_draws[name]))
            assert np.all(np.isfinite(on_draws[name]))


@pytest.mark.e2e
def test_generated_nuts_diagnostics_emits_treedepth_jsonl(tmp_path):
    """Generated NUTS execution emits output events for treedepth diagnostics."""
    import dataclasses

    from jaxtap import read_jsonl

    from tuningfork.catalog import execute_recipe
    from tuningfork.diagnostics._tap import compute_saturation_fraction

    base_recipe = _load(_NUTS_RECIPE)
    recipe = dataclasses.replace(
        base_recipe,
        base_method_params={
            **base_recipe.base_method_params,
            "max_num_doublings": 1,
        },
    )
    run_root = tmp_path / "nuts"
    result = execute_recipe(
        recipe,
        run_root,
        num_samples=3,
        num_chains=1,
        diagnostics=True,
        env={"TMPDIR": str(run_root)},
        timeout=120,
    )
    assert result.artifact_path is not None
    files = _tap_jsonl(run_root)
    assert files
    events = read_jsonl(files[0])
    outputs = [e for e in events if getattr(e, "kind", "carry") == "output"]
    assert outputs
    assert compute_saturation_fraction(files[0], max_num_doublings=1)[0] > 0


@pytest.mark.e2e
def test_generated_nuts_default_cap_has_zero_saturation(tmp_path):
    """Healthy generated NUTS does not falsely saturate its default cap."""
    from tuningfork.catalog import execute_recipe
    from tuningfork.diagnostics._tap import compute_saturation_fraction

    run_root = tmp_path / "nuts_healthy"
    result = execute_recipe(
        _load(_NUTS_RECIPE),
        run_root,
        num_samples=3,
        num_chains=1,
        diagnostics=True,
        env={"TMPDIR": str(run_root)},
        timeout=120,
    )
    assert result.artifact_path is not None
    files = _tap_jsonl(run_root)
    assert files
    sat_n, total, fraction = compute_saturation_fraction(files[0], max_num_doublings=10)
    assert total > 0 and sat_n == 0 and fraction == 0.0


@pytest.mark.e2e
def test_generated_dynamic_hmc_has_no_treedepth_alerts(tmp_path):
    """Generated HMCInfo observations never become treedepth alerts."""
    from jaxtap import read_jsonl

    from tuningfork.catalog import execute_recipe
    from tuningfork.diagnostics._tap import compute_saturation_fraction

    run_root = tmp_path / "dynamic_hmc"
    result = execute_recipe(
        _load(_DYNAMIC_HMC_RECIPE),
        run_root,
        num_samples=3,
        num_chains=1,
        diagnostics=True,
        env={"TMPDIR": str(run_root)},
        timeout=120,
    )
    assert result.artifact_path is not None
    files = _tap_jsonl(run_root)
    assert files
    events = read_jsonl(files[0])
    assert any(getattr(e, "kind", "carry") != "output" for e in events)
    assert not [e for e in events if getattr(e, "kind", "carry") == "output"]
    assert compute_saturation_fraction(files[0], max_num_doublings=10) == (0, 0, 0.0)


@pytest.mark.fast
def test_speed_benchmark_calls_disable_diagnostics():
    """Benchmark call sites explicitly disable generated diagnostics."""
    import ast

    for path in (_REPO_ROOT / "benchmarks/_benchmark_helpers.py",):
        tree = ast.parse(path.read_text(), filename=str(path))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "execute_recipe"
        ]
        assert calls, f"{path}: no execute_recipe benchmark calls found"
        for call in calls:
            kw = {item.arg: item.value for item in call.keywords}
            assert isinstance(kw.get("diagnostics"), ast.Constant)
            assert kw["diagnostics"].value is False


@pytest.mark.fast
def test_artifact_dir_env_var(tmp_path, monkeypatch):
    """Tap artifact directory and disabled environment values remain stable."""
    from tuningfork.diagnostics._tap import is_tap_enabled, tap_artifact_dir

    custom = tmp_path / "tap"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(custom))
    assert is_tap_enabled() and tap_artifact_dir() == custom
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", "0")
    assert not is_tap_enabled()
    monkeypatch.delenv("TUNINGFORK_TAP_DIAGNOSTICS")
    assert not is_tap_enabled()


@pytest.mark.fast
def test_unknown_and_non_nuts_tap_paths_fail_closed():
    """Unknown samplers are skipped and HMCInfo methods never arm treedepth."""
    from tuningfork.diagnostics._tap import _NUTS_FAMILY, is_algorithm_tap_compatible

    assert not is_algorithm_tap_compatible("future_sampler")
    assert "dynamic_hmc" not in _NUTS_FAMILY
    assert "nuts" in _NUTS_FAMILY


@pytest.mark.slow
def test_synthetic_scan_nan_attribution(tmp_path):
    """Synthetic pathology retains negative-path step attribution coverage."""
    from jaxtap import read_jsonl

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    onset = 4
    artifact = tmp_path / "synthetic.jsonl"

    def body(carry, step):
        dep = jax.lax.cond(
            step >= onset, lambda: jnp.float32(1), lambda: jnp.float32(0)
        )
        chol = jnp.linalg.cholesky(
            jnp.array([[1.0, dep], [dep, 1.0]], dtype=jnp.float32)
        )
        return (
            carry + jnp.float32(0) * jnp.sum(jnp.where(jnp.isfinite(chol), chol, 0)),
            0.0,
        )

    with tap_diagnostics_context(artifact_path=artifact, sample_every=1):
        jax.lax.scan(body, jnp.float32(0), jnp.arange(8, dtype=jnp.int32))
    events = read_jsonl(artifact)
    bad = [
        e
        for e in events
        if "cholesky" in str(e.path) and not bool(np.asarray(e.value).all())
    ]
    assert bad and min(e.step for e in bad) >= onset


@pytest.mark.slow
def test_mclmc_divergence_y_tap_negative_path(tmp_path):
    """Synthetic MCLMC cliff retains divergence-alert negative-path coverage."""
    import blackjax
    from jaxtap import read_jsonl

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    key, run_key = jax.random.split(jax.random.key(7))

    def logdensity(position):
        x = jnp.ravel(position)
        return jax.lax.cond(x[0] > 0, lambda: -jnp.inf, lambda: -0.5 * jnp.sum(x**2))

    state = blackjax.mclmc.init(jnp.zeros(3), logdensity, key)
    artifact = tmp_path / "mclmc.jsonl"
    with tap_diagnostics_context(
        artifact_path=artifact, base_method_name="mclmc", sample_every=1
    ):
        blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=blackjax.mclmc.build_kernel(),
            num_steps=20,
            state=state,
            rng_key=run_key,
            logdensity_fn=logdensity,
        )
    outputs = [
        e for e in read_jsonl(artifact) if getattr(e, "kind", "carry") == "output"
    ]
    assert outputs and any(bool(np.asarray(e.value).any()) for e in outputs)


@pytest.mark.slow
def test_mclmc_healthy_y_tap_has_no_divergence_alerts(tmp_path):
    """Healthy MCLMC adaptation emits observations without divergence flags."""
    import blackjax
    from jaxtap import read_jsonl

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    key, run_key = jax.random.split(jax.random.key(42))

    def logdensity(position):
        return -0.5 * jnp.sum(jnp.ravel(position) ** 2)

    artifact = tmp_path / "mclmc_healthy.jsonl"
    state = blackjax.mclmc.init(jnp.zeros(3), logdensity, key)
    with tap_diagnostics_context(
        artifact_path=artifact, base_method_name="mclmc", sample_every=1
    ):
        blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=blackjax.mclmc.build_kernel(),
            num_steps=20,
            state=state,
            rng_key=run_key,
            logdensity_fn=logdensity,
        )
    outputs = [
        e for e in read_jsonl(artifact) if getattr(e, "kind", "carry") == "output"
    ]
    assert outputs and not any(bool(np.asarray(e.value).any()) for e in outputs)
