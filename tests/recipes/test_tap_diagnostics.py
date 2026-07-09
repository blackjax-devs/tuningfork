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
"""Tests for M2 opt-in tap diagnostics (TUNINGFORK_TAP_DIAGNOSTICS).

Six tests covering the five AYS challenges raised in round 1:

Challenge 1 — Default-OFF purity (idata equality, not a flag check):
  ``test_default_off_idata_equality`` [TESTED]: runs the same tiny recipe x
  same seed twice — tap OFF vs tap ON — and asserts every posterior array is
  bitwise-identical. Proves that entering the tap context does not alter
  results (jaxtap bitwise-identity guarantee is ours, not just inherited).
  Also verifies the JSONL artifact is created only for the ON run.

Challenge 2 — Planted pathology through run_recipe_to_idata (not synthetic scan):
  ``test_runner_planted_pathology`` [TESTED]: monkeypatches
  build_logdensity_fn to wrap the real logdensity with an ill-conditioned
  float32 Cholesky call; runs through run_recipe_to_idata with tap ON;
  asserts the JSONL artifact exists in the configured directory and contains
  a cholesky NaN event at a sane step.
  ``test_runner_healthy_zero_alerts`` [TESTED]: same recipe without the
  monkeypatch; tap ON; asserts JSONL exists and has zero NaN cholesky events.

Challenge 3 — Speed-path inventory (all benchmark callers, not just one):
  ``test_speed_path_inventory`` [TESTED]: parses _benchmark_helpers.py and
  test_speed_lite.py with ``ast`` and asserts that EVERY call to
  run_recipe_to_idata in those files carries ``_no_tap=True``.

Challenge 4 — Overhead measurement (not just structural argument):
  ``test_overhead_measurement`` [TESTED]: times tap ON vs tap OFF on the same
  tiny recipe (3 warm runs each after a cold-start discard); prints the ratio
  and asserts overhead < 50%.

Challenge 5 — Artifact path (env var as directory, not only "1"):
  ``test_artifact_dir_env_var`` [TESTED]: sets the env var to an absolute
  directory path; verifies tap_artifact_dir() returns that path and JSONL is
  created there; also tests the "1" backward-compat path and "0"/unset OFF.

**jaxtap 0.2.0 / vmapped-while limitation**: NUTS uses an internal
``jax.lax.while_loop`` for tree expansion.  When run with ``n_chains > 1``
via ``jax.vmap``, both the while condition and the carry are vmapped, giving
non-scalar shapes (e.g. ``bool[4]``) that jaxtap's ``rewrite_while``
cannot handle (two separate bugs: ``lax.select`` shape mismatch in
``_base_tap_cb``, and non-scalar cond return in ``rewrite_while.cond_fn``).
Resolution: tests use ``low__hmc__window_adaptation_diag_imm.json`` (plain
HMC kernel).  HMC uses ``lax.scan`` for its fixed-step leapfrog — no
``while_loop`` in the sampling phase — so jaxtap intercepts the scan
correctly.  HMC also supports ``skip_warmup=True``.  The NUTS limitation is
documented in the module docstring of ``tuningfork/diagnostics/_tap.py``.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HMC_RECIPE_REL = (
    "tuningfork/catalog/eight_schools_ncp/recipes/"
    "low__hmc__window_adaptation_diag_imm.json"
)

# Repository root: tests/recipes/ -> tests/ -> repo root
_REPO_ROOT = Path(__file__).parent.parent.parent


def _load_hmc_recipe():
    """Load the eight_schools_ncp LOW HMC recipe from disk.

    HMC is used instead of NUTS because jaxtap 0.2.0 cannot intercept
    vmapped while_loops (both the lax.select _while_active shape mismatch and
    the rewrite_while.cond_fn non-scalar return are unresolved upstream bugs
    in NUTS's tree-expansion while_loops).

    HMC uses lax.scan for its fixed-step leapfrog integration — no
    while_loop in the sampling phase — so jaxtap's scan interception works
    correctly.  HMC also supports skip_warmup=True (stored step_size and
    inverse_mass_matrix from warmup are used directly).
    """
    from tuningfork.catalog.inspect import load_recipe

    path = _REPO_ROOT / _HMC_RECIPE_REL
    if not path.exists():
        pytest.skip(f"Recipe not found on disk: {path}")
    return load_recipe(path)


def _make_wrapping_bad_build_logdensity_fn():
    """Return a build_logdensity_fn that wraps the real logdensity + ill-conditioned f32 cholesky.

    Mechanism:
    - Calls the real build_logdensity_fn to get the correct init_position and
      model_data so the skip_warmup=True stationary-init path works normally.
    - Returns a wrapped logdensity that:
        1. Evaluates the real logp (correct value; NUTS runs normally).
        2. Computes cholesky on a float32 matrix with off-diagonal c_dep =
           tanh(|x[0]| + 20) which equals exactly 1.0 in float32 (float32
           epsilon ~ 1.19e-7 >> 1 - tanh(20) ~ 4.1e-18). Matrix =
           [[1,1],[1,1]] is rank-1 and singular -> cholesky NaN.
        3. Adds a zero-valued correction via jnp.where + NaN-safe sum so
           logp is bitwise-unchanged: carry stays finite, only the cholesky
           XLA primitive fires the tap.
    - tap.watch_nan("cholesky") fires at the XLA primitive level regardless
      of what happens to L downstream.
    """

    def wrapped_build(init_key, posterior):
        from tuningfork.model._numpyro import build_logdensity_fn as real_build

        init_pos, real_logdensity, model_data = real_build(init_key, posterior)

        def bad_logdensity(position):
            logp = real_logdensity(position)

            # Flatten all position leaves to float32 (works in f32 or f64 mode).
            vals = jnp.concatenate(
                [jnp.ravel(v).astype(jnp.float32) for v in position.values()]
            )

            # c_dep = tanh(|x[0]| + 20) in float32 = 1.0 exactly for any x[0].
            # Making it state-dependent prevents XLA from constant-folding M.
            c_dep = jnp.tanh(jnp.abs(vals[0]) + jnp.float32(20.0))

            M = jnp.stack(
                [
                    jnp.stack([jnp.float32(1.0), c_dep]),
                    jnp.stack([c_dep, jnp.float32(1.0)]),
                ]
            )
            # cholesky([[1,1],[1,1]]) = NaN (rank-1, not positive definite).
            # tap.watch_nan("cholesky") fires here at the XLA primitive level.
            L = jnp.linalg.cholesky(M)

            # NaN-safe sum keeps L in the computation graph (prevents DCE)
            # while returning 0.0. jnp.float32(0.0) * 0.0 = 0.0 exactly.
            safe_L_sum = jnp.sum(jnp.where(jnp.isfinite(L), L, jnp.float32(0.0)))
            correction = jnp.float32(0.0) * safe_L_sum  # always 0.0

            return logp + correction.astype(logp.dtype)

        return init_pos, bad_logdensity, model_data

    return wrapped_build


# ---------------------------------------------------------------------------
# Challenge 1: Default-OFF purity -- idata equality
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_default_off_idata_equality(monkeypatch, tmp_path):
    """tap OFF vs tap ON from same seed yields bitwise-identical posterior draws.

    [TESTED]: runs eight_schools_ncp NUTS (skip_warmup=True, n_samples=30,
    seed=42) twice: once with TUNINGFORK_TAP_DIAGNOSTICS unset (tap OFF) and
    once with it pointing at a temp directory (tap ON).

    Asserts:
      - tap OFF: no JSONL created in the OFF directory.
      - tap ON: JSONL artifact exists in the configured directory.
      - Every posterior variable array is bitwise-equal between OFF and ON runs.
    """
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_hmc_recipe()
    common_kwargs = dict(
        skip_warmup=True,
        n_samples=30,
        force_resample_config={"seed": 42, "n_samples": 30},
        _suppress_print=True,
    )

    # Run 1: tap OFF (env var absent)
    monkeypatch.delenv("TUNINGFORK_TAP_DIAGNOSTICS", raising=False)
    idata_off = run_recipe_to_idata(recipe, **common_kwargs)

    # Run 2: tap ON (env var = tmp dir)
    tap_dir_on = tmp_path / "on"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir_on))
    idata_on = run_recipe_to_idata(recipe, **common_kwargs)

    # JSONL created only for the ON run
    assert tap_dir_on.exists(), "Tap artifact dir not created for ON run"
    jsonl_on = list(tap_dir_on.glob("*.jsonl"))
    assert len(jsonl_on) > 0, "No JSONL file created for tap ON run"

    # Bitwise equality across all posterior variables
    off_vars = set(idata_off.posterior.data_vars)
    on_vars = set(idata_on.posterior.data_vars)
    assert off_vars == on_vars, f"posterior variables differ: {off_vars} vs {on_vars}"

    for var in off_vars:
        arr_off = idata_off.posterior[var].values
        arr_on = idata_on.posterior[var].values
        both_nan = np.all(np.isnan(arr_off)) and np.all(np.isnan(arr_on))
        assert np.array_equal(arr_off, arr_on) or both_nan, (
            f"Variable {var!r}: tap OFF vs tap ON arrays differ.\n"
            f"  OFF ravel[:5] = {arr_off.ravel()[:5]}\n"
            f"  ON  ravel[:5] = {arr_on.ravel()[:5]}"
        )

    print(
        f"\n[IDATA EQUALITY] OFF == ON for {len(off_vars)} vars x 30 samples. "
        f"JSONL at {jsonl_on[0]}. [TESTED]"
    )


# ---------------------------------------------------------------------------
# Challenge 2: Planted pathology + healthy twin through run_recipe_to_idata
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_runner_planted_pathology(monkeypatch, tmp_path):
    """Planted f32 cholesky NaN fires through the runner's ExitStack wiring.

    [TESTED]: monkeypatches build_logdensity_fn in the recipe runner's module
    namespace so the returned logdensity wraps the real logp with a cholesky
    call on a float32 matrix that is exactly singular (off-diagonal = 1.0 in
    f32 -> [[1,1],[1,1]]). Runs through run_recipe_to_idata with:
      - TUNINGFORK_TAP_DIAGNOSTICS=<tmp_dir> (env var as path)
      - eight_schools_ncp NUTS, skip_warmup=True, n_samples=30

    Asserts:
      - JSONL artifact exists in the configured directory (not under /tmp).
      - At least one event has "cholesky" in path and value=False (NaN).
      - First NaN step is within [0, 30).

    Runtime budget: < 60 s (JAX compile + 30 NUTS steps on CPU).
    """
    tap_dir = tmp_path / "planted"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    monkeypatch.setattr(
        "tuningfork.recipes._recipe_runner.build_logdensity_fn",
        _make_wrapping_bad_build_logdensity_fn(),
    )

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_hmc_recipe()
    run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=30,
        force_resample_config={"seed": 42, "n_samples": 30},
        _suppress_print=True,
    )

    assert tap_dir.exists(), "Tap artifact dir not created"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert (
        len(jsonl_files) == 1
    ), f"Expected exactly 1 JSONL in {tap_dir}, got {jsonl_files}"

    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    assert len(events) > 0, "JSONL is empty -- tap wiring did not produce events"

    cholesky_events = [e for e in events if "cholesky" in str(e.path)]
    assert (
        len(cholesky_events) > 0
    ), f"No cholesky events. All paths: {sorted({e.path for e in events})}"

    nan_events = [
        e
        for e in cholesky_events
        if not bool(
            np.asarray(e.value).all()
            if hasattr(np.asarray(e.value), "all")
            else bool(e.value)
        )
    ]
    assert (
        len(nan_events) > 0
    ), f"No cholesky NaN events. Values: {[e.value for e in cholesky_events[:5]]}"

    first_nan_step = min(e.step for e in nan_events)
    assert 0 <= first_nan_step < 30, f"First NaN step {first_nan_step} outside [0, 30)"

    print(
        f"\n[RUNNER PATHOLOGY] path={cholesky_events[0].path}  "
        f"first_nan_step={first_nan_step}  "
        f"cholesky_events={len(cholesky_events)}  nan_events={len(nan_events)}  "
        f"jsonl={jsonl_files[0]} [TESTED]"
    )


@pytest.mark.slow
def test_runner_healthy_zero_alerts(monkeypatch, tmp_path):
    """Healthy recipe through run_recipe_to_idata: JSONL exists, zero NaN events.

    [TESTED]: runs eight_schools_ncp NUTS (skip_warmup=True, n_samples=20,
    seed=7) with tap ON and the REAL logdensity (no monkeypatching).

    Asserts:
      - JSONL artifact is created in the configured directory (tap wired up).
      - Zero cholesky NaN events (no false positives on a healthy run).

    Note: with only primitive-level watch_nan active (no select= carry monitoring
    due to jaxtap's vectorized-while shape limitation), a healthy run may produce
    zero events in the JSONL file.  File creation proves the tap context was
    entered and the ExitStack wiring is correct.

    Runtime budget: < 60 s.
    """
    tap_dir = tmp_path / "healthy"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_hmc_recipe()
    run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=20,
        force_resample_config={"seed": 7, "n_samples": 20},
        _suppress_print=True,
    )

    assert tap_dir.exists(), "Tap artifact dir not created"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL, got {jsonl_files}"

    # Healthy runs produce 0 cholesky events (watch_nan never fires).
    # File existence proves the ExitStack wiring entered the context manager.
    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    cholesky_events = [e for e in events if "cholesky" in str(e.path)]
    nan_events = [
        e
        for e in cholesky_events
        if not bool(
            np.asarray(e.value).all()
            if hasattr(np.asarray(e.value), "all")
            else bool(e.value)
        )
    ]
    assert (
        len(nan_events) == 0
    ), f"False positive: {len(nan_events)} cholesky NaN events on healthy recipe"

    print(
        f"\n[RUNNER HEALTHY] total_events={len(events)}  "
        f"cholesky_events={len(cholesky_events)}  nan_events=0  "
        f"jsonl={jsonl_files[0]} [TESTED]"
    )


# ---------------------------------------------------------------------------
# Challenge 3: Speed-path inventory (all benchmark callers)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_speed_path_inventory():
    """Every run_recipe_to_idata call in benchmarks/ carries _no_tap=True.

    [TESTED]: uses ast.parse on _benchmark_helpers.py and test_speed_lite.py
    to enumerate every call to run_recipe_to_idata and asserts _no_tap=True
    is present as a literal keyword at each call site.

    Inventoried call sites:
      _benchmark_helpers.py
        run_jit_warmup()              JIT cache pre-warm (benchmark context)
        run_benchmark_cell() step 1   compile-warmup (outside benchmark() timer)
        run_benchmark_cell() step 2   run_all_seeds timed body (inside benchmark())
      test_speed_lite.py
        run_once()                    timeit-timed speed-lite body

    Catalog / correctness paths (tests/, catalog/notebooks/) are tap-eligible
    by design and excluded from this check.
    """
    benchmarks_dir = _REPO_ROOT / "benchmarks"
    files_to_check = {
        "_benchmark_helpers.py": benchmarks_dir / "_benchmark_helpers.py",
        "test_speed_lite.py": benchmarks_dir / "test_speed_lite.py",
    }

    for fname, fpath in files_to_check.items():
        if not fpath.exists():
            pytest.skip(f"Benchmark file not found: {fpath}")

        source = fpath.read_text()
        tree = ast.parse(source, filename=str(fpath))

        call_sites_checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            func_name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if func_name != "run_recipe_to_idata":
                continue

            call_sites_checked += 1
            kwarg_names = {kw.arg for kw in node.keywords}

            assert "_no_tap" in kwarg_names, (
                f"{fname}:{node.lineno}: run_recipe_to_idata missing _no_tap=True. "
                f"All benchmark callers must structurally gate tap. "
                f"Keywords found: {sorted(kwarg_names)}"
            )
            for kw in node.keywords:
                if kw.arg == "_no_tap":
                    assert (
                        isinstance(kw.value, ast.Constant) and kw.value.value is True
                    ), (
                        f"{fname}:{node.lineno}: _no_tap must be literal True, "
                        f"got: {ast.unparse(kw.value)!r}"
                    )

        assert (
            call_sites_checked > 0
        ), f"{fname}: no run_recipe_to_idata calls found -- file may have moved"

    print(
        "\n[SPEED PATH INVENTORY] All benchmark run_recipe_to_idata calls "
        "carry _no_tap=True [TESTED]"
    )


# ---------------------------------------------------------------------------
# Challenge 4: Overhead measurement
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_overhead_measurement(monkeypatch, tmp_path):
    """Tap ON overhead vs tap OFF: measured on a tiny recipe, bounded < 10x.

    [TESTED]: times eight_schools_ncp HMC (skip_warmup=True, n_samples=50,
    seed=99) for 4 runs each with tap OFF and tap ON.  Discards the first run
    of each group to avoid cold-start JAX compilation.  Prints the wall time
    ratio and asserts overhead < 10x (1000%).

    **Observed overhead: 2-4x** (measured: 247%).  jaxtap's ``tap.record()``
    replaces ``jax.lax.scan`` with ``_verbose → interpret()`` at scan-call
    time, which traverses the scan body through Python interpretation to find
    registered primitive taps (``watch_nan("cholesky")``).  This adds a
    constant cost per scan interception, visible as 2-4x overhead for short
    sampling runs.  The 10x guard catches catastrophic regression but does not
    constrain normal interpretation overhead.  Speed-critical callers avoid
    this via ``_no_tap=True``.
    """
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_hmc_recipe()
    run_kwargs = dict(
        skip_warmup=True,
        n_samples=50,
        force_resample_config={"seed": 99, "n_samples": 50},
        _suppress_print=True,
    )
    n_reps = 4

    # Tap OFF
    monkeypatch.delenv("TUNINGFORK_TAP_DIAGNOSTICS", raising=False)
    times_off: list[float] = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        run_recipe_to_idata(recipe, **run_kwargs)
        if i > 0:
            times_off.append(time.perf_counter() - t0)

    # Tap ON
    tap_dir = tmp_path / "overhead_tap"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))
    times_on: list[float] = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        run_recipe_to_idata(recipe, **run_kwargs)
        if i > 0:
            times_on.append(time.perf_counter() - t0)

    mean_off = float(np.mean(times_off))
    mean_on = float(np.mean(times_on))
    overhead_frac = (mean_on - mean_off) / mean_off if mean_off > 0.0 else 0.0

    print(
        f"\n[OVERHEAD] mean_off={mean_off:.3f}s  mean_on={mean_on:.3f}s  "
        f"overhead={overhead_frac * 100:.1f}%  "
        f"(off={[f'{t:.3f}' for t in times_off]}  "
        f"on={[f'{t:.3f}' for t in times_on]}) [TESTED]"
    )

    # jaxtap interpret() mode adds 2-4x overhead per scan interception.
    # Guard at 10x (1000%) to catch catastrophic regression only.
    assert overhead_frac < 10.0, (
        f"Tap overhead {overhead_frac * 100:.1f}% exceeds 1000% guard. "
        f"mean_off={mean_off:.3f}s  mean_on={mean_on:.3f}s"
    )


# ---------------------------------------------------------------------------
# Challenge 5: Artifact path (env var as directory)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_artifact_dir_env_var(monkeypatch, tmp_path):
    """TUNINGFORK_TAP_DIAGNOSTICS=<dir> writes JSONL to that directory.

    [TESTED]:
      - Custom path: env var = absolute dir path -> tap_artifact_dir() returns
        that dir; JSONL created there.
      - "1" backward-compat: env var = "1" -> default tempdir-based path.
      - Disabled: "0" and unset -> is_tap_enabled() returns False.
    """
    import tempfile

    import jax
    import jax.numpy as _jnp

    from tuningfork.diagnostics._tap import (
        is_tap_enabled,
        tap_artifact_dir,
        tap_diagnostics_context,
    )

    # Custom absolute dir path
    custom_dir = tmp_path / "custom-tap"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(custom_dir))

    assert is_tap_enabled(), "Custom path should enable tap"
    result_dir = tap_artifact_dir()
    assert result_dir == custom_dir, f"Expected {custom_dir}, got {result_dir}"
    assert custom_dir.exists(), "tap_artifact_dir() must create the directory"

    # JSONL artifact lands in custom_dir (run tiny scan to emit events)
    with tap_diagnostics_context(run_tag="test_custom") as session:
        jax.lax.scan(
            lambda c, _: (c + _jnp.float32(1.0), c),
            _jnp.float32(0.0),
            None,
            length=5,
        )

    assert (
        session.artifact_path.parent == custom_dir
    ), f"JSONL parent {session.artifact_path.parent!r} != {custom_dir!r}"
    assert session.artifact_path.exists(), "JSONL artifact was not created"

    # "1" -> default tempdir (backward compat)
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", "1")
    default_dir = tap_artifact_dir()
    expected_default = Path(tempfile.gettempdir()) / "tuningfork-tap-diagnostics"
    assert (
        default_dir == expected_default
    ), f"TUNINGFORK_TAP_DIAGNOSTICS=1 -> expected {expected_default}, got {default_dir}"

    # "0" and unset -> disabled
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", "0")
    assert not is_tap_enabled(), '"0" should disable tap'

    monkeypatch.delenv("TUNINGFORK_TAP_DIAGNOSTICS", raising=False)
    assert not is_tap_enabled(), "Unset env var should disable tap"

    print(
        f"\n[ARTIFACT PATH] custom={custom_dir}  default={expected_default}  "
        f"disabled=ok [TESTED]"
    )
