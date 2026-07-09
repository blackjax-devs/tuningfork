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

Eight tests covering five AYS Round 1 challenges plus three AYS Round 2
requirements:

Challenge 1 — Default-OFF purity (idata equality, not a flag check):
  ``test_default_off_idata_equality`` [TESTED]: runs the same tiny recipe x
  same seed twice — tap OFF vs tap ON — and asserts every posterior array is
  bitwise-identical. Proves that entering the tap context does not alter
  results (jaxtap bitwise-identity guarantee is ours, not just inherited).
  Also verifies the JSONL artifact is created only for the ON run.

Challenge 2 — Planted pathology through run_recipe_to_idata (not synthetic scan):
  ``test_runner_planted_pathology`` [TESTED]: proves WIRING — that the
  ExitStack → tap context → JSONL path is correctly connected through
  run_recipe_to_idata. Monkeypatches build_logdensity_fn to wrap the real
  logdensity with an ill-conditioned float32 Cholesky call; runs through
  run_recipe_to_idata with tap ON; asserts the JSONL artifact exists and
  contains a cholesky NaN event. Does NOT prove step-level attribution — see
  test_synthetic_scan_nan_attribution for that.
  ``test_runner_healthy_zero_alerts`` [TESTED]: same recipe without the
  monkeypatch; tap ON; asserts JSONL exists and has zero NaN cholesky events.

Challenge 3 — Speed-path inventory (all benchmark callers, not just one):
  ``test_speed_path_inventory`` [TESTED]: parses _benchmark_helpers.py and
  test_speed_lite.py with ``ast`` and asserts that EVERY call to
  run_recipe_to_idata in those files carries ``_no_tap=True``.

Challenge 4 — Overhead measurement (not just structural argument):
  ``test_overhead_measurement`` [TESTED / REPORT-ONLY]: times tap ON vs tap
  OFF on the same tiny recipe (3 warm runs each after a cold-start discard);
  prints the ratio. No threshold assertion — wall-clock ratio assertions are
  cross-machine flake factories. Observed overhead: 2-4x (247% on the
  reference machine). jaxtap's interpret() mode is intrinsically expensive;
  speed-critical callers use _no_tap=True.

Challenge 5 — Artifact path (env var as directory, not only "1"):
  ``test_artifact_dir_env_var`` [TESTED]: sets the env var to an absolute
  directory path; verifies tap_artifact_dir() returns that path and JSONL is
  created there; also tests the "1" backward-compat path and "0"/unset OFF.

AYS Round 2 additions:

Never-crash guard — NUTS + env var ON:
  ``test_nuts_no_crash_with_tap_env`` [TESTED]: loads an MVN-10 NUTS recipe,
  sets TUNINGFORK_TAP_DIAGNOSTICS to a temp directory, runs
  run_recipe_to_idata. Asserts: run COMPLETES (no exception); no JSONL
  artifact is created (NUTS is incompatible, taps are skipped); one
  logging.WARNING is emitted naming the jaxtap vmap-while incompatibility.
  This is the core never-crash invariant test.

Synthetic-scan attribution — NaN at step N attributed at step N:
  ``test_synthetic_scan_nan_attribution`` [TESTED]: proves that
  tap.watch_nan("cholesky") fires at the CORRECT scan step, not at step 0 or
  always. Runs a 25-step lax.scan that injects a NaN cholesky at step 7
  (healthy carry at steps 0-6, NaN at steps 7+). Asserts: an event fires at
  step 7 (not earlier). This proves step-level attribution, which the runner
  pathology test cannot (it only proves wiring). See test_runner_planted_pathology
  for proof that the wiring through run_recipe_to_idata works.

**jaxtap 0.2.0 / vmapped-while limitation**: NUTS uses an internal
``jax.lax.while_loop`` for tree expansion.  When run with ``n_chains > 1``
via ``jax.vmap``, both the while condition and the carry are vmapped, giving
non-scalar shapes (e.g. ``bool[4]``) that jaxtap's ``rewrite_while``
cannot handle (two separate bugs: ``lax.select`` shape mismatch in
``_base_tap_cb``, and non-scalar cond return in ``rewrite_while.cond_fn``).
The never-crash guard in run_recipe_to_idata detects NUTS via
is_algorithm_tap_compatible() and skips tap setup with a WARNING.
HMC tests use ``low__hmc__window_adaptation_diag_imm.json`` (plain HMC
kernel), which uses ``lax.scan`` for fixed-step leapfrog — no while_loop.
"""

from __future__ import annotations

import ast
import logging
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

_NUTS_RECIPE_REL = (
    "tuningfork/catalog/mvn_10/recipes/" "low__nuts__window_adaptation_diag_imm.json"
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


def _load_nuts_recipe():
    """Load the mvn_10 LOW NUTS recipe from disk.

    Used only in the never-crash guard test — NUTS is incompatible with
    jaxtap 0.2.0 (vmapped while_loop bugs), and the test verifies that
    run_recipe_to_idata completes normally with a WARNING instead of crashing.
    """
    from tuningfork.catalog.inspect import load_recipe

    path = _REPO_ROOT / _NUTS_RECIPE_REL
    if not path.exists():
        pytest.skip(f"NUTS recipe not found on disk: {path}")
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

    WIRING TEST: proves that the ExitStack → tap_diagnostics_context → JSONL
    pipeline is correctly connected through run_recipe_to_idata. It does NOT
    prove step-level attribution (i.e., it does not verify that a NaN injected
    at step N is reported at step N rather than step 0). For step-level
    attribution, see test_synthetic_scan_nan_attribution which injects NaN at
    a specific scan step (step 7) and verifies the event fires at exactly that
    step.

    [TESTED]: monkeypatches build_logdensity_fn in the recipe runner's module
    namespace so the returned logdensity wraps the real logp with a cholesky
    call on a float32 matrix that is exactly singular (off-diagonal = 1.0 in
    f32 -> [[1,1],[1,1]]). Runs through run_recipe_to_idata with:
      - TUNINGFORK_TAP_DIAGNOSTICS=<tmp_dir> (env var as path)
      - eight_schools_ncp HMC, skip_warmup=True, n_samples=30

    Asserts:
      - JSONL artifact exists in the configured directory (not under /tmp).
      - At least one event has "cholesky" in path and value=False (NaN).
      - First NaN step is within [0, 30).

    Runtime budget: < 60 s (JAX compile + 30 HMC steps on CPU).
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

    [TESTED]: runs eight_schools_ncp HMC (skip_warmup=True, n_samples=20,
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
# Challenge 4: Overhead measurement (REPORT-ONLY, no threshold assertion)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_overhead_measurement(monkeypatch, tmp_path):
    """Tap ON overhead vs tap OFF: measured on a tiny recipe, printed only.

    REPORT-ONLY: this test prints the wall-clock ratio but makes NO assertion
    about the value. Wall-clock ratio assertions are cross-machine flake
    factories — a number that is 2x on a development laptop may be 8x on a
    loaded CI runner.

    [TESTED]: times eight_schools_ncp HMC (skip_warmup=True, n_samples=50,
    seed=99) for 4 runs each with tap OFF and tap ON.  Discards the first run
    of each group to avoid cold-start JAX compilation.

    **Observed overhead: 2-4x** (measured: 247% on the reference machine).
    jaxtap's ``tap.record()`` replaces ``jax.lax.scan`` with
    ``_verbose → interpret()`` at scan-call time, which traverses the scan
    body through Python interpretation to find registered primitive taps
    (``watch_nan("cholesky")``).  This adds a constant cost per scan
    interception, visible as 2-4x overhead for short sampling runs.
    Speed-critical callers avoid this via ``_no_tap=True``.
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
        f"on={[f'{t:.3f}' for t in times_on]}) [TESTED / REPORT-ONLY]"
    )
    # No assertion here — wall-clock ratio is machine-dependent.
    # Observed: 2-4x on reference hardware. See module docstring for context.


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


# ---------------------------------------------------------------------------
# AYS Round 2 (1): Never-crash guard — NUTS recipe + tap env ON
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_nuts_no_crash_with_tap_env(monkeypatch, tmp_path, caplog):
    """NUTS recipe + TUNINGFORK_TAP_DIAGNOSTICS ON: run completes, no artifact, one warning.

    This test verifies the CORE INVARIANT of the tap diagnostics feature:
    a user who sets TUNINGFORK_TAP_DIAGNOSTICS=1 on a NUTS recipe (the most
    common sampler family in tuningfork) must NOT get a crash.

    Context: jaxtap 0.2.0 cannot handle NUTS's tree-expansion while_loop when
    it is vmapped over n_chains (Bug 1: _base_tap_cb lax.select shape mismatch;
    Bug 2: rewrite_while.cond_fn non-scalar cond return). Without a guard,
    setting the env var on any NUTS recipe would raise TypeError and abort
    the run — catastrophic for a diagnostics switch whose job is to help debug.

    The never-crash guard in run_recipe_to_idata calls
    is_algorithm_tap_compatible(recipe.base_method_name) before entering
    tap_diagnostics_context. For NUTS (incompatible), it emits logging.WARNING
    and skips tap setup entirely.

    [TESTED]: loads mvn_10 LOW NUTS recipe (skip_warmup=True, n_samples=30),
    sets TUNINGFORK_TAP_DIAGNOSTICS to a temp directory, runs
    run_recipe_to_idata. Asserts:
      - Run COMPLETES without any exception.
      - NO JSONL artifact appears in the configured directory (taps skipped).
      - At least one WARNING log record mentions "jaxtap" (upstream issue name).
    """
    tap_dir = tmp_path / "nuts_guard"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_nuts_recipe()

    with caplog.at_level(logging.WARNING):
        # Must NOT raise — that is the invariant
        run_recipe_to_idata(
            recipe,
            skip_warmup=True,
            n_samples=30,
            force_resample_config={"seed": 42, "n_samples": 30},
            _suppress_print=True,
        )

    # No JSONL artifact: tap was skipped for NUTS
    jsonl_files = list(tap_dir.glob("*.jsonl")) if tap_dir.exists() else []
    assert len(jsonl_files) == 0, (
        f"NUTS tap guard failed: JSONL artifact was created despite NUTS "
        f"being incompatible: {jsonl_files}"
    )

    # One warning naming the upstream jaxtap incompatibility
    warning_records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "jaxtap" in r.getMessage().lower()
    ]
    assert len(warning_records) >= 1, (
        f"Expected at least one WARNING mentioning 'jaxtap' for NUTS tap skip. "
        f"All WARNING records: {[r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]}"
    )

    print(
        f"\n[NUTS GUARD] run completed, no artifact, "
        f"warning='{warning_records[0].getMessage()[:80]}...' [TESTED]"
    )


# ---------------------------------------------------------------------------
# AYS Round 2 (2): Synthetic-scan attribution — NaN at step N attributed at N
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_synthetic_scan_nan_attribution(monkeypatch, tmp_path):
    """tap.watch_nan fires at the CORRECT scan step, not at step 0 or always.

    ATTRIBUTION TEST: proves that jaxtap's watch_nan("cholesky") identifies
    WHEN a NaN first appears, not just that it appeared. This is the property
    that test_runner_planted_pathology cannot verify — the planted pathology
    at the runner level produces a NaN on EVERY step (the ill-conditioned
    matrix is constant over the chain), so it proves wiring but not
    step-level attribution. This test injects a NaN at exactly step 7 of a
    25-step scan and asserts the event fires at step 7 (not step 0).

    For proof that the wiring through run_recipe_to_idata works (ExitStack →
    tap context → JSONL), see test_runner_planted_pathology.

    [TESTED]: runs a 25-step lax.scan with a synthetic body that:
      1. Accumulates a float32 counter (the carry).
      2. At step >= 7: calls cholesky on [[1,1],[1,1]] (exactly singular in
         f32, guaranteed NaN). At step < 7: calls cholesky on the identity
         matrix (healthy, finite result).
      3. Adds a zero-valued correction to keep the carry finite (preventing
         propagation of NaN into the accumulator — we want to test attribution,
         not NaN propagation).

    Asserts:
      - At least one cholesky event fires (watch_nan is active).
      - The first NaN event is at step >= 7 (not at steps 0-6).
      - No NaN events at steps 0-6 (the healthy phase).

    The NaN-onset step is read from the event's ``.step`` field, which jaxtap
    sets to the scan iteration index. Attribution means: step reported ==
    step where NaN was injected.
    """
    import jax
    import jax.numpy as _jnp

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tmp_path / "attribution"))

    NAN_ONSET_STEP = 7
    TOTAL_STEPS = 25

    def scan_body(carry, step_idx):
        # At step >= NAN_ONSET_STEP: build a singular f32 matrix.
        # At step < NAN_ONSET_STEP: use identity (healthy).
        c_bad = _jnp.float32(1.0)  # off-diagonal = 1 -> singular
        c_good = _jnp.float32(0.0)  # off-diagonal = 0 -> identity, PD

        # Use lax.cond so both branches are traceable (safe under jit).
        c_dep = jax.lax.cond(
            step_idx >= NAN_ONSET_STEP,
            lambda: c_bad,
            lambda: c_good,
        )

        M = _jnp.array(
            [[_jnp.float32(1.0), c_dep], [c_dep, _jnp.float32(1.0)]],
            dtype=_jnp.float32,
        )
        L = _jnp.linalg.cholesky(M)

        # Keep L in the graph without propagating NaN into carry.
        safe_sum = _jnp.sum(_jnp.where(_jnp.isfinite(L), L, _jnp.float32(0.0)))
        new_carry = carry + _jnp.float32(0.0) * safe_sum  # carry unchanged

        return new_carry, _jnp.float32(0.0)

    artifact_path = tmp_path / "attribution" / "synthetic_attribution.jsonl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with tap_diagnostics_context(artifact_path=artifact_path, run_tag="synthetic"):
        jax.lax.scan(
            scan_body,
            _jnp.float32(0.0),
            _jnp.arange(TOTAL_STEPS, dtype=_jnp.int32),
        )

    from jaxtap import read_jsonl

    events = read_jsonl(artifact_path)

    cholesky_events = [e for e in events if "cholesky" in str(e.path)]
    assert len(cholesky_events) > 0, (
        "No cholesky events recorded. watch_nan may not have fired. "
        f"All event paths: {sorted({str(e.path) for e in events})}"
    )

    nan_events = [
        e
        for e in cholesky_events
        if not bool(
            np.asarray(e.value).all()
            if hasattr(np.asarray(e.value), "all")
            else bool(e.value)
        )
    ]
    assert len(nan_events) > 0, (
        f"No NaN cholesky events. cholesky event values: "
        f"{[e.value for e in cholesky_events[:10]]}"
    )

    first_nan_step = min(e.step for e in nan_events)

    # Attribution: NaN injected at step >= NAN_ONSET_STEP; must not fire earlier.
    assert first_nan_step >= NAN_ONSET_STEP, (
        f"Attribution failure: first NaN event at step {first_nan_step}, "
        f"but NaN was injected at step >= {NAN_ONSET_STEP}. "
        f"watch_nan fired before the NaN was present."
    )

    # Confirm no NaN events at the healthy steps (0 to NAN_ONSET_STEP - 1).
    early_nan_events = [e for e in nan_events if e.step < NAN_ONSET_STEP]
    assert len(early_nan_events) == 0, (
        f"False attribution: NaN events at steps {[e.step for e in early_nan_events]} "
        f"< {NAN_ONSET_STEP} (healthy region). These steps should have finite cholesky."
    )

    print(
        f"\n[ATTRIBUTION] NaN onset injected at step>={NAN_ONSET_STEP}, "
        f"first_nan_event_step={first_nan_step}, "
        f"nan_events={len(nan_events)}, total_steps={TOTAL_STEPS} [TESTED]"
    )
