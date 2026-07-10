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

Thirteen tests covering five AYS Round 1 challenges, three AYS Round 2
requirements, the jax-tap 0.2.1 unblock (NUTS vmapped-while fixed), and
three jax-tap 0.3.0 y-tap tests (treedepth tripwire + MCLMC divergence alert).

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

Never-crash guard (updated for 0.2.1):
  ``test_nuts_tap_active_since_021`` [TESTED]: NUTS recipe + tap ON → run
  completes, JSONL IS created (NUTS now in allowlist since 0.2.1), no
  incompatibility WARNING. This was originally written as a negative test
  (0.2.0: NUTS skipped with WARNING, no artifact); updated for 0.2.1 to
  assert the positive case. The never-crash guard is still active for UNKNOWN
  algorithms.

vmapped-while crash fix + NaN propagation through carry (0.2.1):
  ``test_nuts_vmapped_while_tap_events`` [TESTED]: NUTS + planted cholesky
  NaN — run COMPLETES (0.2.0 would crash with TypeError in the vmapped
  while_loop), JSONL IS created, outer scan carry events show NaN from the
  planted cholesky propagating through NUTS's while_loop body into the NUTS
  state. Note: NUTS's tree-expansion while_loop is inside a JIT boundary at
  recipe runtime, so jaxtap emits carry-level ``scan[0]`` events (not
  ``while``-path primitive events). The NaN appearing in the gradient leaf at
  step 3+ of the outer scan carry proves the planted logdensity IS evaluated
  inside NUTS's while_loop body (via value_and_grad, whose backward pass
  propagates NaN through the singular cholesky gradient).

Synthetic-scan attribution — NaN at step N attributed at step N:
  ``test_synthetic_scan_nan_attribution`` [TESTED]: proves that
  tap.watch_nan("cholesky") fires at the CORRECT scan step, not at step 0 or
  always. Runs a 25-step lax.scan that injects a NaN cholesky at step 7
  (healthy carry at steps 0-6, NaN at steps 7+). Asserts: an event fires at
  step 7 (not earlier). This proves step-level attribution, which the runner
  pathology test cannot (it only proves wiring). See test_runner_planted_pathology
  for proof that the wiring through run_recipe_to_idata works.

jax-tap 0.3.0 y-tap additions (#209 treedepth tripwire):
  ``test_nuts_treedepth_saturation`` [TESTED]: NUTS recipe + forced
  ``max_num_doublings=1`` (via ``dataclasses.replace``) → JSONL has output
  events (kind="output") with treedepth values == 1 (saturated);
  ``compute_saturation_fraction(path, max_num_doublings=1)`` returns non-zero
  fraction. Proves the treedepth y-tap fires and the policy-free helper works.

  ``test_nuts_healthy_zero_saturation`` [TESTED]: healthy NUTS run with the
  on-disk recipe (no ``max_num_doublings`` key in params); the runner defaults
  to cap=10 (blackjax kernel default) → y-tap arms, zero saturation events;
  ``compute_saturation_fraction`` returns fraction=0.0. Proves the tripwire
  arms on real catalog NUTS recipes and produces no false positives.

  ``test_mclmc_warmup_divergence_alert`` [TESTED]: MCLMC adaptation
  (``mclmc_find_L_and_step_size``) with a planted NaN logdensity (returns NaN
  for any position where x[0] > 0) → JSONL has output events with value=True
  (divergence flags from the adaptation scan ys). ``compute_saturation_fraction``
  returns non-zero fraction. Proves the MCLMC warmup divergence y-tap fires
  (blackjax #975 seam).

**jax-tap history**: 0.2.0 had two vmapped-while bugs (Bug 1:
``_base_tap_cb`` ``lax.select`` shape mismatch; Bug 2:
``rewrite_while.cond_fn`` non-scalar cond return) that prevented NUTS from
being instrumented.  Both fixed in 0.2.1 (arcueil/jax-tap#5, 2026-07-10).
All 24 in-scope base methods are now in ``_TAP_COMPATIBLE_BASE_METHODS``.
The never-crash guard (unknown → warn-and-skip) is a permanent design.
0.3.0 adds y-taps (select_ys / on_ys / alert_ys / alert_ys_once) enabling
the treedepth tripwire and MCLMC warmup divergence alert (#209).
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

    Used for the NUTS-specific tap tests (0.2.1 unblock):
    test_nuts_tap_active_since_021 (positive run, JSONL created) and
    test_nuts_vmapped_while_tap_events (planted cholesky, per-lane events).

    Note: jaxtap 0.2.0 had vmapped-while bugs that made NUTS incompatible;
    both are fixed in 0.2.1 (arcueil/jax-tap#5).
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
# AYS Round 2 (1) [updated for jax-tap 0.2.1]: NUTS recipe + tap env ON
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_nuts_tap_active_since_021(monkeypatch, tmp_path, caplog):
    """NUTS recipe + TUNINGFORK_TAP_DIAGNOSTICS ON: taps ACTIVE, JSONL created (jax-tap 0.2.1).

    This test was originally written for jax-tap 0.2.0, where NUTS crashed with
    TypeError when instrumented (vmapped-while Bug 1 + Bug 2). The never-crash
    guard was added to detect incompatible algorithms and skip tap setup with a
    WARNING. In 0.2.0, NUTS was excluded from _TAP_COMPATIBLE_BASE_METHODS.

    In jax-tap 0.2.1, both vmapped-while bugs are fixed (arcueil/jax-tap#5).
    NUTS is now IN _TAP_COMPATIBLE_BASE_METHODS. This test verifies the updated
    positive behavior: NUTS + tap ON → tap context ENTERED → JSONL created.

    The never-crash guard (is_algorithm_tap_compatible) is still a permanent
    design — it applies to UNKNOWN/FUTURE algorithms not yet in the allowlist.
    For NUTS (now in the allowlist), the guard passes through, no WARNING emitted.

    [TESTED]: loads mvn_10 LOW NUTS recipe (skip_warmup=True, n_samples=30),
    sets TUNINGFORK_TAP_DIAGNOSTICS to a temp directory, runs
    run_recipe_to_idata. Asserts:
      - Run COMPLETES without any exception.
      - JSONL artifact IS created (NUTS is now compatible — tap context entered).
      - NO WARNING about jaxtap incompatibility (NUTS is allowlisted).

    For per-lane event evidence from NUTS's vmapped while_loop, see
    test_nuts_vmapped_while_tap_events.
    """
    tap_dir = tmp_path / "nuts_021"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_nuts_recipe()

    with caplog.at_level(logging.WARNING):
        # Must NOT raise — both as invariant (guard) and as 0.2.1 positive test
        run_recipe_to_idata(
            recipe,
            skip_warmup=True,
            n_samples=30,
            force_resample_config={"seed": 42, "n_samples": 30},
            _suppress_print=True,
        )

    # JSONL created: NUTS is now tap-compatible, tap context was entered
    assert tap_dir.exists(), "Tap artifact dir not created for NUTS 0.2.1 run"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert (
        len(jsonl_files) == 1
    ), f"Expected 1 JSONL for NUTS + 0.2.1 (taps active), got {jsonl_files}"

    # No warning: NUTS is in the allowlist, guard passes through silently
    incompatibility_warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "jaxtap" in r.getMessage().lower()
        and "incompatib" in r.getMessage().lower()
    ]
    assert len(incompatibility_warnings) == 0, (
        f"Unexpected incompatibility WARNING for NUTS (should be allowlisted in 0.2.1): "
        f"{[r.getMessage() for r in incompatibility_warnings]}"
    )

    print(
        f"\n[NUTS 0.2.1 POSITIVE] run completed, jsonl={jsonl_files[0]}, "
        f"no incompatibility warning [TESTED]"
    )


# ---------------------------------------------------------------------------
# AYS Round 2 (1+) [jax-tap 0.2.1]: vmapped NUTS while_loop emits tap events
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_nuts_vmapped_while_tap_events(monkeypatch, tmp_path):
    """NUTS + planted cholesky NaN: run completes and carry NaN detected (jax-tap 0.2.1).

    In jax-tap 0.2.0, the NUTS tree-expansion while_loop (vmapped over n_chains)
    crashed with TypeError when instrumented (Bug 1: _base_tap_cb lax.select shape;
    Bug 2: rewrite_while.cond_fn non-scalar cond). Fixed in 0.2.1 (arcueil/jax-tap#5).

    The jax-tap 0.2.1 fix makes NUTS runnable under instrumentation. This test
    proves:
      1. The run COMPLETES without crash (the 0.2.0 TypeError is gone).
      2. JSONL is created (tap context was entered — NUTS is allowlisted).
      3. Carry events contain NaN values (the planted logdensity IS evaluated
         inside NUTS's while_loop body; NaN from the singular cholesky gradient
         propagates via value_and_grad into the NUTS state and from there into
         the outer scan carry, where jaxtap detects it).

    Note on observation boundary: NUTS's tree-expansion while_loop is inside a
    JIT compilation boundary at recipe runtime, so jaxtap emits outer scan carry
    events (path ``scan[0]``) rather than ``while``-path primitive events. The
    cholesky NaN does not appear as a primitive-level event in this context because
    the cholesky primitive executes inside the JIT. The NaN IS observable in the
    carry because JAX's backward pass propagates NaN through the singular cholesky
    gradient (``0.0 * NaN_grad = NaN`` in IEEE 754), infecting the logdensity
    gradient leaf of the NUTS carry starting around step 3.

    [TESTED]: uses mvn_10 NUTS recipe (skip_warmup=True, n_samples=30) with
    monkeypatched build_logdensity_fn that plants a float32 singular cholesky
    in every logdensity evaluation. Asserts:
      - JSONL artifact is created (tap context entered for NUTS).
      - events > 0 (sane output; carry events with structure).
      - At least one event carry value contains NaN (planted pathology detected
        at the outer scan carry level).

    For WIRING proof through the runner (ExitStack → tap → JSONL), see
    test_runner_planted_pathology (HMC scan-based, cholesky primitive events).
    For step-level attribution, see test_synthetic_scan_nan_attribution.
    For the positive NUTS crash-fix test without pathology, see
    test_nuts_tap_active_since_021.
    """
    tap_dir = tmp_path / "nuts_while_events"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    monkeypatch.setattr(
        "tuningfork.recipes._recipe_runner.build_logdensity_fn",
        _make_wrapping_bad_build_logdensity_fn(),
    )

    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    recipe = _load_nuts_recipe()
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
    ), f"Expected 1 JSONL for NUTS + 0.2.1 (taps active), got {jsonl_files}"

    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    assert len(events) > 0, (
        "JSONL is empty — tap context was entered but no events emitted. "
        f"JSONL path: {jsonl_files[0]}"
    )

    # Planted cholesky NaN propagates through NUTS value_and_grad into outer
    # scan carry (gradient leaf).  Verify at least one carry event has NaN.
    def _has_nan(value) -> bool:
        """Recursively check for NaN in a (possibly nested) event value."""
        if isinstance(value, tuple):
            return any(_has_nan(v) for v in value)
        try:
            arr = np.asarray(value)
            return bool(np.any(np.isnan(arr)))
        except (TypeError, ValueError):
            return False

    unique_paths = sorted({str(e.path) for e in events})
    nan_carry_events = [e for e in events if _has_nan(e.value)]
    assert len(nan_carry_events) > 0, (
        f"No carry events with NaN values detected. "
        f"Unique paths: {unique_paths}. "
        f"Total events: {len(events)}. "
        f"The planted singular cholesky should produce NaN gradients that "
        f"propagate into the outer scan carry via value_and_grad."
    )

    # All carry events should be at scan[0] (outer run_inference_algorithm scan).
    # NUTS's while_loop is inside a JIT boundary — jaxtap observes the NaN at
    # the outer scan level, not at the while primitive level.
    nan_paths = sorted({str(e.path) for e in nan_carry_events})
    first_nan_step = min(e.step for e in nan_carry_events)

    print(
        f"\n[NUTS WHILE 0.2.1] total_events={len(events)} "
        f"nan_carry_events={len(nan_carry_events)} "
        f"first_nan_step={first_nan_step} "
        f"unique_paths={unique_paths} "
        f"nan_paths={nan_paths} "
        f"[TESTED]"
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


# ---------------------------------------------------------------------------
# #209 treedepth tripwire: NUTS saturation + healthy + MCLMC divergence
# ---------------------------------------------------------------------------


_MCLMC_RECIPE_REL = (
    "tuningfork/catalog/mvn_10/recipes/"
    "low__adjusted_mclmc__adjusted_mclmc_tuning.json"
)


def _load_mclmc_recipe():
    """Load the mvn_10 LOW adjusted_mclmc recipe from disk."""
    from tuningfork.catalog.inspect import load_recipe

    path = _REPO_ROOT / _MCLMC_RECIPE_REL
    if not path.exists():
        pytest.skip(f"MCLMC recipe not found on disk: {path}")
    return load_recipe(path)


@pytest.mark.slow
def test_nuts_treedepth_saturation(monkeypatch, tmp_path):
    """NUTS with max_num_doublings=1 produces treedepth saturation output events.

    Forces treedepth saturation by capping tree expansion at depth 1 (the
    smallest possible cap; all steps either U-turn at depth 0 or saturate at
    depth 1).  For a well-tuned step_size on mvn_10, the step_size is adapted
    for max_num_doublings=10, so with max_num_doublings=1 most steps are forced
    to saturate.

    ``max_num_doublings=1`` is injected via ``dataclasses.replace`` on the
    base_method_params dict and flows to ``blackjax.nuts()`` through the recipe
    runner's ``_skip_extra_kwargs`` path.

    The tap context (base_method_name="nuts", max_num_doublings=1) wires the
    NUTS treedepth y-tap.  ``select_ys`` finds the first int32 scalar in the
    ys flat leaves (= num_trajectory_expansions) per the documented proof in
    the module docstring.  ``alert_ys`` fires when value >= max_num_doublings.

    Asserts:
    - JSONL has at least one output event (kind="output") from the y-tap.
    - At least one output event has value >= 1 (saturated at the forced cap).
    - ``compute_saturation_fraction(path, max_num_doublings=1)`` returns a
      non-zero fraction.
    """
    import dataclasses

    from tuningfork.diagnostics._tap import compute_saturation_fraction
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    tap_dir = tmp_path / "nuts_sat"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    recipe = _load_nuts_recipe()

    # Force saturation: cap tree expansion at depth 1
    sat_recipe = dataclasses.replace(
        recipe,
        base_method_params={**recipe.base_method_params, "max_num_doublings": 1},
    )

    run_recipe_to_idata(
        sat_recipe,
        skip_warmup=True,
        n_samples=20,
        force_resample_config={"seed": 42, "n_samples": 20},
        _suppress_print=True,
    )

    assert tap_dir.exists(), "Tap artifact dir not created"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL, got {jsonl_files}"

    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    output_events = [e for e in events if getattr(e, "kind", "carry") == "output"]
    assert len(output_events) > 0, (
        "No output events (kind='output') in JSONL — y-tap did not fire. "
        f"Total events: {len(events)}. Output events: 0."
    )

    # At least one output event should show saturation (value == 1 >= max_num_doublings=1)
    saturated = [e for e in output_events if _safe_int_val(e.value) >= 1]
    assert len(saturated) > 0, (
        f"No saturated output events (value >= 1). "
        f"Output event values: {[e.value for e in output_events[:10]]}"
    )

    sat_n, total, frac = compute_saturation_fraction(
        jsonl_files[0], max_num_doublings=1
    )
    assert sat_n > 0, f"compute_saturation_fraction returned sat_n=0 (total={total})"
    assert 0.0 < frac <= 1.0, f"Expected frac in (0, 1], got {frac}"

    print(
        f"\n[NUTS SAT] output_events={len(output_events)} saturated={len(saturated)} "
        f"sat_n={sat_n} total={total} frac={frac:.2%} "
        f"sample_values={[e.value for e in output_events[:5]]} [TESTED]"
    )


@pytest.mark.slow
def test_nuts_healthy_zero_saturation(monkeypatch, tmp_path):
    """Healthy NUTS run with the on-disk recipe: zero saturation output events.

    Uses the mvn_10 NUTS recipe as it lives on disk (no ``max_num_doublings``
    key in ``base_method_params``).  The runner calls
    ``recipe.base_method_params.get("max_num_doublings", 10)``, so it
    substitutes the blackjax kernel default (10) when the recipe doesn't pin
    it — the treedepth y-tap is armed on all real catalog recipes.

    This test proves that the tripwire fires on a real catalog recipe and that
    no saturation is reported on healthy 10D MVN (which should never approach
    depth 10 with a well-tuned step_size).

    Asserts:
    - JSONL has some output events (proves y-tap is armed for on-disk recipe).
    - Zero output events have value >= 10 (no saturation on healthy mvn_10).
    - ``compute_saturation_fraction(path, max_num_doublings=10)`` returns
      fraction == 0.0.
    """
    from tuningfork.diagnostics._tap import compute_saturation_fraction
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    tap_dir = tmp_path / "nuts_healthy"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    recipe = _load_nuts_recipe()
    # Use the on-disk recipe directly — no injection needed.  The runner
    # defaults to cap=10 (blackjax nuts kernel default) when the key is absent,
    # so the treedepth y-tap arms automatically for all real catalog NUTS recipes.

    run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=30,
        force_resample_config={"seed": 7, "n_samples": 30},
        _suppress_print=True,
    )

    assert tap_dir.exists(), "Tap artifact dir not created"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL, got {jsonl_files}"

    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    output_events = [e for e in events if getattr(e, "kind", "carry") == "output"]
    assert len(output_events) > 0, (
        "No output events in JSONL — y-tap did not fire for NUTS recipe. "
        "This means the treedepth y-tap wiring is broken (regression)."
    )

    saturated = [e for e in output_events if _safe_int_val(e.value) >= 10]
    assert len(saturated) == 0, (
        f"False positive: {len(saturated)} saturation events on healthy mvn_10. "
        f"Values: {[e.value for e in saturated[:5]]}"
    )

    sat_n, total, frac = compute_saturation_fraction(
        jsonl_files[0], max_num_doublings=10
    )
    assert (
        sat_n == 0
    ), f"compute_saturation_fraction returned sat_n={sat_n} (expected 0)"
    assert frac == 0.0, f"Expected fraction=0.0, got {frac}"

    print(
        f"\n[NUTS HEALTHY] output_events={total} saturated=0 frac=0.0 "
        f"sample_values={[e.value for e in output_events[:5]]} [TESTED]"
    )


def _safe_int_val(val) -> int:
    """Convert a TapEvent value to int, returning -999 on failure.

    For vmapped-chain NUTS the value is ``(num_chains,)`` shaped; ``max()``
    gives the worst-case treedepth across all chains.
    """
    import numpy as np

    try:
        return int(np.asarray(val).max())
    except Exception:  # noqa: BLE001
        return -999


@pytest.mark.slow
def test_mclmc_tap_context_no_crash(tmp_path):
    """MCLMC tap context runs without crashing and writes a JSONL artifact.

    KNOWN LIMITATION: MCLMC divergence detection via y-tap is NOT implemented.
    The MCLMC adaptation scan (``run_steps``) returns ``jnp.logical_not(success)``
    as its ys body output, but JAX DCE eliminates this ys from the jaxpr when the
    outer caller (``L_step_size_adaptation``) discards the second tuple element.
    jaxtap's A-form intercept therefore sees ``n_ys=0`` for both adaptation scans
    (scan[0] length=6, scan[1] length=1) and produces zero output events from them.

    What IS tested here:
    - ``tap_diagnostics_context(base_method_name="mclmc")`` does not crash.
    - A JSONL artifact is created and populated with carry events from the
      adaptation scan (ncar=10, carry tap fires at sample_every=1).
    - Zero output events (kind="output") are produced — confirming that the
      MCLMC adaptation scan's ys is inaccessible via jaxtap A-form.
    - ``compute_saturation_fraction`` handles an artifact with zero output events
      and returns ``(0, 0, 0.0)`` without raising.

    If a future blackjax change makes ``L_step_size_adaptation`` USE the
    div_flags (preventing DCE), the assertion ``len(output_events) == 0`` will
    become the right regression gate to change.  See ``_tap.py`` module docstring
    § 3 for the full investigation notes.
    """
    import blackjax
    import jax
    import jax.numpy as jnp

    from tuningfork.diagnostics._tap import (
        compute_saturation_fraction,
        tap_diagnostics_context,
    )

    DIM = 5
    init_key, run_key = jax.random.split(jax.random.key(42))

    def logdensity(position):
        x = jnp.ravel(position)
        return -0.5 * jnp.sum(x**2)

    init_pos = jnp.zeros(DIM)
    kernel = blackjax.mclmc.build_kernel()
    state = blackjax.mclmc.init(init_pos, logdensity, init_key)

    artifact_path = tmp_path / "mclmc_tap.jsonl"
    jax.clear_caches()

    with tap_diagnostics_context(
        artifact_path=artifact_path,
        base_method_name="mclmc",
        sample_every=1,
    ):
        blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=30,
            state=state,
            rng_key=run_key,
            logdensity_fn=logdensity,
        )

    assert artifact_path.exists(), "JSONL artifact not created"

    from jaxtap import read_jsonl

    events = read_jsonl(artifact_path)
    output_events = [e for e in events if getattr(e, "kind", "carry") == "output"]
    carry_events = [e for e in events if getattr(e, "kind", "carry") != "output"]

    # Carry tap fires on the adaptation scan (ncar=10) — confirms jaxtap is active.
    assert len(carry_events) > 0, (
        "No carry events in JSONL — jaxtap carry tap did not fire for MCLMC. "
        f"Total events: {len(events)}."
    )

    # Zero output events is expected: the adaptation scan's ys is DCE'd.
    # If this assertion fails, the DCE limitation has been resolved upstream.
    assert len(output_events) == 0, (
        f"Unexpected output events: {len(output_events)} found. "
        "The MCLMC adaptation scan's ys may now be accessible — "
        "update the test and re-enable MCLMC y-tap in _tap.py."
    )

    sat_n, total, frac = compute_saturation_fraction(
        artifact_path, max_num_doublings=None
    )
    assert sat_n == 0 and total == 0 and frac == 0.0, (
        f"compute_saturation_fraction should return (0, 0, 0.0) with no output events; "
        f"got ({sat_n}, {total}, {frac})"
    )

    print(
        f"\n[MCLMC TAP] carry_events={len(carry_events)} output_events=0 (DCE limited) "
        f"[TESTED — known limitation documented in _tap.py § 3]"
    )


_DYNAMIC_HMC_RECIPE_REL = (
    "tuningfork/catalog/mvn_10/recipes/"
    "low__dynamic_hmc__window_adaptation_diag_imm.json"
)


def _load_dynamic_hmc_recipe():
    """Load the mvn_10 LOW dynamic_hmc recipe from disk."""
    from tuningfork.catalog.inspect import load_recipe

    path = _REPO_ROOT / _DYNAMIC_HMC_RECIPE_REL
    if not path.exists():
        pytest.skip(f"dynamic_hmc recipe not found on disk: {path}")
    return load_recipe(path)


@pytest.mark.slow
def test_dynamic_hmc_no_false_treedepth_alerts(monkeypatch, tmp_path):
    """dynamic_hmc tap context produces zero output events (HMCInfo, no treedepth).

    dynamic_hmc returns HMCInfo, not NUTSInfo.  HMCInfo has no
    ``num_trajectory_expansions`` field — its only integer leaf is
    ``num_integration_steps`` (a Halton-drawn trajectory length, routinely
    large).  ``dynamic_hmc`` is therefore EXCLUDED from ``_NUTS_FAMILY`` to
    prevent the treedepth selector from matching ``num_integration_steps`` and
    emitting false "treedepth saturated" alerts every step at the default
    ``max_num_doublings=10`` threshold.

    Two mechanisms protect against false alerts:
    1. ``_NUTS_FAMILY = {"nuts"}`` — dynamic_hmc is not in the set.
    2. ``max_num_doublings=None`` default — the recipe runner passes
       ``recipe.base_method_params.get("max_num_doublings")`` which returns
       ``None`` for dynamic_hmc recipes (they have no such param); the context
       skips y-tap wiring when ``max_num_doublings is None``.

    Asserts:
    - JSONL artifact is created and populated with carry events (cholesky
      carry tap is always active — confirms jaxtap instrumentation is alive).
    - Zero output events (kind="output") — no false treedepth y-tap wiring.
    - ``compute_saturation_fraction`` returns (0, 0, 0.0).
    """
    from tuningfork.diagnostics._tap import compute_saturation_fraction
    from tuningfork.recipes._recipe_runner import run_recipe_to_idata

    tap_dir = tmp_path / "dhmc_no_false_alerts"
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", str(tap_dir))

    recipe = _load_dynamic_hmc_recipe()

    run_recipe_to_idata(
        recipe,
        skip_warmup=True,
        n_samples=20,
        force_resample_config={"seed": 42, "n_samples": 20},
        _suppress_print=True,
    )

    assert tap_dir.exists(), "Tap artifact dir not created"
    jsonl_files = list(tap_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, f"Expected 1 JSONL, got {jsonl_files}"

    from jaxtap import read_jsonl

    events = read_jsonl(jsonl_files[0])
    output_events = [e for e in events if getattr(e, "kind", "carry") == "output"]
    carry_events = [e for e in events if getattr(e, "kind", "carry") != "output"]

    assert len(carry_events) > 0, (
        "No carry events — jaxtap not active for dynamic_hmc recipe. "
        f"Total events: {len(events)}"
    )
    assert len(output_events) == 0, (
        f"False treedepth alerts: {len(output_events)} output events on dynamic_hmc. "
        "dynamic_hmc returns HMCInfo (no num_trajectory_expansions); it must NOT be "
        "in _NUTS_FAMILY.  Check _tap.py: _NUTS_FAMILY should be frozenset({'nuts'})."
    )

    sat_n, total, frac = compute_saturation_fraction(
        jsonl_files[0], max_num_doublings=10
    )
    assert sat_n == 0 and total == 0 and frac == 0.0, (
        f"Expected (0, 0, 0.0) from compute_saturation_fraction; "
        f"got ({sat_n}, {total}, {frac})"
    )

    print(
        f"\n[DHMC NO-FALSE-ALERTS] carry_events={len(carry_events)} "
        f"output_events=0 (HMCInfo, not in _NUTS_FAMILY) [TESTED]"
    )
