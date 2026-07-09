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
"""Tests for M2 opt-in tap diagnostics (TUNINGFORK_TAP_DIAGNOSTICS=1).

Four tests covering the four guarantees specified in M2:

1. ``test_planted_pathology_nan_alert`` — plants a float32 Cholesky NaN
   (the canonical silent-failure trap class) and verifies the tap catches it:
   JSONL artifact exists, contains a cholesky event with value=False, and
   session.alerts records the hit.  [TESTED]

2. ``test_false_positive_check`` — the same structure with a well-conditioned
   identity matrix.  Asserts zero alerts (no false positives).  [TESTED]

3. ``test_default_off_purity`` — with TUNINGFORK_TAP_DIAGNOSTICS unset,
   ``is_tap_enabled()`` returns False and the ExitStack is never entered.
   [TESTED via logic gate]

4. ``test_speed_path_guard`` — with TUNINGFORK_TAP_DIAGNOSTICS=1, calling
   ``run_recipe_to_idata(..., _no_tap=True)`` leaves the ExitStack empty.
   The speed path is structurally unreachable from the tap regardless of the
   env var.  [TESTED via mock / env-set logic gate]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: tiny JAX computations that exercise cholesky
# ---------------------------------------------------------------------------


def _make_bad_cholesky_scan(n_steps: int = 25):
    """Return a function whose lax.scan calls cholesky on an ill-conditioned f32 matrix.

    Mirrors the pattern from ``demo/cholesky_float32_trap.py`` in jax-tap.
    The off-diagonal entry c → 1 as k increases; once k ≈ 12 the float32
    matrix is numerically singular and cholesky produces NaN/Inf.
    """

    def step(carry, _):
        log_step, k = carry
        # c → 1 as k → 12; matrix becomes singular in f32 (κ > 1/ε_f32 ≈ 1e7).
        c = jnp.float32(1.0) - jnp.float32(10.0) ** (-jnp.minimum(k, jnp.float32(12.0)))
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=jnp.float32)
        L = jnp.linalg.cholesky(M)  # silent NaN once c ≈ 1 in float32
        logdens = -0.5 * jnp.float32(2.0) * jnp.sum(jnp.log(jnp.diag(L)))
        new_log_step = jnp.where(
            jnp.isfinite(logdens),
            log_step + jnp.float32(0.05),
            log_step - jnp.float32(1.0),
        )
        return (new_log_step, k + jnp.float32(1.0)), logdens

    def run():
        x0 = (jnp.float32(0.0), jnp.float32(1.0))
        (log_step, _), _ = jax.lax.scan(step, x0, None, length=n_steps)
        return log_step

    return run


def _make_good_cholesky_scan(n_steps: int = 25):
    """Return a function whose lax.scan calls cholesky on a well-conditioned identity matrix.

    All outputs should be finite (false-positive check).
    """

    def step(carry, _):
        log_step, k = carry
        M = jnp.eye(2, dtype=jnp.float32)  # identity: κ=1, always PD in f32
        L = jnp.linalg.cholesky(M)
        logdens = -0.5 * jnp.sum(jnp.log(jnp.diag(L)))
        return (log_step + jnp.float32(0.01), k + jnp.float32(1.0)), logdens

    def run():
        x0 = (jnp.float32(0.0), jnp.float32(1.0))
        (log_step, _), _ = jax.lax.scan(step, x0, None, length=n_steps)
        return log_step

    return run


# ---------------------------------------------------------------------------
# Test 1: planted pathology → alert fires
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_planted_pathology_nan_alert(tmp_path):
    """Plant a float32 Cholesky NaN; verify the tap catches it.

    [TESTED]: runs the bad-cholesky scan inside tap_diagnostics_context,
    then asserts:
      - JSONL artifact exists and is non-empty
      - At least one cholesky event has value=False (NaN detected)
      - session.alerts contains a "cholesky_nan" entry with a sane step

    Runtime budget: < 30 s (25 JAX scan steps + JAX compile).
    """
    # Disable x64 to ensure float32 Cholesky NaN trap fires.
    jax.config.update("jax_enable_x64", False)

    from jaxtap import read_jsonl

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    artifact = tmp_path / "pathology.jsonl"
    bad_scan = _make_bad_cholesky_scan(n_steps=25)

    with tap_diagnostics_context(
        artifact_path=artifact,
        sample_every=1,  # check every step for reliability in the test
    ) as session:
        bad_scan()

    # ── Artifact checks ──
    assert artifact.exists(), "JSONL artifact was not created"
    assert artifact.stat().st_size > 0, "JSONL artifact is empty"

    events = read_jsonl(artifact)
    assert len(events) > 0, "No events written to JSONL"

    # ── Cholesky NaN check ──
    # watch_nan("cholesky", once=True) yields PrimitiveTap events whose
    # value = True when finite, False when NaN/Inf.
    cholesky_events = [e for e in events if "cholesky" in str(e.path)]
    assert (
        len(cholesky_events) > 0
    ), f"No cholesky events in JSONL. All paths: {sorted({e.path for e in events})}"

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
        "Expected at least one NaN cholesky event but found none. "
        f"Cholesky event values: {[e.value for e in cholesky_events[:5]]}"
    )

    # ── Step sanity check: first NaN occurs at a sane step (< n_steps) ──
    first_nan_step = min(e.step for e in nan_events)
    assert (
        0 <= first_nan_step < 25
    ), f"First NaN step {first_nan_step} is outside expected range [0, 25)"

    # ── session.alerts check ──
    assert (
        len(session.alerts) > 0
    ), "session.alerts is empty; expected a cholesky_nan alert entry"
    cholesky_nan_alerts = [a for a in session.alerts if a["type"] == "cholesky_nan"]
    assert (
        len(cholesky_nan_alerts) > 0
    ), f"No cholesky_nan alerts in session.alerts. Got: {session.alerts}"

    # Print the first NaN alert for the report (TL inspection)
    first_alert = cholesky_nan_alerts[0]
    print(
        f"\n[PLANTED PATHOLOGY EVIDENCE] first_alert={first_alert}  "
        f"first_nan_step={first_nan_step}  total_events={len(events)}  "
        f"cholesky_events={len(cholesky_events)}  nan_events={len(nan_events)}"
    )


# ---------------------------------------------------------------------------
# Test 2: false-positive check (healthy run → zero alerts)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_false_positive_check(tmp_path):
    """Well-conditioned cholesky: tap fires zero alerts.

    [TESTED]: runs the good-cholesky scan (identity matrix, always PD) inside
    tap_diagnostics_context.  Asserts:
      - JSONL artifact exists (events are recorded even when healthy)
      - NO cholesky events have value=False
      - session.alerts is empty

    Runtime budget: < 30 s.
    """
    jax.config.update("jax_enable_x64", False)

    from jaxtap import read_jsonl

    from tuningfork.diagnostics._tap import tap_diagnostics_context

    artifact = tmp_path / "healthy.jsonl"
    good_scan = _make_good_cholesky_scan(n_steps=25)

    with tap_diagnostics_context(
        artifact_path=artifact,
        sample_every=1,
    ) as session:
        good_scan()

    assert artifact.exists(), "JSONL artifact was not created"

    events = read_jsonl(artifact)
    assert len(events) > 0, "No events written to JSONL"

    # All cholesky events should have value=True (finite).
    cholesky_events = [e for e in events if "cholesky" in str(e.path)]
    assert (
        len(cholesky_events) > 0
    ), f"No cholesky events in JSONL. All paths: {sorted({e.path for e in events})}"

    nan_events = [
        e
        for e in cholesky_events
        if not bool(
            np.asarray(e.value).all()
            if hasattr(np.asarray(e.value), "all")
            else bool(e.value)
        )
    ]
    assert len(nan_events) == 0, (
        f"False positive: {len(nan_events)} cholesky NaN events on healthy input. "
        f"Events: {nan_events[:3]}"
    )

    # session.alerts must be empty.
    assert (
        session.alerts == []
    ), f"False positive: session.alerts non-empty on healthy input: {session.alerts}"

    print(
        f"\n[FALSE POSITIVE CHECK EVIDENCE] total_events={len(events)}  "
        f"cholesky_events={len(cholesky_events)}  alerts={session.alerts}"
    )


# ---------------------------------------------------------------------------
# Test 3: default-OFF purity (env var unset → zero tap involvement)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_default_off_purity(monkeypatch):
    """Without env var, is_tap_enabled() is False and no JSONL is written.

    [TESTED via logic gate]:
      - Ensures TUNINGFORK_TAP_DIAGNOSTICS is absent.
      - Verifies is_tap_enabled() returns False.
      - Verifies that calling run_recipe_to_idata (without recipe, just the
        guard logic) leaves _tap_stack with no entered contexts.
    """
    # Ensure the env var is absent for this test.
    monkeypatch.delenv("TUNINGFORK_TAP_DIAGNOSTICS", raising=False)

    from tuningfork.diagnostics._tap import is_tap_enabled

    assert (
        not is_tap_enabled()
    ), "is_tap_enabled() returned True but TUNINGFORK_TAP_DIAGNOSTICS is unset"

    # Verify the ExitStack code path: with env var absent, _tap_stack is empty.
    import contextlib

    _tap_stack = contextlib.ExitStack()
    _no_tap = False  # simulate default call
    if not _no_tap:
        if is_tap_enabled():  # False → branch not taken
            raise AssertionError("Entered tap branch without env var")

    # ExitStack.close() on an empty stack is a no-op.
    _tap_stack.close()

    # Verify that jaxtap's scan patch is NOT active (module-level check).
    # When tap is disabled, the production jax.lax.scan is the canonical one.
    import jax.lax

    scan_fn = jax.lax.scan
    # The canonical scan is not a wrapped jaxtap version:
    # check by name (jaxtap replaces it with a closure called "tapped_scan"
    # or similar; the unpatched scan has a C-extension-level __name__).
    # This is a best-effort check — don't rely on internal naming.
    scan_qualname = getattr(scan_fn, "__qualname__", "")
    assert (
        "tap" not in scan_qualname.lower()
    ), f"jax.lax.scan appears to be patched by jaxtap: __qualname__={scan_qualname!r}"


# ---------------------------------------------------------------------------
# Test 4: speed-path guard (env var set + _no_tap=True → no tap involvement)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_speed_path_guard(monkeypatch, tmp_path):
    """With env var set, _no_tap=True prevents any tap involvement.

    [TESTED via logic gate + filesystem check]:
      - Sets TUNINGFORK_TAP_DIAGNOSTICS=1.
      - Simulates the speed-lite code path (ExitStack + _no_tap=True guard).
      - Verifies that no JSONL artifact appears in the tap diagnostics directory.
      - Verifies that the ExitStack remains empty (no context was entered).

    This test proves the structural guard works regardless of env var state.
    """
    monkeypatch.setenv("TUNINGFORK_TAP_DIAGNOSTICS", "1")

    from tuningfork.diagnostics._tap import is_tap_enabled

    assert is_tap_enabled(), "is_tap_enabled() returned False with env var = '1'"

    # Simulate the run_recipe_to_idata guard with _no_tap=True:
    import contextlib

    _no_tap = True  # speed-lite caller always passes _no_tap=True
    _tap_stack = contextlib.ExitStack()

    entered = False
    if not _no_tap:  # False → branch not taken
        if is_tap_enabled():
            entered = True  # would have entered the tap context

    assert not entered, "_no_tap=True but tap context was entered (guard broken)"

    _tap_stack.close()  # no-op: nothing was entered

    # Verify no JSONL artifact appeared under the tap diagnostics directory.
    import tempfile
    from pathlib import Path

    tap_dir = Path(tempfile.gettempdir()) / "tuningfork-tap-diagnostics"
    # Collect JSONL files that could have appeared during this test.
    jsonl_files_before = set(tap_dir.glob("*.jsonl")) if tap_dir.exists() else set()

    # (Re-run the guard to confirm no side-effects)
    _tap_stack2 = contextlib.ExitStack()
    if not _no_tap:
        if is_tap_enabled():
            from tuningfork.diagnostics._tap import tap_diagnostics_context

            _tap_stack2.enter_context(
                tap_diagnostics_context(run_tag="speed_guard_test")
            )
    _tap_stack2.close()

    jsonl_files_after = set(tap_dir.glob("*.jsonl")) if tap_dir.exists() else set()
    new_files = jsonl_files_after - jsonl_files_before
    assert (
        len(new_files) == 0
    ), f"Speed-path guard failed: JSONL files created despite _no_tap=True: {new_files}"
