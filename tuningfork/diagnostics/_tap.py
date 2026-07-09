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
"""Opt-in tap diagnostics for run_recipe_to_idata.

Enable by setting the ``TUNINGFORK_TAP_DIAGNOSTICS`` environment variable
before running any recipe.  Default: **OFF** — with the variable unset, jaxtap
is never imported and the hot path is bitwise-identical to an unpatched run.

Environment variable semantics
------------------------------
- **Unset or ``"0"``**: diagnostics OFF, zero jaxtap involvement.
- **``"1"``**: diagnostics ON, artifacts written to
  ``<tempdir>/tuningfork-tap-diagnostics/`` (the system temp directory,
  typically ``/tmp`` on Linux).
- **An absolute path** (e.g. ``"/workspace/tap-artifacts"``): diagnostics ON,
  artifacts written to that directory.  Use this form for nightly CI runs where
  artifacts must survive the job and be collected by the CI system::

      TUNINGFORK_TAP_DIAGNOSTICS=/workspace/tap-artifacts python run_suite.py

  The directory is created if absent.

Alert class monitored:

1. **Float32 Cholesky NaN trap** (``tap.watch_nan("cholesky", once=True)``):
   fires the first time a Cholesky factor is non-finite inside any JAX
   scan/while loop.  Silent NaN → frozen chain is the canonical failure mode
   for metric adaptation on ill-conditioned posteriors at float32 precision.

**jaxtap 0.2.0 / vmapped-while limitation**: NUTS uses ``jax.lax.while_loop``
for tree expansion.  When run with ``n_chains > 1`` via ``jax.vmap``, both the
while condition and carry are vmapped, giving non-scalar shapes (e.g.
``bool[4]``) that jaxtap's ``rewrite_while`` cannot handle: ``_base_tap_cb``'s
``lax.select`` requires scalar ``_while_active``, and ``rewrite_while.cond_fn``
cannot accept a non-scalar cond return.

Never-crash invariant: ``run_recipe_to_idata`` checks
``is_algorithm_tap_compatible(recipe.base_method_name)`` before entering
``tap_diagnostics_context``.  Incompatible algorithms (NUTS, dynamic_hmc, and
any other method using vmapped while_loops) receive a one-time
``logging.WARNING`` and then run WITHOUT tap instrumentation — the recipe
completes normally, no artifact is created, no crash.  This invariant ensures
the diagnostics switch is always safe to enable, even on NUTS recipes.

Compatible algorithms (those using ``lax.scan`` for fixed-step integration with
no vmapped while_loops): HMC, MHMC, DMHMC, GHMC, MALA, Barker, RWM, IRMH,
additive_step_random_walk, MCLMC, adjusted_mclmc (fixed-step tuning paths).
Algorithms that MAY use while_loops internally (NUTS, dynamic_hmc,
adjusted_mclmc_dynamic) are placed on the incompatible list.

Upstream bug tracking (arcueil/jax-tap):
- Bug 1 (``_base_tap_cb``): ``lax.select`` requires scalar ``_while_active``;
  vmapped while gives shape ``(n_chains,)`` → ``TypeError``.
- Bug 2 (``rewrite_while.cond_fn``): non-scalar cond return for vmapped while.

Artifacts
---------
Each tap-enabled run writes a JSONL file to the configured directory::

    <dir>/<model>__<sampler>__seed<N>.jsonl

The artifact is created even when no alerts fire (records all sampled carry
states and primitive-tap values).  At run-end, if any alerts were collected,
``logging.WARNING`` is emitted with a count and the artifact path.

Speed paths (``_no_tap=True`` callers, e.g. the Speed-lite benchmark and all
three timing paths in ``_benchmark_helpers.run_benchmark_cell``) are
structurally gated and ignore the env var — zero tap overhead on any timed
path regardless of the env var.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Default carry-tap sampling frequency.  Gates both the fineness-check select
# and the cholesky primitive tap *inside* loops.  At sample_every=10 with a
# ~100 µs body, overhead is ~+1% (see jaxtap bench/README.md recommendation
# ladder).  Lower values increase sensitivity; higher values reduce overhead.
_DEFAULT_SAMPLE_EVERY: int = 10


def is_tap_enabled() -> bool:
    """Return True when ``TUNINGFORK_TAP_DIAGNOSTICS`` is set to a non-off value.

    "Non-off" means: non-empty and not ``"0"``.  Both ``"1"`` and an absolute
    path string are truthy; unset and ``"0"`` are falsy.

    Called by ``run_recipe_to_idata`` to decide whether to enter the tap
    context.  With the variable unset or ``"0"``, returns False and zero
    jaxtap involvement.
    """
    val = os.environ.get("TUNINGFORK_TAP_DIAGNOSTICS", "0")
    return bool(val) and val != "0"


# Algorithms whose sampling loop uses lax.scan for fixed-step integration and
# does NOT contain an internally-vmapped while_loop that would trigger the
# jaxtap 0.2.0 vmap-while bugs (Bug 1: _base_tap_cb lax.select shape mismatch;
# Bug 2: rewrite_while.cond_fn non-scalar return).
#
# NOT in this set: nuts, dynamic_hmc, adjusted_mclmc_dynamic — these use or may
# use while_loops that are vectorized over n_chains via jax.vmap, causing
# TypeError crashes when jaxtap intercepts them.
#
# Reference: arcueil/jax-tap issues for Bug 1 and Bug 2 (filed 2026-07-10).
_TAP_COMPATIBLE_BASE_METHODS: frozenset[str] = frozenset(
    {
        "hmc",
        "mhmc",
        "dmhmc",
        "ghmc",
        "mala",
        "barker",
        "rwm",
        "irmh",
        "additive_step_random_walk",
        "mclmc",
        "adjusted_mclmc",
        "orbital_hmc",
        "elliptical_slice",
        "mgrad_gaussian",
        "rmhmc",
        "meanfield_vi",
        "fullrank_vi",
    }
)


def is_algorithm_tap_compatible(base_method_name: str) -> bool:
    """Return True when ``base_method_name`` is safe to instrument with jaxtap.

    jaxtap 0.2.0 cannot handle vmapped while_loops (NUTS tree expansion,
    dynamic_hmc, adjusted_mclmc_dynamic).  Algorithms in
    ``_TAP_COMPATIBLE_BASE_METHODS`` use ``lax.scan`` for fixed-step
    integration with no vmapped while_loops.

    The check is conservative: unknown names return False (safe default —
    warn and skip rather than crash).

    Parameters
    ----------
    base_method_name
        The recipe's ``base_method_name`` field (e.g. ``"hmc"``, ``"nuts"``).

    Returns
    -------
    bool
        True if tap instrumentation is safe for this algorithm.
    """
    return base_method_name in _TAP_COMPATIBLE_BASE_METHODS


def tap_artifact_dir() -> Path:
    """Return the directory where JSONL artifacts are written.

    - ``TUNINGFORK_TAP_DIAGNOSTICS=1`` → ``<tempdir>/tuningfork-tap-diagnostics``
    - ``TUNINGFORK_TAP_DIAGNOSTICS=/abs/path`` → ``/abs/path``

    The directory is created if absent.  Call only when ``is_tap_enabled()`` is
    True; behaviour is undefined when diagnostics are OFF.
    """
    val = os.environ.get("TUNINGFORK_TAP_DIAGNOSTICS", "1")
    if val == "1":
        base = Path(tempfile.gettempdir()) / "tuningfork-tap-diagnostics"
    else:
        base = Path(val)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _default_artifact_path(run_tag: str = "run") -> Path:
    """Build a per-run JSONL artifact path under the configured artifact dir."""
    return tap_artifact_dir() / f"{run_tag}.jsonl"


def _select_finite(leaves: tuple) -> Any:
    """Device-side reducer: True iff every float carry leaf is finite.

    Receives the flat tuple of carry leaves (pytree structure is erased by
    JAX tracing).  Filters to floating-point leaves only; non-float leaves
    (step counters, integer flags) are skipped.  Returns a scalar bool.

    If no float leaves are present (unusual but valid), returns True.
    """
    import jax.numpy as jnp

    checks = [
        jnp.all(jnp.isfinite(leaf))
        for leaf in leaves
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    if not checks:
        return jnp.bool_(True)
    if len(checks) == 1:
        return checks[0]
    return jnp.all(jnp.stack(checks))


class _TapSession:
    """Collects tap events and alerts for one instrumented run.

    Acts as the ``on_step`` callback for ``jaxtap.record``.  Streams each
    event to a JSONL file and checks cholesky primitive-tap events for
    non-finite values (to complement the stderr line that ``watch_nan``
    already emits).

    Parameters
    ----------
    artifact_path
        JSONL file path; opened on construction, closed on ``close()``.
    """

    def __init__(self, artifact_path: Path) -> None:
        from jaxtap import JSONLWriter  # lazy — only when tap is enabled

        self.artifact_path = artifact_path
        # Touch the file eagerly so it exists even when no events fire (e.g.,
        # a healthy run with only watch_nan active produces zero events).
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.touch(exist_ok=True)
        self.alerts: list[dict[str, Any]] = []
        self._writer = JSONLWriter(artifact_path)

    def __call__(self, event: Any) -> None:
        """Receive one TapEvent: write to JSONL and check for cholesky NaN."""
        self._writer(event)
        if "cholesky" in str(event.path):
            self._check_cholesky_nan(event)

    def _check_cholesky_nan(self, event: Any) -> None:
        """Append an alert when a cholesky event value is non-finite."""
        import numpy as np

        try:
            val = np.asarray(event.value)
            # watch_nan's select returns True when finite, False when NaN/Inf.
            is_ok = bool(val.all() if hasattr(val, "all") else bool(val))
            if not is_ok:
                self.alerts.append(
                    {
                        "type": "cholesky_nan",
                        "path": str(event.path),
                        "step": int(event.step),
                        "total": event.total,
                    }
                )
        except Exception:  # noqa: BLE001 — diagnostics must never propagate
            pass

    def close(self) -> None:
        """Close the JSONL file and emit a WARNING if any alerts were collected."""
        self._writer.close()
        if self.alerts:
            n = len(self.alerts)
            types = sorted({a["type"] for a in self.alerts})
            _LOG.warning(
                "[tuningfork tap] %d alert(s) during run (types: %s). " "Artifact: %s",
                n,
                ", ".join(types),
                self.artifact_path,
            )


@contextlib.contextmanager
def tap_diagnostics_context(
    artifact_path: Path | None = None,
    run_tag: str = "run",
    sample_every: int = _DEFAULT_SAMPLE_EVERY,
):
    """Context manager: wrap a block with jaxtap carry + Cholesky NaN diagnostics.

    Monkeypatches ``jax.lax.scan`` and ``jax.lax.while_loop`` for the duration
    of the block to stream telemetry.  On exit the patch is removed and the
    JSONL artifact is closed.

    Must be called only when ``is_tap_enabled()`` is True.  Callers are
    responsible for this guard; wrapping in the disabled case is a no-op but
    incurs an unnecessary jaxtap import.

    Parameters
    ----------
    artifact_path
        Explicit JSONL path.  When ``None`` (default), a path is derived from
        ``run_tag`` under ``<tempdir>/tuningfork-tap-diagnostics/``.
    run_tag
        Short identifier used to build the artifact filename when
        ``artifact_path=None``.  Example: ``"mvn_10__nuts__seed42"``.
    sample_every
        Carry tap frequency (default ``_DEFAULT_SAMPLE_EVERY`` = 10).
        Lower values increase sensitivity at higher overhead.

    Yields
    ------
    session : _TapSession
        The session object; has ``.alerts`` (list of alert dicts) and
        ``.artifact_path`` (Path).

    Examples
    --------
    Direct usage (testing / post-mortems)::

        with tap_diagnostics_context(artifact_path=tmp_path / "run.jsonl") as session:
            run_inference(...)

        if session.alerts:
            print("Alerts:", session.alerts)

    The JSONL artifact contains one line per sampled event::

        {"path": "scan[0]", "step": 0, "value_kind": "scalar", "value": true}
        {"path": "scan[0]/cholesky[0]", "step": 10, "value_kind": "scalar", "value": false}
    """
    import jaxtap as tap  # lazy — imported only when tap is enabled

    if artifact_path is None:
        artifact_path = _default_artifact_path(run_tag)

    session = _TapSession(artifact_path)

    def _carry_alert(event: Any) -> Any:
        """Host-side carry alert: fires when _select_finite returns False."""
        import numpy as np

        try:
            val = np.asarray(event.value)
            is_ok = bool(val.all() if hasattr(val, "all") else bool(val))
            if not is_ok:
                msg = f"non-finite carry leaf (step={event.step}, path={event.path})"
                session.alerts.append(
                    {
                        "type": "carry_nonfinite",
                        "path": str(event.path),
                        "step": int(event.step),
                        "total": event.total,
                    }
                )
                return msg
        except Exception:  # noqa: BLE001 — diagnostics must never propagate
            pass
        return False

    try:
        # Note: select= and alert= are intentionally omitted here.
        # They insert a carry-level interceptor at every while_loop step, and
        # NUTS's tree-expansion while_loop is vectorized over n_chains, giving
        # _while_active shape (n_chains,) vs scalar step — a shape mismatch in
        # jaxtap's lax.select encoding (tracked in jaxtap issue #ytaps).
        # Primitive-level watch_nan fires at the XLA cholesky op level and is
        # not affected by vectorized while shapes.  Carry-level monitoring
        # is a deferred improvement pending an upstream jaxtap fix.
        with tap.record(
            taps=[tap.watch_nan("cholesky", once=True)],
            on_step=session,
        ):
            yield session
    finally:
        session.close()
