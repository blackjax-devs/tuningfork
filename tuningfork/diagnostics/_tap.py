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
"""Opt-in tap diagnostics for generated recipe execution.

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

Alert classes monitored
-----------------------
1. **Float32 Cholesky NaN trap** (``tap.watch_nan("cholesky", once=True)``):
   fires the first time a Cholesky factor is non-finite inside any JAX
   scan/while loop.  Silent NaN → frozen chain is the canonical failure mode
   for metric adaptation on ill-conditioned posteriors at float32 precision.

2. **NUTS treedepth saturation** (y-tap, jax-tap 0.3.0): fires when the
   ``num_trajectory_expansions`` field of ``NUTSInfo`` (the scan ys) reaches
   ``max_num_doublings``.  Saturation = NUTS stopped NOT because the U-turn
   criterion fired, but because it hit the hard cap — evidence the trajectory
   length budget is insufficient for the posterior's geometry.  The tripwire
   gate is OFFLINE (policy-free); ``compute_saturation_fraction`` reads the
   JSONL and returns the fraction; the nightly/cert consumer decides thresholds.

   **Scope — NUTSInfo-returning methods only** (``_NUTS_FAMILY = {"nuts"}``):
   ``num_trajectory_expansions`` is a field of ``NUTSInfo`` and does NOT exist
   in any other info type in the 24-method inventory (verified 2026-07-10;
   see ``_NUTS_FAMILY`` comment for the full audit table).  ``dynamic_hmc``
   was previously included in error — it returns ``HMCInfo`` whose only int
   leaf is ``num_integration_steps`` (Halton-drawn trajectory length, routine
   large values), which the ``ndim ≤ 1`` selector would have matched, producing
   false "treedepth saturated" alerts every step at the default
   ``max_num_doublings=10`` threshold.  It is now excluded.  The treedepth
   y-tap is only armed when ``base_method_name in {"nuts"}`` AND
   ``max_num_doublings is not None`` (see ``tap_diagnostics_context``).

   **How the treedepth leaf is identified** (documented for audit — applies
   to ``nuts`` only, which returns ``NUTSInfo`` via ``run_inference_algorithm``):
   ``run_inference_algorithm`` scans over steps and returns
   ``(state, info)`` as the per-step ys where ``state`` is ``HMCState`` and
   ``info`` is ``NUTSInfo``.  Flattening ``(HMCState, NUTSInfo)`` with
   ``jax.tree_util.tree_leaves`` yields a tuple whose FIRST ``int32`` scalar
   is always ``NUTSInfo.num_trajectory_expansions`` — this holds for any
   position pytree shape because ``HMCState`` contains only float leaves, and
   within ``NUTSInfo`` the bool leaves (``is_divergent``, ``is_turning``) have
   ``bool`` dtype (not ``int32``), so the first ``int32`` scalar is
   unambiguously ``num_trajectory_expansions``.  Verified by a probe run on
   mvn_10 (d=10, dict position): 18 total leaves, integer scalars at indices
   15 (num_trajectory_expansions) and 16 (num_integration_steps).  The ordering
   is a JAX pytree traversal guarantee (NamedTuples are traversed
   field-by-field in declaration order).

   The selector returns a sentinel ``jnp.int32(-1)`` when no integer leaf with
   ndim ≤ 1 is found (i.e. the kernel's ys doesn't include a treedepth field).
   The alert predicate guards against the sentinel, so non-NUTS kernels never
   fire treedepth alerts.

   **Vmapped-chain shape**: when the recipe runner vmaps over ``num_chains``
   chains, each scan step returns ``infos.num_trajectory_expansions`` with
   shape ``(num_chains,)`` rather than ``()``.  The selector therefore checks
   ``leaf.ndim <= 1`` (scalar or 1-D) rather than ``leaf.shape == ()``, and
   ``alert_ys`` uses ``np.asarray(val).max()`` to obtain the worst-case
   treedepth across all chains in that step.

3. **MCLMC warmup divergence flag** (y-tap, ``mclmc`` only, blackjax ≥ 1.6):
   fires when the per-step divergence flag ``jnp.logical_not(success)`` is
   True inside the adaptation scan of ``mclmc_find_L_and_step_size``.  The
   seam was added in blackjax#975 (``mclmc_adaptation.py:301``) and ships in
   blackjax 1.6.

   **Scope — mclmc only**: ``adjusted_mclmc`` and ``adjusted_mclmc_dynamic``
   call ``adjusted_mclmc_find_L_and_step_size``, whose scan ys is
   ``(info, state_position)`` — not a bool divergence flag.  They receive
   only the cholesky NaN carry tap.  Verified against blackjax 1.6 source
   (``adjusted_mclmc_adaptation.py:303-311``).

   **The ndim ≤ 1 guard in select_ys is load-bearing**: the ESS autocorrelation
   sub-scan emits a ``(1, DIM)`` bool ys (ndim=2) that a naive dtype check
   false-positives on.  The guard rejects it while accepting the
   adaptation-flag scalar (ndim=0) and any vmapped-chain vector (ndim=1).
   Verified via ``/tmp/tap-dce-repro/repro2.py`` (2026-07-10).

**jaxtap 0.2.x / vmapped-while history (FIXED in 0.2.1)**:
NUTS uses ``jax.lax.while_loop`` for tree expansion.  When run with
``n_chains > 1`` via ``jax.vmap``, the vmapped while_loop crashed in two
places in 0.2.0: Bug 1 (``_base_tap_cb`` ``lax.select`` shape mismatch) and
Bug 2 (``rewrite_while.cond_fn`` non-scalar cond return).  Both fixed in
``jax-tap 0.2.1`` (arcueil/jax-tap#5).  All algorithms are now in
``_TAP_COMPATIBLE_BASE_METHODS`` and receive full instrumentation.

**Upstream bug tracking (arcueil/jax-tap)**:
- Bug 1 (``_base_tap_cb``): ``lax.select`` shape mismatch — FIXED in 0.2.1.
- Bug 2 (``rewrite_while.cond_fn``): non-scalar cond return — FIXED in 0.2.1.

Never-crash invariant (permanent design, applies to UNKNOWN algorithms):
generated programs check ``is_algorithm_tap_compatible`` before
entering ``tap_diagnostics_context``.  Unknown or future algorithms default
to False (warn-and-skip rather than crash) until explicitly added to the
allowlist.  This guard stays even after the 0.2.1 fix — it protects against
new upstream regressions and future unregistered methods.

vmap×while per-lane semantics (jax-tap 0.2.1): when a vmapped while_loop
is instrumented, each lane (chain) emits its own events independently.
A 4-chain NUTS run with depth-D trees emits up to 4 × D events per sample
from the tree-expansion while_loop.  Use ``sample_every`` to control volume
for long runs (default ``_DEFAULT_SAMPLE_EVERY`` = 10 applies to scan;
while_loop events are not subject to ``sample_every`` gating in 0.2.1).

Artifacts
---------
Each tap-enabled run writes a JSONL file to the configured directory::

    <dir>/<model>__<sampler>__seed<N>.jsonl

The artifact is created even when no alerts fire (records all sampled carry
states and primitive-tap values).  At run-end, if any alerts were collected,
``logging.WARNING`` is emitted with a count and the artifact path.

Speed paths pass ``diagnostics=False`` to :func:`tuningfork.catalog.execute_recipe`;
they are structurally gated and ignore the environment variable, so timed paths
have zero tap overhead.
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

# Algorithm families for y-tap wiring.
# Families are disjoint; unknown methods get no y-tap (safe default).
#
# Audit of all 24 in-scope base methods for NUTSInfo (which carries
# num_trajectory_expansions) — verified against blackjax source 2026-07-10:
#
#   nuts                → NUTSInfo         ← ONLY method with num_trajectory_expansions
#   dynamic_hmc         → HMCInfo          (no num_trajectory_expansions; a draw-length
#                                           int leaves num_integration_steps instead)
#   hmc, mhmc, rmhmc   → HMCInfo
#   dmhmc               → HMCInfo (via dynamic_hmc kernel with multinomial proposal)
#   ghmc                → HMCInfo
#   adjusted_mclmc,
#   adjusted_mclmc_dynamic → HMCInfo
#   laplace_hmc/dhmc/
#   mhmc/dmhmc          → LaplaceHMCInfo (HMCInfo subtype, no treedepth field)
#   mclmc               → MCLMCInfo
#   mala, barker, rwm,
#   irmh, additive_step_random_walk,
#   orbital_hmc,
#   elliptical_slice,
#   mgrad_gaussian      → own Info NamedTuples, no treedepth field
#   meanfield_vi,
#   fullrank_vi         → VI info types, no treedepth field
#
# Conclusion: num_trajectory_expansions exists in NUTSInfo ONLY.
# Source ref: blackjax/mcmc/nuts.py:36 (NUTSInfo), :72 (num_trajectory_expansions field).
# grep -rn "num_trajectory_expansions" blackjax/ → nuts.py:56 + :72, no other matches.
_NUTS_FAMILY: frozenset[str] = frozenset({"nuts"})

# MCLMC divergence y-tap: only "mclmc" calls mclmc_find_L_and_step_size, whose
# adaptation scan returns jnp.logical_not(success) as the per-step ys (seam added
# in blackjax#975, mclmc_adaptation.py:301).  Available from blackjax 1.6.
#
# adjusted_mclmc and adjusted_mclmc_dynamic call
# adjusted_mclmc_find_L_and_step_size, whose scan ys is (info, state_position)
# — not a bool divergence flag.  They get no MCLMC y-tap.
# Verified against blackjax 1.6 source (adjusted_mclmc_adaptation.py:303-311).
_MCLMC_FAMILY: frozenset[str] = frozenset({"mclmc"})


def is_tap_enabled() -> bool:
    """Return True when ``TUNINGFORK_TAP_DIAGNOSTICS`` is set to a non-off value.

    "Non-off" means: non-empty and not ``"0"``.  Both ``"1"`` and an absolute
    path string are truthy; unset and ``"0"`` are falsy.

    Called by generated recipe programs to decide whether to enter the tap
    context.  With the variable unset or ``"0"``, returns False and zero
    jaxtap involvement.
    """
    val = os.environ.get("TUNINGFORK_TAP_DIAGNOSTICS", "0")
    return bool(val) and val != "0"


# Algorithms known to be safe for jaxtap instrumentation.
#
# As of jax-tap 0.2.1, ALL 24 in-scope base methods are compatible:
# the 0.2.0 vmapped-while bugs (Bug 1 + Bug 2) that previously excluded
# nuts / dynamic_hmc / adjusted_mclmc_dynamic are fixed in 0.2.1
# (arcueil/jax-tap#5, released 2026-07-10).
#
# The guard mechanism is PERMANENT: unknown algorithm names return False
# (warn-and-skip) by default to protect against new upstream regressions
# and future unregistered methods not yet in the allowlist.  Only add methods
# here after verifying they work with tap.record() on vmapped kernels.
_TAP_COMPATIBLE_BASE_METHODS: frozenset[str] = frozenset(
    {
        # HMC family (fixed-step leapfrog via lax.scan)
        "hmc",
        "mhmc",
        "dmhmc",
        "ghmc",
        # NUTS / dynamic family (vmapped while_loop; safe since jax-tap 0.2.1)
        "nuts",
        "dynamic_hmc",
        # MALA / RWM family
        "mala",
        "barker",
        "rwm",
        "irmh",
        "additive_step_random_walk",
        # MCLMC family
        "mclmc",
        "adjusted_mclmc",
        "adjusted_mclmc_dynamic",  # vmapped while; safe since jax-tap 0.2.1
        # Other
        "orbital_hmc",
        "elliptical_slice",
        "mgrad_gaussian",
        "rmhmc",
        # Laplace marginal family (HMC-based sampling kernels)
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
        # VI family
        "meanfield_vi",
        "fullrank_vi",
    }
)


def is_algorithm_tap_compatible(base_method_name: str) -> bool:
    """Return True when ``base_method_name`` is safe to instrument with jaxtap.

    All 24 in-scope base methods are in ``_TAP_COMPATIBLE_BASE_METHODS`` as of
    ``jax-tap 0.2.1``.  Previously, ``nuts``, ``dynamic_hmc``, and
    ``adjusted_mclmc_dynamic`` were excluded due to vmapped-while_loop bugs
    (arcueil/jax-tap#5), fixed in 0.2.1.

    The check is conservative: **unknown names return False** (safe default —
    warn and skip rather than crash).  This protects against new upstream
    regressions and future unregistered methods not yet in the allowlist.

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


def _make_nuts_ys_wiring(
    max_num_doublings: int,
) -> tuple[Any, Any, bool]:
    """Build (select_ys, alert_ys, alert_ys_once) for NUTS treedepth saturation.

    ``select_ys`` scans the flat ys leaves for the first ``int32`` leaf with
    ndim ≤ 1 (scalar OR 1-D array), which is always
    ``NUTSInfo.num_trajectory_expansions`` (the treedepth).  The ndim ≤ 1
    check handles both single-chain (shape ``()``) and vmapped multi-chain
    (shape ``(num_chains,)``) cases.  See the module docstring for the full
    proof that this identification is robust across all model position shapes.

    Returns a sentinel ``jnp.int32(-1)`` when no integer scalar is found
    (non-NUTS ys structure — the alert guard handles this gracefully).

    Parameters
    ----------
    max_num_doublings
        The recipe's ``max_num_doublings`` value (typically 10).  An alert
        fires when treedepth reaches this value.

    Returns
    -------
    select_ys
        Device-side callable for ``tap.record(select_ys=...)``.
    alert_ys
        Host-side callable for ``tap.record(alert_ys=...)``.
    alert_ys_once
        Whether to fire the alert only once (False = fire every saturated step).
    """
    import jax.numpy as jnp

    def select_ys(ys_leaves: tuple) -> Any:
        for leaf in ys_leaves:
            if (
                hasattr(leaf, "dtype")
                and jnp.issubdtype(leaf.dtype, jnp.integer)
                and leaf.ndim <= 1  # scalar () or vmapped (num_chains,)
            ):
                return leaf  # num_trajectory_expansions
        return jnp.int32(-1)  # sentinel: no treedepth in this ys

    def alert_ys(event: Any) -> Any:
        import numpy as np

        try:
            # max() across chains handles both scalar and (num_chains,) shapes.
            val = int(np.asarray(event.value).max())
            if val < 0:
                return False  # sentinel from non-NUTS scan
            if val >= max_num_doublings:
                return (
                    f"output: treedepth saturated at step {event.step} "
                    f"(depth={val}/{max_num_doublings})"
                )
        except Exception:  # noqa: BLE001
            pass
        return False

    return select_ys, alert_ys, False  # alert_ys_once=False: fire every saturation


def _make_mclmc_ys_wiring() -> tuple[Any, Any, bool]:
    """Build (select_ys, alert_ys, alert_ys_once) for MCLMC warmup divergence.

    The MCLMC adaptation scan (``mclmc_find_L_and_step_size``) returns
    ``jnp.logical_not(success)`` as its per-step ys (blackjax#975 seam,
    ``mclmc_adaptation.py:301``).  A True value means the step diverged.

    ``select_ys`` finds the first bool leaf with ndim ≤ 1 in the flat ys tuple.
    The ndim ≤ 1 check is load-bearing: the ESS autocorrelation sub-scan emits
    a ``(1, DIM)`` bool ys (ndim=2) that would otherwise produce false positives;
    the guard rejects it while accepting the adaptation-flag scalar (ndim=0) and
    any vmapped-chain vector (ndim=1).  Verified via ``/tmp/tap-dce-repro/repro2.py``
    (2026-07-10).

    Returns a sentinel ``jnp.bool_(False)`` when no bool scalar/vector is found
    (non-MCLMC ys structure — the alert predicate treats False as no-divergence).

    Returns
    -------
    select_ys
        Device-side callable for ``tap.record(select_ys=...)``.
    alert_ys
        Host-side callable for ``tap.record(alert_ys=...)``.
    alert_ys_once
        Whether to fire the alert only once (False = fire every diverging step).
    """
    import jax.numpy as jnp

    def select_ys(ys_leaves: tuple) -> Any:
        for leaf in ys_leaves:
            if (
                hasattr(leaf, "dtype")
                and jnp.issubdtype(leaf.dtype, jnp.bool_)
                and leaf.ndim <= 1  # scalar () or vmapped (num_chains,)
            ):
                return leaf  # logical_not(success) divergence flag
        return jnp.bool_(False)  # sentinel: no bool divergence flag in this ys

    def alert_ys(event: Any) -> Any:
        import numpy as np

        try:
            # any() across chains handles both scalar and (num_chains,) shapes.
            val = bool(np.asarray(event.value).any())
            if val:
                return (
                    f"output: MCLMC warmup divergence at step {event.step} "
                    f"(flag={event.value})"
                )
        except Exception:  # noqa: BLE001
            pass
        return False

    return select_ys, alert_ys, False  # alert_ys_once=False: fire every diverging step


def compute_saturation_fraction(
    jsonl_path: Path | str,
    max_num_doublings: int | None = None,
) -> tuple[int, int, float]:
    """Compute the treedepth saturation fraction from a run's JSONL artifact.

    The nightly/cert consumer decides the acceptable threshold; this helper
    is POLICY-FREE (reads events and counts, no assertion).

    For NUTS recipes: output events hold ``num_trajectory_expansions`` values.
    Saturation = value >= ``max_num_doublings``.

    For MCLMC recipes: output events hold a bool divergence flag.
    Pass ``max_num_doublings=None`` to count True-flag events.

    Parameters
    ----------
    jsonl_path
        Path to the JSONL artifact written by ``tap_diagnostics_context``.
    max_num_doublings
        NUTS saturation threshold.  When ``None``, counts output events where
        the value is truthy (suitable for MCLMC divergence flags).

    Returns
    -------
    saturation_count
        Number of output events that indicate saturation/divergence.
    total_output_events
        Total number of output events in the artifact.
    fraction
        ``saturation_count / total_output_events``, or 0.0 if no output events.

    Examples
    --------
    ::

        sat_n, total, frac = compute_saturation_fraction(
            "/tmp/run.jsonl", max_num_doublings=10
        )
        if frac > 0.01:  # policy: flag if >1% saturated
            print(f"Treedepth saturation: {frac:.1%} ({sat_n}/{total})")
    """
    from jaxtap import read_jsonl

    events = read_jsonl(Path(jsonl_path))
    output_events = [e for e in events if getattr(e, "kind", "carry") == "output"]
    if not output_events:
        return 0, 0, 0.0

    if max_num_doublings is not None:
        saturated = [
            e for e in output_events if _safe_int(e.value) >= max_num_doublings
        ]
    else:
        saturated = [e for e in output_events if _safe_bool(e.value)]

    sat_n = len(saturated)
    total = len(output_events)
    return sat_n, total, sat_n / total if total > 0 else 0.0


def _safe_int(val: Any) -> int:
    """Convert a TapEvent value to int, returning -1 on failure.

    For vmapped-chain NUTS the value is ``(num_chains,)`` shaped; ``max()``
    gives the worst-case treedepth across all chains for saturation counting.
    """
    import numpy as np

    try:
        return int(np.asarray(val).max())
    except Exception:  # noqa: BLE001
        return -1


def _safe_bool(val: Any) -> bool:
    """Convert a TapEvent value to bool, returning False on failure."""
    try:
        return bool(val)
    except Exception:  # noqa: BLE001
        return False


class _TapSession:
    """Collects tap events and alerts for one instrumented run.

    Acts as the ``on_step`` callback for ``jaxtap.record``.  Streams each
    event to a JSONL file and checks cholesky primitive-tap events for
    non-finite values (to complement the stderr line that ``watch_nan``
    already emits).

    Also used as ``on_ys`` callback: y-tap events (treedepth, divergence) are
    written to the same JSONL file.

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
                "[tuningfork tap] %d alert(s) during run (types: %s). Artifact: %s",
                n,
                ", ".join(types),
                self.artifact_path,
            )


@contextlib.contextmanager
def tap_diagnostics_context(
    artifact_path: Path | None = None,
    run_tag: str = "run",
    sample_every: int = _DEFAULT_SAMPLE_EVERY,
    base_method_name: str | None = None,
    max_num_doublings: int | None = None,
):
    """Context manager: wrap a block with jaxtap carry + y-tap diagnostics.

    Monkeypatches ``jax.lax.scan`` and ``jax.lax.while_loop`` for the duration
    of the block to stream telemetry.  On exit the patch is removed and the
    JSONL artifact is closed.

    Must be called only when ``is_tap_enabled()`` is True.  Callers are
    responsible for this guard; wrapping in the disabled case is a no-op but
    incurs an unnecessary jaxtap import.

    Alert classes wired (all write to the same JSONL artifact):

    - **Float32 Cholesky NaN** (primitive tap, all recipes): fires the first
      time a Cholesky factor is non-finite inside any scan/while body.
    - **NUTS treedepth saturation** (y-tap, ``nuts`` only): fires when
      ``num_trajectory_expansions`` reaches ``max_num_doublings``.  Armed only
      when ``base_method_name="nuts"`` AND ``max_num_doublings is not None``.
      Offline tripwire — use ``compute_saturation_fraction`` to read results;
      no threshold policy here.
    - **MCLMC warmup divergence** (y-tap, ``mclmc`` only, blackjax ≥ 1.6):
      fires when ``jnp.logical_not(success)`` is True in the adaptation scan.
      Armed automatically when ``base_method_name="mclmc"``; no extra param
      needed.  See module docstring §3 for scope and the ndim ≤ 1 guard note.
      ``adjusted_mclmc`` and ``adjusted_mclmc_dynamic`` receive only the
      cholesky NaN carry tap (different scan ys structure).

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
    base_method_name
        The recipe's base method name (e.g. ``"nuts"``, ``"mclmc"``).  When
        in ``_NUTS_FAMILY`` **and** ``max_num_doublings is not None``, arms the
        treedepth y-tap.  When in ``_MCLMC_FAMILY``, arms the warmup-divergence
        y-tap (no extra param needed).  When ``None`` or in neither family,
        only the cholesky NaN carry tap is active.
    max_num_doublings
        NUTS treedepth cap.  Alerts fire when
        ``num_trajectory_expansions >= max_num_doublings``.  **Default ``None``
        = treedepth y-tap disabled** (safe for non-NUTS methods or callers that
        do not know the recipe's cap).  Pass explicitly from the recipe's
        ``base_method_params["max_num_doublings"]`` to arm the tripwire.
        Ignored for non-NUTS recipes.

    Yields
    ------
    session : _TapSession
        The session object; has ``.alerts`` (list of alert dicts) and
        ``.artifact_path`` (Path).

    Examples
    --------
    Direct usage (testing / post-mortems)::

        with tap_diagnostics_context(
            artifact_path=tmp_path / "run.jsonl",
            base_method_name="nuts",
            max_num_doublings=10,
        ) as session:
            run_inference(...)

        sat_n, total, frac = compute_saturation_fraction(
            session.artifact_path, max_num_doublings=10
        )

    The JSONL artifact contains carry events (``kind="carry"`` / ``kind`` key
    absent for pre-0.3.0 compat) and y-tap output events (``kind="output"``).
    """
    import jaxtap as tap  # lazy — imported only when tap is enabled

    if artifact_path is None:
        artifact_path = _default_artifact_path(run_tag)

    session = _TapSession(artifact_path)

    # Build y-tap wiring based on recipe family.  Never-crash: unknown methods
    # get no y-tap (only the cholesky NaN carry tap is always active).
    _select_ys: Any = None
    _alert_ys: Any = None
    _alert_ys_once: bool = False

    if base_method_name is not None:
        if base_method_name in _NUTS_FAMILY and max_num_doublings is not None:
            _select_ys, _alert_ys, _alert_ys_once = _make_nuts_ys_wiring(
                max_num_doublings
            )
        elif base_method_name in _MCLMC_FAMILY:
            _select_ys, _alert_ys, _alert_ys_once = _make_mclmc_ys_wiring()

    def _on_ys(event: Any) -> None:
        """Receive y-tap output events: write to the same JSONL."""
        session._writer(event)

    try:
        record_kwargs: dict[str, Any] = {
            "taps": [tap.watch_nan("cholesky", once=True)],
            "on_step": session,
            "sample_every": sample_every,
        }
        if _select_ys is not None:
            record_kwargs["select_ys"] = _select_ys
            record_kwargs["on_ys"] = _on_ys
            record_kwargs["alert_ys"] = _alert_ys
            record_kwargs["alert_ys_once"] = _alert_ys_once

        with tap.record(**record_kwargs):
            yield session
    finally:
        session.close()
