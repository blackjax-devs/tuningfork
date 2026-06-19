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
"""Two-part statistical-equivalence parity gate for vmap-over-chains multichain sampling.

Replaces the brittle bit-exact full-trajectory parity check.  On chaotic / curved
targets (banana) the XLA float-ordering for a width-N vectorized vmap batch differs
from scalar sequential execution by ~machine-eps per step, and a positive-Lyapunov
sampler amplifies that difference exponentially — yielding O(1) position disagreement
by sample ~100 for ANY correct chaotic sampler.  The full-trajectory test produced a
false alarm; there is NO loop fallback — vmap is the canonical multichain path.

This gate splits the question into two:

  (A) STRUCTURAL micro-parity at sample 1 — "is there a bug?"
      Identical RNG + identical kernel ⇒ the FIRST step must agree to ~machine-eps
      (< 1e-10 in float64).  A genuine key-broadcast / shape / code bug shows O(1)
      at sample 1.  This preserves bug-catching power without the brittle full-traj bar.

  (B) STATISTICAL equivalence over a probe run of n_ref samples — "sound despite chaos?"
      KS per marginal dimension (p > 0.05 primary, D < 0.05 coarse backstop),
      |Δmean_acc| < K_SE · SE  (SE = √(acc·(1−acc)/n_ref), K_SE=4 default),
      num_integration_steps exact (Δ == 0),
      divergence count exact (Δ == 0).

Decision rule (NO loop fallback — a fallback would mask real bugs)
------------------------------------------------------------------
  (A) struct PASS + (B) stat PASS  → ``"VMAP_OK"``   (vmap is canonical)
  (A) struct FAIL                  → ``"BLOCK_BUG"`` (key / codegen bug)
  (A) struct PASS + (B) stat FAIL  → ``"BLOCK_BUG"`` (distributional drift is a real defect)

References
----------
- Spec: experiments/mclmc_scaling/vmap_parity_gate_PATCH_SPEC.md
- Diagnosis: experiments/mclmc_scaling/diag_vmap_parity.py
- Injected-bug validation: experiments/mclmc_scaling/diag_injected_bug.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from scipy.stats import ks_2samp

__all__ = [
    "ParityGateResult",
    "multichain_parity_gate",
    "VMAP_OK",
    "BLOCK_BUG",
]

# Verdict constants — import these rather than hardcoding strings in callers.
VMAP_OK = "VMAP_OK"
BLOCK_BUG = "BLOCK_BUG"


@dataclass
class ParityGateResult:
    """Result of the two-part multichain parity gate.

    Attributes
    ----------
    verdict : str
        One of ``"VMAP_OK"`` or ``"BLOCK_BUG"``.
    struct_ok : bool
        True if (A) structural micro-parity passed (max|loop−vmap| at sample 1 < struct_tol).
    struct_sample1_abs : float
        Observed max-abs position difference between loop and vmap at sample index 0 (chain 0).
    stat_ok : bool | None
        True if (B) statistical equivalence passed.  ``None`` when (A) failed and (B) was not run.
    ks_D : float | None
        Max KS statistic across all marginal dimensions.  ``None`` when stat_ok is None.
    ks_p : float | None
        Min KS p-value across all marginal dimensions.  ``None`` when stat_ok is None.
    acc_abs : float | None
        |Δ mean acceptance rate| between loop and vmap ensembles.
    acc_threshold : float | None
        The ``K_SE · SE`` acceptance bar used for this run (for diagnostic output).
    nsteps_abs : float | None
        |Δ mean trajectory length| between loop and vmap ensembles (must be exactly 0).
    div_abs : float | None
        |Δ mean divergence rate| between loop and vmap ensembles.
    full_traj_max_abs : float | None
        Full-trajectory max|loop−vmap| (informational only; NOT gated — see spec).
    """

    verdict: str
    struct_ok: bool
    struct_sample1_abs: float
    stat_ok: bool | None = None
    ks_D: float | None = None
    ks_p: float | None = None
    acc_abs: float | None = None
    acc_threshold: float | None = None
    nsteps_abs: float | None = None
    div_abs: float | None = None
    full_traj_max_abs: float | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict for storage in recipe metadata."""
        return {
            "verdict": self.verdict,
            "struct_ok": self.struct_ok,
            "struct_sample1_abs": self.struct_sample1_abs,
            "stat_ok": self.stat_ok,
            "ks_D": self.ks_D,
            "ks_p": self.ks_p,
            "acc_abs": self.acc_abs,
            "acc_threshold": self.acc_threshold,
            "nsteps_abs": self.nsteps_abs,
            "div_abs": self.div_abs,
            "full_traj_max_abs": self.full_traj_max_abs,
        }


def multichain_parity_gate(
    sample_loop: Callable,
    sample_vmap: Callable,
    keys_loop: list,
    keys_stacked: Any,
    *,
    probe_n: int = 2000,
    ks_p: float = 0.05,
    ks_D: float = 0.05,
    K_SE: float = 4.0,
    struct_tol: float = 1e-10,
) -> ParityGateResult:
    """Run the two-part multichain parity gate.

    Compares loop-per-chain vs vmap-over-chains sampling to detect implementation
    bugs (part A) and verify statistical equivalence despite fp chaos (part B).

    There is NO loop fallback: vmap is the canonical multichain path.  A B-fail is
    a real distributional defect that must be investigated, not silently rerouted.

    Parameters
    ----------
    sample_loop
        Callable ``(n: int) -> (positions, div, acc, nsteps)``.
        ``positions`` shape: ``(num_chains, n, d)``.
        ``div``, ``acc``, ``nsteps``: scalars (mean over chains/steps).
    sample_vmap
        Callable with the same signature as ``sample_loop``.
    keys_loop
        Python list of per-chain JAX random keys used by the loop path.
    keys_stacked
        JAX array of stacked keys (shape ``(num_chains, 2)``); the vmap path vmaps over this.
    probe_n
        Number of samples for the statistical-equivalence probe (part B).
        Spec recommends n_ref ≥ 2000 (calibrated on banana acc~0.95, d=2).
        Default 2000.
    ks_p
        KS p-value threshold for part (B).  Default 0.05.
    ks_D
        KS statistic backstop for part (B).  Default 0.05 (coarse; catches low-n cases).
    K_SE
        Multiplier for the acceptance-rate statistical bar: ``|Δacc| < K_SE · SE``
        where ``SE = sqrt(acc·(1−acc)/probe_n)``.  Default 4.0.
        **Do NOT use a hard-coded constant** — the SE form auto-scales with model acc
        and probe_n, so a real bug (O(0.1) acc shift) always trips while MC noise
        (O(SE)) doesn't.  A fixed 5e-3 bar sits below the estimator's own noise
        floor at acc~0.95 and n_ref=2000 (SE≈4.9e-3) — that was the calibration
        error that aborted the first banana re-cert.
    struct_tol
        Max |loop−vmap| at sample index 0, chain 0, for part (A).  Default 1e-10.

    Returns
    -------
    ParityGateResult
        Verdict and all supporting metrics.

    Notes
    -----
    - **RNG key-identity assertion**: before sampling, the gate asserts that each
      ``keys_loop[i]`` produces identical ``key_data`` when split 50 ways compared
      to ``keys_stacked[i]``.  This rules out a key-broadcast bug as the cause of
      any observed discrepancy.
    - **Inject bugs at the key level**, not positions.  A position-level injection
      changes the structural comparison's denominator and may not reproduce the
      O(1) signal at sample 1 that a key-handling bug would cause.
    - **Do NOT gate on full-trajectory max|loop−vmap|**.  Under positive Lyapunov
      exponents this saturates at O(1) by ~sample 100 even for a bug-free correct
      implementation on chaotic targets.  The value is stored as informational only.
    - **nsteps and div are exact** (integer-valued, RNG-driven): require Δ == 0,
      not just |Δ| < ε.
    """
    num_chains = len(keys_loop)

    # ── Part (A): RNG key-identity assertion ──────────────────────────────────
    # Confirm that the stacked vmap keys produce identical splits to the loop keys.
    # A key-broadcast difference (e.g. using a single key for all chains) would
    # show O(1) at sample 1 AND here.
    for ci in range(num_chains):
        loop_split = jax.random.split(keys_loop[ci], 50)
        vmap_split = jax.random.split(keys_stacked[ci], 50)
        loop_kd = np.asarray(jax.random.key_data(loop_split))
        vmap_kd = np.asarray(jax.random.key_data(vmap_split))
        if not np.array_equal(loop_kd, vmap_kd):
            raise AssertionError(
                f"RNG key-identity assertion failed for chain {ci}: "
                f"keys_loop[{ci}] and keys_stacked[{ci}] produce different splits. "
                "This indicates a key-broadcast or stacking bug in the caller."
            )

    # ── Run probe ─────────────────────────────────────────────────────────────
    pos_loop, div_loop, acc_loop, ns_loop = sample_loop(probe_n)
    pos_vmap, div_vmap, acc_vmap, ns_vmap = sample_vmap(probe_n)

    pos_loop = np.asarray(pos_loop)  # (num_chains, probe_n, d)
    pos_vmap = np.asarray(pos_vmap)

    # ── Part (A): structural micro-parity at sample index 0, chain 0 ─────────
    struct_sample1_abs = float(np.max(np.abs(pos_loop[0, 0] - pos_vmap[0, 0])))
    struct_ok = struct_sample1_abs < struct_tol

    # Full-trajectory (informational only — NOT gated; expected O(1) on chaotic targets)
    full_traj_max_abs = float(np.max(np.abs(pos_loop - pos_vmap)))

    if not struct_ok:
        return ParityGateResult(
            verdict=BLOCK_BUG,
            struct_ok=False,
            struct_sample1_abs=struct_sample1_abs,
            stat_ok=None,
            full_traj_max_abs=full_traj_max_abs,
        )

    # ── Part (B): statistical equivalence ────────────────────────────────────
    d = pos_loop.shape[2]
    pool_loop = pos_loop.reshape(-1, d)  # (num_chains * probe_n, d)
    pool_vmap = pos_vmap.reshape(-1, d)

    ks_results = [ks_2samp(pool_loop[:, j], pool_vmap[:, j]) for j in range(d)]
    ks_D_obs = max(float(r.statistic) for r in ks_results)
    ks_p_obs = min(float(r.pvalue) for r in ks_results)

    # KS passes if p > threshold OR D is small (coarse backstop for small n)
    ks_ok = all(r.pvalue > ks_p or r.statistic < ks_D for r in ks_results)

    # Acceptance: STATISTICAL bar — K_SE · SE, NOT a hard-coded constant.
    # SE scales with acc and probe_n so the bar is model-agnostic.  A fixed
    # 5e-3 sits below the noise floor at acc~0.95, n=2000 (SE≈4.9e-3).
    acc_loop_f = float(acc_loop)
    acc_vmap_f = float(acc_vmap)
    acc_abs = abs(acc_loop_f - acc_vmap_f)
    acc_hat = 0.5 * (acc_loop_f + acc_vmap_f)
    acc_se = float(np.sqrt(max(acc_hat * (1.0 - acc_hat) / probe_n, 0.0)))
    acc_threshold = K_SE * acc_se
    acc_ok = acc_abs < acc_threshold

    # nsteps and div are integer-valued, RNG-driven — require exact equality.
    nsteps_abs = abs(float(ns_loop) - float(ns_vmap))
    div_abs_val = abs(float(div_loop) - float(div_vmap))
    nsteps_ok = nsteps_abs == 0.0
    div_ok = div_abs_val == 0.0

    stat_ok = ks_ok and acc_ok and nsteps_ok and div_ok

    # NO loop fallback — B-fail is a real defect, not a graceful degradation.
    verdict = VMAP_OK if stat_ok else BLOCK_BUG

    return ParityGateResult(
        verdict=verdict,
        struct_ok=True,
        struct_sample1_abs=struct_sample1_abs,
        stat_ok=stat_ok,
        ks_D=ks_D_obs,
        ks_p=ks_p_obs,
        acc_abs=acc_abs,
        acc_threshold=acc_threshold,
        nsteps_abs=nsteps_abs,
        div_abs=div_abs_val,
        full_traj_max_abs=full_traj_max_abs,
    )
