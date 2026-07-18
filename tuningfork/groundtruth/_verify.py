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
"""Verification: gate + coherence check of generated GT vs committed GT.

The ``--verify`` flag in the CLI calls ``verify_groundtruth`` after generation
to confirm that the newly generated draws pass the quality gate and are
statistically coherent with the committed GT.

Two checks are performed:

1. **Gate check** — same thresholds as the committed summary:
   ``max_rhat ≤ 1.01``, ``min_bulk_ess ≥ 400``, ``divergence_rate ≤ 0.001``
   (NUTS models), ``min_e_bfmi ≥ 0.3`` (NUTS models, when available).

2. **Coherence check** — per-site z-score of the new mean vs the committed
   mean, with a dimension-aware Bonferroni-adjusted t-threshold and a
   materiality co-primary gate.  The overall verdict is PASS, REVIEW
   (statistically flagged but immaterial — printed loudly, counts as pass),
   or FAIL (statistically flagged and material — returns False).

Coherence formula
-----------------
For each site dimension ``d``::

    se_denom[d] = max(sqrt(se_new[d]**2 + se_com[d]**2), _SE_FLOOR)
    z_d         = |mean_new[d] - mean_com[d]| / se_denom[d]
    mat_d       = |mean_new[d] - mean_com[d]| / max(std_com[d], _SE_FLOOR)

where ``se = between_chain_se`` from the respective ``summary_v2.json``.
``_SE_FLOOR = 1e-8`` prevents division-by-zero for scalar / near-zero-SE sites.

The threshold is dimension-aware (Bonferroni over ``D_total`` scalar dims
across all matching sites)::

    nu     = 2 * (n_chains - 1)      # degrees of freedom
    z_crit = t.ppf(1 - alpha/(2*D_total), nu)

A dimension is a **hard FAIL** iff ``z_d > z_crit`` AND ``mat_d > _TAU_SCI``.
A dimension is **REVIEW** iff ``z_d > z_crit`` but ``mat_d <= _TAU_SCI``
(statistically flagged but immaterial — printed loudly but gate passes).
The gate FAILS if any dimension hard-fails, if a site present in the
committed summary is absent from the generated summary, or if a matched site
has fewer generated dims than committed dims (shape shrink).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary

__all__ = ["verify_groundtruth"]

# Gate thresholds (same as _emit.py, calibrated for 10×10k full runs)
_MAX_RHAT: float = 1.01
_MIN_BULK_ESS: float = 400.0
_MAX_DIV_RATE: float = 0.001
_MIN_EBFMI: float = 0.3

# Coherence SE floor (prevents division-by-zero on scalar / near-zero-SE sites).
# Under the dual gate this absolute floor only affects REVIEW labelling, not
# pass/fail: when a dim trips z > z_crit its verdict is decided by materiality
# (mat_d vs _TAU_SCI), so the floored z only influences whether the dim is REVIEW
# or unchecked — not whether the gate hard-fails.
_SE_FLOOR: float = 1e-8

# Materiality threshold: |Δμ| / std_committed must strictly exceed this to
# be a hard FAIL (strict > so that the boundary 0.05σ is REVIEW, not FAIL).
# Mirrors the W1 gate sibling in calibration/_gate/w1_realm.py.
_TAU_SCI: float = 0.05

# Default n_chains when the summary metadata field is absent
_DEFAULT_N_CHAINS: int = 10


def _check_gate(generated_summary: dict) -> tuple[bool, dict[str, Any]]:
    """Run the quality gate; returns ``(passed, details_dict)``."""
    gate = generated_summary.get("quality_gate", {})
    generator = generated_summary.get("generator", "")
    is_nuts = generator not in ("analytic_iid",)

    max_rhat = gate.get("max_rhat", float("inf"))
    min_bulk = gate.get("min_bulk_ess", 0.0)
    total_div = gate.get("total_divergences", 0)
    n_total = generated_summary.get("n_total", 1)
    div_rate = total_div / n_total if n_total > 0 else 0.0
    min_ebfmi = gate.get("min_e_bfmi")

    rhat_ok = max_rhat <= _MAX_RHAT
    ess_ok = (not is_nuts) or (min_bulk >= _MIN_BULK_ESS)
    div_ok = (not is_nuts) or (div_rate <= _MAX_DIV_RATE)
    ebfmi_ok = (not is_nuts) or (min_ebfmi is None) or (min_ebfmi >= _MIN_EBFMI)

    passed = rhat_ok and ess_ok and div_ok and ebfmi_ok
    details = {
        "max_rhat": max_rhat,
        "rhat_ok": rhat_ok,
        "min_bulk_ess": min_bulk,
        "ess_ok": ess_ok,
        "divergence_rate": div_rate,
        "div_ok": div_ok,
        "min_e_bfmi": min_ebfmi,
        "ebfmi_ok": ebfmi_ok,
        "passed": passed,
    }
    return passed, details


def _check_coherence(
    generated_summary: dict,
    committed_summary: dict,
    alpha: float = 0.05,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Per-site coherence check with dimension-aware threshold and materiality gate.

    Note: k̂ / heavy-tail diagnostics are intentionally out of scope here.
    Coherence is on posterior *means* (CLT regime), not on tail shape; pareto-k
    values affect the reliability of the SE estimates but are not checked by
    this function — they are part of the quality gate upstream.

    Returns ``(all_pass, per_site_results, meta)`` where:
    - ``all_pass`` is True for PASS or REVIEW, False for FAIL.
    - ``per_site_results`` is a list of per-site dicts.
    - ``meta`` contains ``D_total``, ``nu``, ``z_crit``, ``n_chains``, ``alpha``,
      ``missing_committed_sites``, and ``shrunk_sites``.
    """
    gen_per_site = generated_summary.get("per_site", {})
    com_per_site = committed_summary.get("per_site", {})

    # Extract n_chains from generated summary; fall back to default with a warning.
    n_chains = generated_summary.get("n_chains")
    if n_chains is None:
        print(
            "[coherence] WARNING: 'n_chains' not found in generated summary; "
            f"defaulting to {_DEFAULT_N_CHAINS}."
        )
        n_chains = _DEFAULT_N_CHAINS

    # First pass: compute D_total and detect shape-shrunk sites.
    # D_total counts scalar dims in non-shrunk matching sites (Bonferroni denominator).
    # A shape-shrunk site (generated has fewer dims than committed) is a hard FAIL
    # and its dims are excluded from D_total (they will not be z-tested).
    D_total = 0
    shrunk_sites: list[str] = []
    for site in gen_per_site:
        if site not in com_per_site:
            continue
        n_gen = len(np.asarray(gen_per_site[site]["mean"]).ravel())
        n_com = len(np.asarray(com_per_site[site]["mean"]).ravel())
        if n_gen < n_com:
            shrunk_sites.append(site)
        else:
            D_total += n_com  # test committed dims (n_gen >= n_com)

    if shrunk_sites:
        for s in shrunk_sites:
            n_gen = len(np.asarray(gen_per_site[s]["mean"]).ravel())
            n_com = len(np.asarray(com_per_site[s]["mean"]).ravel())
            print(
                f"[coherence] FAIL: site '{s}': generated has {n_gen} dim(s) "
                f"but committed has {n_com} — shape shrink, dropped dims unchecked."
            )

    # Dimension-aware t-threshold (Bonferroni over D_total dims).
    nu = 2 * (n_chains - 1)
    if D_total > 0:
        z_crit = float(scipy_stats.t.ppf(1.0 - alpha / (2.0 * D_total), nu))
    else:
        z_crit = float("inf")  # degenerate: no dims to check

    # Check for committed sites absent from generated — these are hard FAILs.
    missing_committed_sites = [s for s in com_per_site if s not in gen_per_site]
    if missing_committed_sites:
        for s in missing_committed_sites:
            print(
                f"[coherence] FAIL: committed site '{s}' is absent from generated summary."
            )

    results: list[dict[str, Any]] = []
    any_hard_fail = bool(missing_committed_sites) or bool(shrunk_sites)

    for site in gen_per_site:
        if site not in com_per_site:
            print(
                f"[coherence] NOTE: generated site '{site}' is not in committed summary"
                " — skipping."
            )
            continue

        # Shape-shrunk sites: already recorded as FAIL; add a result entry and skip z-test.
        if site in shrunk_sites:
            results.append(
                {
                    "site": site,
                    "max_z": float("nan"),
                    "passed": False,
                    "verdict": "FAIL",
                    "z_per_dim": [],
                    "mat_per_dim": [],
                    "worst_dim": -1,
                    "hard_fail_dims": [],
                    "review_dims": [],
                    "shape_shrink": True,
                }
            )
            continue

        gen_s = gen_per_site[site]
        com_s = com_per_site[site]

        mean_new = np.asarray(gen_s["mean"]).ravel()
        mean_com = np.asarray(com_s["mean"]).ravel()
        se_new = np.asarray(gen_s["between_chain_se"]).ravel()
        se_com = np.asarray(com_s["between_chain_se"]).ravel()
        std_com = np.asarray(com_s.get("std", np.ones(len(mean_com)))).ravel()

        # Truncate to committed dim count (n_gen >= n_com guaranteed here).
        n = len(mean_com)
        mean_new = mean_new[:n]
        se_new = se_new[:n] if len(se_new) >= n else np.full(n, _SE_FLOOR)
        se_com = se_com[:n] if len(se_com) >= n else np.full(n, _SE_FLOOR)
        std_com = std_com[:n] if len(std_com) >= n else np.ones(n)

        # Denominator: pooled SE (sqrt of sum of squares), floored.
        se_denom = np.maximum(np.sqrt(se_new**2 + se_com**2), _SE_FLOOR)
        z_vals = np.abs(mean_new - mean_com) / se_denom
        max_z = float(np.max(z_vals))

        # Materiality: |Δμ| / std_committed, guarding std_com=0.
        std_com_safe = np.where(std_com == 0.0, 1.0, std_com)
        mat_vals = np.abs(mean_new - mean_com) / std_com_safe

        # Per-dim verdicts: strict > on materiality so that mat=_TAU_SCI is REVIEW.
        over_z = z_vals > z_crit
        over_mat = mat_vals > _TAU_SCI
        hard_fail_dims = np.where(over_z & over_mat)[0]
        review_dims = np.where(over_z & ~over_mat)[0]

        site_hard_fail = len(hard_fail_dims) > 0
        site_review = len(review_dims) > 0
        site_verdict = (
            "FAIL" if site_hard_fail else ("REVIEW" if site_review else "PASS")
        )

        if site_hard_fail:
            any_hard_fail = True

        results.append(
            {
                "site": site,
                "max_z": max_z,
                "passed": not site_hard_fail,
                "verdict": site_verdict,
                "z_per_dim": z_vals.tolist(),
                "mat_per_dim": mat_vals.tolist(),
                "worst_dim": int(np.argmax(z_vals)),
                "hard_fail_dims": hard_fail_dims.tolist(),
                "review_dims": review_dims.tolist(),
            }
        )

    all_pass = not any_hard_fail
    meta: dict[str, Any] = {
        "D_total": D_total,
        "nu": nu,
        "z_crit": z_crit,
        "n_chains": n_chains,
        "alpha": alpha,
        "missing_committed_sites": missing_committed_sites,
        "shrunk_sites": shrunk_sites,
    }
    return all_pass, results, meta


def verify_groundtruth(
    model_name: str,
    generated_summary: dict,
    generated_draws_path: Path | None = None,
    *,
    alpha: float = 0.05,
    print_results: bool = True,
) -> bool:
    """Check generated GT quality and coherence vs committed catalog GT.

    Runs two checks: (1) **Gate** — max_rhat ≤ 1.01, min_bulk_ess ≥ 400,
    divergence_rate ≤ 0.001, min_e_bfmi ≥ 0.3 (NUTS models).
    (2) **Coherence** — dimension-aware t-test with materiality co-primary gate
    (alpha=``alpha`` Bonferroni-corrected over all scalar dims across matching
    sites; a dim hard-fails iff the z exceeds the adjusted critical value AND
    |Δμ|/std_committed > 0.05; statistically flagged but immaterial dims are
    REVIEW — printed loudly but counted as PASS).

    A ``UserWarning`` is always emitted (regardless of ``print_results``) when
    the coherence check finds any REVIEW dims, hard FAILs, missing committed
    sites, or shape-shrunk sites, so the signal is never fully swallowed.

    Parameters
    ----------
    model_name
        Registry model name.
    generated_summary
        Parsed ``summary_v2.json`` from the just-generated run.
    generated_draws_path
        Path to ``draws.npz``; logged for reference, not read.
    alpha
        Family-wise error rate for the Bonferroni-adjusted coherence threshold.
        Default 0.05.
    print_results
        Print gate and coherence results to stdout.

    Returns
    -------
    bool
        ``True`` if gate + coherence both pass (PASS or REVIEW).
        ``False`` if either fails (hard FAIL or missing committed site).
    """
    committed_summary = load_committed_summary(model_name)
    gt_dir = committed_gt_dir(model_name)

    gate_pass, gate_details = _check_gate(generated_summary)
    coh_pass, coh_results, coh_meta = _check_coherence(
        generated_summary, committed_summary, alpha
    )
    overall = gate_pass and coh_pass

    # Emit a programmatic warning for any coherence anomaly so the signal is
    # never fully swallowed when print_results=False (C6 non-silent-REVIEW).
    m = coh_meta
    warn_parts: list[str] = []
    if m["missing_committed_sites"]:
        warn_parts.append(f"missing committed sites: {m['missing_committed_sites']}")
    if m.get("shrunk_sites"):
        warn_parts.append(f"shape-shrunk sites: {m['shrunk_sites']}")
    for r in coh_results:
        if r["verdict"] == "FAIL":
            warn_parts.append(f"FAIL site '{r['site']}' (max_z={r['max_z']:.3f})")
        elif r["verdict"] == "REVIEW":
            warn_parts.append(
                f"REVIEW site '{r['site']}' (max_z={r['max_z']:.3f}, immaterial)"
            )
    if warn_parts:
        warnings.warn(
            f"[verify] coherence anomalies in {model_name}: " + "; ".join(warn_parts),
            UserWarning,
            stacklevel=2,
        )

    if print_results:
        print(f"[verify] model={model_name}  draws={generated_draws_path}")
        print(f"[verify] committed GT:  {gt_dir}")

        # Gate results
        g = gate_details
        rhat_sym = "PASS" if g["rhat_ok"] else "FAIL"
        ess_sym = "PASS" if g["ess_ok"] else "FAIL"
        div_sym = "PASS" if g["div_ok"] else "FAIL"
        ebfmi_sym = "PASS" if g["ebfmi_ok"] else "FAIL"
        ebfmi_val = f"{g['min_e_bfmi']:.4f}" if g["min_e_bfmi"] is not None else "n/a"
        print(f"[gate]   {rhat_sym} max_rhat={g['max_rhat']:.5f} (≤{_MAX_RHAT})")
        print(
            f"         {ess_sym} min_bulk_ess={g['min_bulk_ess']:.0f} (≥{_MIN_BULK_ESS})"
        )
        print(
            f"         {div_sym} div_rate={g['divergence_rate']:.5f} (≤{_MAX_DIV_RATE})"
        )
        print(f"         {ebfmi_sym} min_e_bfmi={ebfmi_val} (≥{_MIN_EBFMI})")
        print(f"[gate]   → {'PASS' if gate_pass else 'FAIL'}")

        # Coherence header: D_total, nu, z_crit
        print(
            f"[coherence] alpha={m['alpha']}  D_total={m['D_total']}  "
            f"nu={m['nu']}  z_crit={m['z_crit']:.3f}"
        )
        for ms in m["missing_committed_sites"]:
            print(f"  FAIL  missing committed site: {ms}")
        for ss in m.get("shrunk_sites", []):
            print(f"  FAIL  shape-shrunk site: {ss}")

        passing = [r for r in coh_results if r["verdict"] == "PASS"]
        review = [r for r in coh_results if r["verdict"] == "REVIEW"]
        failing = [r for r in coh_results if r["verdict"] == "FAIL"]
        n_checked = len(coh_results)

        print(
            f"          {len(passing)}/{n_checked} sites PASS  "
            f"{len(review)} REVIEW  {len(failing)} FAIL"
        )
        for r in coh_results:
            if r["verdict"] in ("FAIL", "REVIEW"):
                max_z_str = f"{r['max_z']:.3f}" if np.isfinite(r["max_z"]) else "n/a"
                worst = r["worst_dim"] if r["worst_dim"] >= 0 else "n/a"
                print(
                    f"  {r['verdict']}  {r['site']}: max_z={max_z_str} "
                    f"(worst dim {worst})"
                )
                for d in r["hard_fail_dims"]:
                    zd = r["z_per_dim"][d]
                    md = r["mat_per_dim"][d]
                    print(
                        f"    dim {d}: z={zd:.3f} > z_crit={m['z_crit']:.3f}  "
                        f"mat={md:.4f} > {_TAU_SCI}  → HARD FAIL"
                    )
                for d in r["review_dims"]:
                    zd = r["z_per_dim"][d]
                    md = r["mat_per_dim"][d]
                    print(
                        f"    dim {d}: z={zd:.3f} > z_crit={m['z_crit']:.3f}  "
                        f"mat={md:.4f} <= {_TAU_SCI}  → REVIEW (immaterial)"
                    )
        if (
            not failing
            and not review
            and not m["missing_committed_sites"]
            and not m.get("shrunk_sites")
        ):
            max_z_overall = max(
                (r["max_z"] for r in coh_results if np.isfinite(r["max_z"])),
                default=0.0,
            )
            print(f"  all sites PASS  (max_z={max_z_overall:.3f})")

        coh_verdict = "PASS" if coh_pass else "FAIL"
        if coh_pass and review:
            coh_verdict = "REVIEW"
        print(f"[coherence] → {coh_verdict}")

        overall_sym = "PASS" if overall else "FAIL"
        print(f"[verify]  → {overall_sym}")

    return overall
