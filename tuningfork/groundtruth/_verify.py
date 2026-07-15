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
   mean, normalized by ``max(between_chain_se_new, between_chain_se_committed)``.
   All sites must have ``max_z ≤ 3.0``.  This is the same framework used
   during the multichain GT migration coherence validation.

Coherence formula
-----------------
For each site dimension ``d``::

    z_d = |mean_new[d] - mean_committed[d]|
          / max(se_new[d], se_committed[d], _SE_FLOOR)

where ``se = between_chain_se`` from the respective ``summary_v2.json``.
``_SE_FLOOR = 1e-8`` prevents division-by-zero for scalar / deterministic sites.
The per-site ``max_z = max(z_d)`` over all dimensions of that site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tuningfork.groundtruth._dispatch import committed_gt_dir, load_committed_summary

__all__ = ["verify_groundtruth"]

# Gate thresholds (same as _emit.py, calibrated for 10×10k full runs)
_MAX_RHAT: float = 1.01
_MIN_BULK_ESS: float = 400.0
_MAX_DIV_RATE: float = 0.001
_MIN_EBFMI: float = 0.3

# Coherence SE floor (prevents division-by-zero on scalar / near-zero-SE sites)
_SE_FLOOR: float = 1e-8


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
    z_threshold: float,
) -> tuple[bool, list[dict[str, Any]]]:
    """Per-site z-score coherence vs committed; returns ``(all_pass, per_site)``."""
    gen_per_site = generated_summary.get("per_site", {})
    com_per_site = committed_summary.get("per_site", {})
    results = []
    all_pass = True

    for site in gen_per_site:
        if site not in com_per_site:
            continue

        gen_s = gen_per_site[site]
        com_s = com_per_site[site]

        mean_new = np.asarray(gen_s["mean"]).ravel()
        mean_com = np.asarray(com_s["mean"]).ravel()
        se_new = np.asarray(gen_s["between_chain_se"]).ravel()
        se_com = np.asarray(com_s["between_chain_se"]).ravel()

        # Pad to common length if shapes differ
        n = min(len(mean_new), len(mean_com))
        mean_new = mean_new[:n]
        mean_com = mean_com[:n]
        se_new = se_new[:n] if len(se_new) >= n else np.full(n, _SE_FLOOR)
        se_com = se_com[:n] if len(se_com) >= n else np.full(n, _SE_FLOOR)

        se_denom = np.maximum(np.maximum(se_new, se_com), _SE_FLOOR)
        z_vals = np.abs(mean_new - mean_com) / se_denom
        max_z = float(np.max(z_vals))
        site_pass = max_z <= z_threshold

        results.append(
            {
                "site": site,
                "max_z": max_z,
                "passed": site_pass,
                "z_per_dim": z_vals.tolist(),
                "worst_dim": int(np.argmax(z_vals)),
            }
        )
        if not site_pass:
            all_pass = False

    return all_pass, results


def verify_groundtruth(
    model_name: str,
    generated_summary: dict,
    generated_draws_path: Path | None = None,
    *,
    z_threshold: float = 3.0,
    print_results: bool = True,
) -> bool:
    """Check generated GT quality and coherence vs committed catalog GT.

    Runs two checks: (1) **Gate** — max_rhat ≤ 1.01, min_bulk_ess ≥ 400,
    divergence_rate ≤ 0.001, min_e_bfmi ≥ 0.3 (NUTS models).
    (2) **Coherence** — per-site mean z-score ≤ ``z_threshold`` vs committed.

    Parameters
    ----------
    model_name
        Registry model name.
    generated_summary
        Parsed ``summary_v2.json`` from the just-generated run.
    generated_draws_path
        Path to ``draws.npz``; logged for reference, not read.
    z_threshold
        Per-site z-score threshold.  Default 3.0.
    print_results
        Print gate and coherence results to stdout.

    Returns
    -------
    bool
        ``True`` if gate + coherence both pass.
    """
    committed_summary = load_committed_summary(model_name)
    gt_dir = committed_gt_dir(model_name)

    gate_pass, gate_details = _check_gate(generated_summary)
    coh_pass, coh_results = _check_coherence(
        generated_summary, committed_summary, z_threshold
    )
    overall = gate_pass and coh_pass

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

        # Coherence results
        failing = [r for r in coh_results if not r["passed"]]
        passing = [r for r in coh_results if r["passed"]]
        print(
            f"[coherence] z_threshold={z_threshold}  "
            f"{len(passing)}/{len(coh_results)} sites pass"
        )
        for r in failing:
            print(
                f"  FAIL  {r['site']}: max_z={r['max_z']:.2f} "
                f"(worst dim {r['worst_dim']})"
            )
        if not failing:
            max_z_overall = max((r["max_z"] for r in coh_results), default=0.0)
            print(f"  all sites pass  (max_z={max_z_overall:.2f})")
        print(f"[coherence] → {'PASS' if coh_pass else 'FAIL'}")

        overall_sym = "PASS" if overall else "FAIL"
        print(f"[verify]  → {overall_sym}")

    return overall
