#!/usr/bin/env python3
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
"""Cross-tabulate re-emit gate failures against the baseline dependency stack.

    Does a failure family concentrate on particular originating versions?


Concentration on one version points at a code-path change between releases.
An even spread points at something intrinsic to the sampler family instead.
The 2x2 that separates them is version x family, with the version held fixed:
a family that fails only under one baseline version, while its siblings from
that same version pass, isolates the interaction from both main effects.

Reads gate outcomes from the sweep logs and baseline versions from a pinned
revision, so it never depends on the working tree.
"""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/jlao/topics/project/blackjax-devs/tuningfork/tf-ess-switch")
SP = Path(
    "/tmp/claude-1000/-home-jlao-project-blackjax-devs/"
    "bcf23c03-03c5-43a8-a367-a205610d679d/scratchpad"
)

log = "".join(
    (SP / n).read_text()
    for n in ("sweep.log", "sweep2.log", "sweep3.log")
    if (SP / n).exists()
)
last = {}
for cell, verdict in re.findall(r"CELL_DONE cell=(\S+) verdict=(\S+)", log):
    last[cell] = verdict


def family(name: str) -> str:
    for f in ("dense_imm", "low_rank_imm", "diag_imm", "chees"):
        if f in name:
            return f
    return "other"


def baseline_stack(cell: str) -> tuple[str, str]:
    model, name = cell.split("/", 1)
    rel = f"tuningfork/catalog/{model}/recipes/{name}"
    p = subprocess.run(
        ["git", "show", f"b09c247:{rel}"], cwd=ROOT, capture_output=True, text=True
    )
    if p.returncode != 0:
        return ("?", "?")
    d = json.loads(p.stdout)
    return (d.get("blackjax_version") or "?", d.get("jax_version") or "?")


rows = []
for cell, verdict in last.items():
    bj, jx = baseline_stack(cell)
    rows.append((cell, verdict, family(cell), bj, jx))

dense = [r for r in rows if r[2] == "dense_imm"]
print(
    f"dense-IMM cells swept: {len(dense)}  failures: {sum(1 for r in dense if r[1] == 'FAIL')}"
)

print("\n--- dense-IMM outcome by originating jax version ---")
by_jax: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
for _, v, _, _, jx in dense:
    by_jax[jx][0] += 1
    by_jax[jx][1] += v == "FAIL"
for jx, (n, f) in sorted(by_jax.items()):
    print(f"  jax {jx:<8} {f}/{n} failed = {100 * f / n:.0f}%")

print("\n--- dense-IMM outcome by originating blackjax version ---")
by_bj: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
for _, v, _, bj, _ in dense:
    by_bj[bj][0] += 1
    by_bj[bj][1] += v == "FAIL"
for bj, (n, f) in sorted(by_bj.items(), key=lambda kv: -kv[1][0]):
    print(f"  {bj:<28} {f}/{n} failed = {100 * f / n:.0f}%")

print("\n--- CONTROL: same cross-tab for NON-dense cells ---")
nond = [r for r in rows if r[2] != "dense_imm"]
by_jax2: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
for _, v, _, _, jx in nond:
    by_jax2[jx][0] += 1
    by_jax2[jx][1] += v == "FAIL"
for jx, (n, f) in sorted(by_jax2.items()):
    print(f"  jax {jx:<8} {f}/{n} failed = {100 * f / n:.0f}%")

print("\n--- dense-IMM failures, listed with their originating stack ---")
for cell, v, _, bj, jx in sorted(dense):
    if v == "FAIL":
        print(f"  {cell:<66} bj={bj} jax={jx}")

print("\n--- excluding lotka_volterra (the model-wide collapse) ---")
d2 = [r for r in dense if not r[0].startswith("lotka_volterra")]
print(
    f"  dense-IMM outside lotka_volterra: {sum(1 for r in d2 if r[1] == 'FAIL')}/{len(d2)} failed"
)
n2 = [r for r in nond if not r[0].startswith("lotka_volterra")]
print(
    f"  non-dense outside lotka_volterra: {sum(1 for r in n2 if r[1] == 'FAIL')}/{len(n2)} failed"
)

print("\n=== INTERACTION: is it dev84, dense, or dev84 x dense? ===")
tab: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
for _, v, fam, bj, _ in rows:
    key = (bj, "dense" if fam == "dense_imm" else "non-dense")
    tab[key][0] += 1
    tab[key][1] += v == "FAIL"
for bj in sorted({r[3] for r in rows}):
    parts = []
    for kind in ("dense", "non-dense"):
        n, f = tab[(bj, kind)]
        parts.append(f"{kind} {f}/{n}" + (f" ({100 * f / n:.0f}%)" if n else ""))
    print(f"  {bj:<28} " + "   ".join(parts))

print("\n=== dev84 x dense, EXCLUDING lotka_volterra ===")
d84 = [
    r for r in rows if r[3] == "1.6.dev84+g0ef2f578a" and not r[0].startswith("lotka")
]
for kind in ("dense", "non-dense"):
    sub = [r for r in d84 if (r[2] == "dense_imm") == (kind == "dense")]
    if sub:
        print(f"  {kind}: {sum(1 for r in sub if r[1] == 'FAIL')}/{len(sub)} failed")

print("\n=== the ill_cond_50 analytic-truth cell: which cohort? ===")
for cell, v, fam, bj, jx in rows:
    if cell == "ill_cond_50/low__dynamic_hmc__window_adaptation_dense_imm.json":
        print(f"  {cell}\n    verdict={v} family={fam} baseline bj={bj} jax={jx}")
