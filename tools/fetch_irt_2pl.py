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
"""Fetch and persist the irt_2pl dataset from posteriordb GitHub.

Provenance:
    Source: posteriordb GitHub repository (stan-dev/posteriordb)
    URL: https://raw.githubusercontent.com/stan-dev/posteriordb/master/
         posterior_database/data/data/irt_2pl.json.zip
    Dataset: irt_2pl (J=100 students, I=20 items, 2000 binary responses).
    Fields: y[j, i] = 1 if student j answered item i correctly, 0 otherwise.

    Note: posteriordb IRT 2PL has NO reference posterior draws
    (reference_posterior_name: null in its metadata). There is no Stan-based
    cross-check available; Tier-A uses Long-NUTS self-check only.

Output CSV:
    bjx_bench/data/irt_2pl.csv — 2000-row long-format table with columns:
        student_id (0-indexed, 0..99), item_id (0-indexed, 0..19), response (0/1).

    The loader in bjx_bench/model/hierarchical/irt_2pl.py reshapes the CSV
    into a (J=100, I=20) 2-D JAX array RESPONSE[j, i].

Usage:
    cd bjx-bench
    uv run python tools/fetch_irt_2pl.py
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

_URL = (
    "https://raw.githubusercontent.com/stan-dev/posteriordb/master/"
    "posterior_database/data/data/irt_2pl.json.zip"
)

_OUT = Path(__file__).parent.parent / "bjx_bench" / "data" / "irt_2pl.csv"


def main() -> None:
    print(f"Fetching {_URL} ...")
    req = urllib.request.Request(_URL, headers={"User-Agent": "bjx-bench/1.0"})
    zipped = urllib.request.urlopen(req).read()

    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        with zf.open("irt_2pl.json") as f:
            data = json.loads(f.read())

    n_students = data["J"]  # number of students (J in Stan notation)
    n_items = data["I"]  # number of items (I in Stan notation)
    # posteriordb stores y as a list of n_items rows each of length n_students (shape I×J).
    # We transpose to (J×I) so RESPONSE[j, i] = 1 if student j answered item i correctly.
    y_raw = data["y"]  # shape (n_items, n_students) in posteriordb storage

    print(
        f"J={n_students} students, I={n_items} items, total responses={n_students * n_items}"
    )
    print(f"y_raw outer dim={len(y_raw)}, inner dim={len(y_raw[0])}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "item_id", "response"])
        for j in range(n_students):
            for item_idx in range(n_items):
                # y_raw[item_idx][j] → transpose to (j, item_idx) order
                writer.writerow([j, item_idx, int(y_raw[item_idx][j])])

    print(f"Wrote {n_students * n_items} rows to {_OUT}")
    # Verify binary
    unique_vals = sorted(
        {
            int(y_raw[item_idx][j])
            for item_idx in range(n_items)
            for j in range(n_students)
        }
    )
    print("Unique response values:", unique_vals)


if __name__ == "__main__":
    main()
