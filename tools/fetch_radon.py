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
"""Fetch and persist the radon_all dataset from posteriordb GitHub.

Provenance:
    Source: posteriordb GitHub repository (stan-dev/posteriordb)
    URL: https://raw.githubusercontent.com/stan-dev/posteriordb/master/
         posterior_database/data/data/radon_all.json.zip
    Dataset: radon_all (Gelman & Hill 2007, ch. 12 — multilevel radon measurements)
    N=12573 observations, J=386 US counties.
    Fields: county_idx (1-indexed in source → 0-indexed here), floor_measure,
            log_radon, log_uppm (log uranium ppm per observation).

Usage:
    cd tuningfork
    uv run python tools/fetch_radon.py
"""

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

_URL = (
    "https://raw.githubusercontent.com/stan-dev/posteriordb/master/"
    "posterior_database/data/data/radon_all.json.zip"
)

_OUT = Path(__file__).parent.parent / "tuningfork" / "data" / "radon.csv"


def main() -> None:
    print(f"Fetching {_URL} ...")
    req = urllib.request.Request(_URL, headers={"User-Agent": "tuningfork/1.0"})
    zipped = urllib.request.urlopen(req).read()

    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        with zf.open("radon_all.json") as f:
            data = json.loads(f.read())

    N = data["N"]
    J = data["J"]
    county_idx = data["county_idx"]  # 1-indexed
    floor_measure = data["floor_measure"]
    log_radon = data["log_radon"]
    log_uppm = data["log_uppm"]

    print(f"N={N}, J={J}")
    print(f"county_idx range (1-indexed): {min(county_idx)}..{max(county_idx)}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["county_idx", "floor_measure", "log_radon", "log_uppm"])
        for i in range(N):
            # Convert 1-indexed to 0-indexed
            writer.writerow(
                [county_idx[i] - 1, floor_measure[i], log_radon[i], log_uppm[i]]
            )

    print(f"Wrote {N} rows to {_OUT}")


if __name__ == "__main__":
    main()
