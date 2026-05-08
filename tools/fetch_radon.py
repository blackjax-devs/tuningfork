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
    cd bjx-bench
    uv run python tools/fetch_radon.py
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
    "posterior_database/data/data/radon_all.json.zip"
)

_OUT = Path(__file__).parent.parent / "bjx_bench" / "data" / "radon.csv"


def main() -> None:
    print(f"Fetching {_URL} ...")
    req = urllib.request.Request(_URL, headers={"User-Agent": "bjx-bench/1.0"})
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
