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
"""Fetch and persist real S&P 500 daily returns for stochastic volatility model.

Provenance:
    Source: Pyro Datasets — S&P 500 daily returns (GitHub CSV)
    URL: https://github.com/pyro-ppl/datasets/blob/master/SP500.csv?raw=true
    Dataset: S&P 500 daily returns (mean-centered, first 500 observations).
    Note: Real financial data, not synthetic. Replaces the legacy synthetic
    KSC (Kim-Shephard-Chib) data that was used in early development.
    The CSV is also exposed via numpyro.examples.datasets.SP500 for reference.

Output:
    tuningfork/data/stoch_vol_returns.csv — CSV with header "returns" and one
    float per line. Shape: (500,). This matches the format expected by
    tuningfork.model.latent_gaussian.stoch_vol.

Usage:
    cd tuningfork

    # Verify output matches existing CSV (default --check mode)
    uv run python tools/fetch_stoch_vol.py

    # Overwrite the CSV with fresh fetch (--write mode)
    uv run python tools/fetch_stoch_vol.py --write
"""

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np

_URL = "https://github.com/pyro-ppl/datasets/blob/master/SP500.csv?raw=true"
_OUT = Path(__file__).parent.parent / "tuningfork" / "data" / "stoch_vol_returns.csv"


def fetch_sp500_returns() -> np.ndarray:
    """Fetch S&P 500 daily returns from Pyro Datasets GitHub.

    The CSV has columns: [index, DATE, VALUE].
    We extract the VALUE column, take the first 500 rows, and mean-center.

    Returns
    -------
    np.ndarray of shape (500,) dtype float64 — first 500 mean-centered returns.
    """
    print(f"Fetching {_URL} ...")
    req = urllib.request.Request(_URL, headers={"User-Agent": "tuningfork/1.0"})
    response = urllib.request.urlopen(req)
    content = response.read().decode("utf-8")
    lines = content.strip().split("\n")

    # Parse CSV: skip header, extract VALUE column (index 2)
    reader = csv.reader(lines)
    header = next(reader)
    assert header == ["", "DATE", "VALUE"], f"Unexpected header: {header}"

    returns_all = []
    for row in reader:
        if row and len(row) > 2:
            returns_all.append(float(row[2]))

    returns_all = np.array(returns_all, dtype=np.float64)

    # Take first 500 and mean-center
    returns_500 = returns_all[:500]
    returns_centered = returns_500 - np.mean(returns_500)

    return returns_centered


def write_csv(returns: np.ndarray, path: Path) -> None:
    """Write returns to CSV with header 'returns'.

    Parameters
    ----------
    returns
        1-D numpy array of returns.
    path
        Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy.savetxt with header and comments=""
    # header: column names
    # comments: prefix for header (empty string = no prefix)
    # fmt: float format
    np.savetxt(path, returns, header="returns", comments="", fmt="%.6f")


def read_csv(path: Path) -> np.ndarray:
    """Read returns from CSV, skipping header.

    Parameters
    ----------
    path
        Input CSV path.

    Returns
    -------
    np.ndarray of shape (N,) dtype float64.
    """
    return np.loadtxt(path, skiprows=1, dtype=np.float64)


def check_mode(returns: np.ndarray, existing_path: Path, atol: float = 1e-6) -> bool:
    """Verify that fresh fetch matches existing CSV.

    Parameters
    ----------
    returns
        Freshly fetched returns array.
    existing_path
        Path to existing CSV.
    atol
        Absolute tolerance for comparison.

    Returns
    -------
    True if match, False otherwise.
    """
    if not existing_path.exists():
        print(f"ERROR: {existing_path} does not exist. Use --write to create it.")
        return False

    existing_returns = read_csv(existing_path)

    if existing_returns.shape != returns.shape:
        print(
            f"ERROR: shape mismatch. Existing: {existing_returns.shape}, "
            f"Fresh: {returns.shape}"
        )
        return False

    if not np.allclose(existing_returns, returns, atol=atol):
        max_diff = np.max(np.abs(existing_returns - returns))
        print(
            f"ERROR: value mismatch. Max diff: {max_diff:.10f} (atol={atol}). "
            f"Use --write to overwrite."
        )
        return False

    print(f"OK: {existing_path} matches fresh fetch.")
    print(f"  Shape: {returns.shape}")
    print(f"  Mean: {np.mean(returns):.10f}")
    print(f"  Std: {np.std(returns):.6f}")
    print(f"  First 5 values: {returns[:5]}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch S&P 500 returns and verify/write CSV."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write fresh CSV to disk (default: verify existing CSV).",
    )
    args = parser.parse_args()

    print("Fetching S&P 500 returns from numpyro.examples.datasets.SP500 ...")
    returns = fetch_sp500_returns()

    print(f"Fetched shape: {returns.shape}")
    print(f"  Mean: {np.mean(returns):.10f}")
    print(f"  Std: {np.std(returns):.6f}")
    print(f"  First 5 values: {returns[:5]}")

    if args.write:
        print(f"\nWriting to {_OUT} ...")
        write_csv(returns, _OUT)
        print(f"Wrote {returns.shape[0]} returns to {_OUT}")
    else:
        print(f"\nVerifying against {_OUT} (--check mode) ...")
        if not check_mode(returns, _OUT):
            sys.exit(1)


if __name__ == "__main__":
    main()
