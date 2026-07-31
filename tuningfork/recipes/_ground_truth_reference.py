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
"""Load and align canonical ground-truth reference artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GROUND_TRUTH_REFERENCE_SCHEMA = "tuningfork.ground-truth-reference.v1"
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class GroundTruthReference:
    """Validated canonical summary and the paths from which it was loaded."""

    model_name: str
    summary: dict[str, Any]
    summary_path: Path
    draws_path: Path
    identity: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifact(path: Path, catalog_root: Path) -> None:
    command = _hydration_command(path, catalog_root)
    if not path.exists():
        raise FileNotFoundError(
            f"canonical ground-truth artifact missing: {path}; "
            f"hydrate it with `{command}`"
        )
    with path.open("rb") as stream:
        head = stream.read(256)
    if head.startswith(_LFS_POINTER_MAGIC):
        raise RuntimeError(
            f"canonical ground-truth artifact is an unhydrated Git-LFS pointer: {path}; "
            f"hydrate it with `{command}`"
        )


def _hydration_command(path: Path, catalog_root: Path) -> str:
    """Return a command whose include path is valid from the nearest Git root."""
    for candidate in (catalog_root, *catalog_root.parents):
        if (candidate / ".git").exists():
            relative = path.relative_to(candidate).as_posix()
            return f'git lfs pull --include="{relative}"'
    relative = path.relative_to(catalog_root).as_posix()
    return f'git lfs pull --include="{relative}"'


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _strict_json_copy(value: Any) -> Any:
    """Validate *value* as finite JSON and return an independent copy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ground-truth metadata contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_strict_json_copy(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("ground-truth metadata keys must be strings")
        return {key: _strict_json_copy(item) for key, item in value.items()}
    raise TypeError(f"ground-truth metadata is not JSON-safe: {type(value).__name__}")


def _positive_int(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or int(value) != value
        or value <= 0
    ):
        raise ValueError(f"summary_v2.json has invalid positive {key}: {value!r}")
    return int(value)


def load_ground_truth_reference(
    catalog_root: Path, model_name: str
) -> GroundTruthReference:
    """Load only the canonical ``groundtruth_samples/blackjax`` artifacts."""
    root = Path(catalog_root)
    base = root / model_name / "groundtruth_samples" / "blackjax"
    summary_path = base / "summary_v2.json"
    draws_path = base / "draws.npz"
    _require_artifact(summary_path, root)
    _require_artifact(draws_path, root)
    try:
        summary = json.loads(
            summary_path.read_text(),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed canonical summary: {summary_path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(
            f"malformed canonical summary: {summary_path}: expected an object"
        )
    per_site = summary.get("per_site")
    if not isinstance(per_site, dict) or not per_site:
        raise ValueError(
            f"malformed canonical summary: {summary_path}: nonempty per_site required"
        )
    n_chains = _positive_int(summary, "n_chains")
    n_draws = _positive_int(summary, "n_draws_per_chain")
    n_total = _positive_int(summary, "n_total")
    if n_total != n_chains * n_draws:
        raise ValueError(
            f"malformed canonical summary: {summary_path}: n_total is inconsistent"
        )
    summary_hash = _sha256(summary_path)
    draws_hash = _sha256(draws_path)
    rel_summary = summary_path.relative_to(root).as_posix()
    rel_draws = draws_path.relative_to(root).as_posix()
    protocol = _strict_json_copy(
        {key: value for key, value in summary.items() if key != "per_site"}
    )
    identity = {
        "schema": GROUND_TRUTH_REFERENCE_SCHEMA,
        "model_name": model_name,
        "summary_path": rel_summary,
        "draws_path": rel_draws,
        "summary_sha256": summary_hash,
        "draws_sha256": draws_hash,
        "lfs_oid": f"sha256:{draws_hash}",
        "protocol": protocol,
    }
    return GroundTruthReference(
        model_name=model_name,
        summary=copy.deepcopy(summary),
        summary_path=summary_path,
        draws_path=draws_path,
        identity=identity,
    )


def align_ground_truth(
    reference: GroundTruthReference,
    draws: Mapping[str, Any],
    *,
    allowed_sites: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Align validated summary statistics with sampled draw sites."""
    per_site = reference.summary.get("per_site")
    if not isinstance(per_site, dict):
        raise ValueError("ground-truth reference has malformed per_site")
    allowed = set(allowed_sites) if allowed_sites is not None else None
    sites = [
        name
        for name in draws
        if name in per_site and (allowed is None or name in allowed)
    ]
    if not sites:
        raise ValueError("ground-truth draws and summary have no overlapping sites")
    required = ("mean", "std", "q05", "q95", "between_chain_se", "bulk_ess")
    aligned: dict[str, dict[str, Any]] = {}
    n_total = reference.summary.get("n_total")
    if (
        isinstance(n_total, bool)
        or not isinstance(n_total, (int, float))
        or n_total <= 0
    ):
        raise ValueError("ground-truth reference has malformed n_total")
    for site in sites:
        stats = per_site[site]
        if not isinstance(stats, dict) or any(
            key not in stats or stats[key] is None for key in required
        ):
            raise ValueError(
                f"ground-truth summary has malformed required statistics for site {site!r}"
            )
        try:
            for key in required:
                values = np.asarray(stats[key], dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError("non-finite value")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ground-truth summary has malformed statistics for site {site!r}"
            ) from exc
        aligned[site] = {key: copy.deepcopy(stats[key]) for key in required}
        aligned[site]["n_total"] = int(n_total)
    return aligned


__all__ = [
    "GROUND_TRUTH_REFERENCE_SCHEMA",
    "GroundTruthReference",
    "align_ground_truth",
    "load_ground_truth_reference",
]
