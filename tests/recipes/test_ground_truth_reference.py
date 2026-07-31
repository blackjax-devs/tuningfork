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

import hashlib
import json

import numpy as np
import pytest

from tuningfork.recipes._ground_truth_reference import (
    align_ground_truth,
    load_ground_truth_reference,
)

pytestmark = pytest.mark.fast


def _write_reference(root, summary=None, *, pointer=False):
    base = root / "toy" / "groundtruth_samples" / "blackjax"
    base.mkdir(parents=True)
    draws = base / "draws.npz"
    if pointer:
        draws.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n"
        )
    else:
        np.savez(draws, x=np.arange(6).reshape(2, 3), y=np.ones((2, 3)))
    if summary is None:
        summary = {
            "schema_version": "gt_v2_multichain",
            "n_chains": 2,
            "n_draws_per_chain": 3,
            "n_total": 6,
            "sampler_config": {"seed": 42},
            "per_site": {
                "x": {
                    k: [1.0]
                    for k in (
                        "mean",
                        "std",
                        "q05",
                        "q95",
                        "between_chain_se",
                        "bulk_ess",
                    )
                },
                "y": {
                    k: [2.0]
                    for k in (
                        "mean",
                        "std",
                        "q05",
                        "q95",
                        "between_chain_se",
                        "bulk_ess",
                    )
                },
            },
        }
    (base / "summary_v2.json").write_text(json.dumps(summary))
    return base


def test_load_identity_hashes_and_protocol(tmp_path):
    base = _write_reference(tmp_path)
    reference = load_ground_truth_reference(tmp_path, "toy")
    assert reference.identity["model_name"] == "toy"
    assert reference.identity["protocol"]["sampler_config"] == {"seed": 42}
    assert (
        reference.identity["draws_sha256"]
        == hashlib.sha256((base / "draws.npz").read_bytes()).hexdigest()
    )
    assert (
        reference.identity["lfs_oid"] == "sha256:" + reference.identity["draws_sha256"]
    )


def test_alignment_and_site_restriction(tmp_path):
    _write_reference(tmp_path)
    reference = load_ground_truth_reference(tmp_path, "toy")
    aligned = align_ground_truth(
        reference,
        {"x": np.zeros(3), "y": np.zeros(3)},
        allowed_sites=("y",),
    )
    assert list(aligned) == ["y"]
    assert aligned["y"]["n_total"] == 6


def test_no_overlap_fails(tmp_path):
    _write_reference(tmp_path)
    reference = load_ground_truth_reference(tmp_path, "toy")
    with pytest.raises(ValueError, match="no overlapping"):
        align_ground_truth(reference, {"z": np.zeros(2)})


def test_missing_and_malformed_artifacts_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="summary_v2.json"):
        load_ground_truth_reference(tmp_path, "toy")
    _write_reference(tmp_path, summary={"per_site": {}})
    with pytest.raises(ValueError, match="nonempty per_site"):
        load_ground_truth_reference(tmp_path, "toy")


def test_unhydrated_pointer_mentions_exact_lfs_command(tmp_path):
    (tmp_path / ".git").mkdir()
    _write_reference(tmp_path, pointer=True)
    with pytest.raises(RuntimeError) as exc:
        load_ground_truth_reference(tmp_path, "toy")
    assert 'git lfs pull --include="toy/groundtruth_samples/blackjax/draws.npz"' in str(
        exc.value
    )
