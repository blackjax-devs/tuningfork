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
"""Groundtruth test helpers and fixtures."""

from __future__ import annotations

from pathlib import Path

__all__ = ["_is_lfs_pointer"]

# Sentinel bytes at the start of every git-LFS pointer stub.
_LFS_MAGIC = b"version https://git-lfs.github.com"


def _is_lfs_pointer(path: Path) -> bool:
    """Return True if *path* is an unsmudged git-LFS pointer stub.

    Committed catalog draws.npz files are LFS-tracked.  When git-LFS is not
    available (e.g. CI without ``lfs: true``), the checkout leaves a ~130-byte
    plain-text pointer file that starts with ``version https://git-lfs.github.com``.
    Passing such a file to ``np.load`` fails with an ``UnpicklingError``.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(_LFS_MAGIC)
    except OSError:
        return False
