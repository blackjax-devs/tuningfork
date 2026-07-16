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
"""Sample layout helpers — reshape single-chain samples into multichain form."""

import numpy as np


def _samples_to_multichain(
    samples: dict,
    n_chunks: int,
    multichain: bool | None = None,
) -> dict:
    """Ensure samples are (n_chains, n_draws, *shape); rechunk if needed.

    When ``multichain`` is explicitly provided, it bypasses the heuristic:
    - ``True``: treat as multichain, return as-is.
    - ``False``: treat as single-chain, reshape into n_chunks segments.
    - ``None`` (default): use heuristic to detect layout (see below).

    **Heuristic (when multichain=None)**:
    If the first array in ``samples`` has ndim ≥ 3, it is definitively
    multichain: single-chain positions are (n_samples, *event_shape);
    multichain are (n_chains, n_samples, *event_shape). For ndim < 3,
    a shape-based fallback distinguishes them conservatively (first dim
    < 64 treated as n_chains). This heuristic is permissive to avoid
    the ≤64 cliff bug (issue #217) where genuine multichain arrays with
    nc>64 were misclassified as single-chain.

    **Precondition**: Callers must call ``jax.block_until_ready(samples)`` before
    invoking this function. JAX arrays passed in are expected to be fully
    materialised; ``np.asarray`` here is used for shape inspection and conversion
    to ArviZ input format only.

    Parameters
    ----------
    samples
        Dict mapping param name → array of shape
        ``(n_chains, n_draws, *event_shape)`` or ``(n_draws, *event_shape)``.
    n_chunks
        Number of contiguous segments to reshape into when single-chain.
    multichain
        Explicit layout hint. When ``True``, return as-is (cast to np).
        When ``False``, rechunk into n_chunks. When ``None`` (default),
        use the heuristic below.

    Returns
    -------
    dict
        Dict where each array has shape ``(n_chains, n_draws, *event_shape)``.
    """
    if not samples:
        return samples
    first = next(iter(samples.values()))
    arr = np.asarray(first)

    # Determine if samples are multichain
    if multichain is not None:
        is_multichain = multichain
    else:
        # Heuristic: ndim ≥ 3 is definitively multichain.
        # Single-chain: (n_samples, *event_shape) has ndim ≤ 2.
        # Multichain: (n_chains, n_samples, *event_shape) has ndim ≥ 3.
        #
        # For ndim < 3: fallback to first-axis heuristic (conservative).
        # The original ≤64 cliff caused issue #217: arrays with nc>64 were
        # misclassified as single-chain and incorrectly rechunked.
        # The new heuristic (ndim ≥ 3) avoids this cliff entirely.
        if arr.ndim >= 3:
            is_multichain = True
        else:
            # ndim <= 2: first dim ≤ 64 → treat as n_chains.
            is_multichain = arr.ndim >= 2 and arr.shape[0] <= 64

    if is_multichain:
        return {k: np.asarray(v) for k, v in samples.items()}

    # Single-chain: reshape into n_chunks chunks
    result = {}
    for name, v in samples.items():
        v_np = np.asarray(v)
        n_total = v_np.shape[0]
        event_shape = v_np.shape[1:]
        chunk_size = n_total // n_chunks
        trimmed = v_np[: n_chunks * chunk_size]
        result[name] = trimmed.reshape(n_chunks, chunk_size, *event_shape)
    return result
