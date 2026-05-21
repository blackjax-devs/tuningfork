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
"""Step-policy registry for ``dynamic_hmc`` and ``dmhmc``.

Reconstructs the ``integration_steps_fn`` callable from a JSON-serialisable
spec dict stored in ``Recipe.step_policy``.  The spec uses a ``kind`` field to
select the distribution family; family-specific fields parameterise it.

Phase A (merged, PR #39):

- ``spec is None`` → V0 library default: ``lambda key: jax.random.randint(key, (), 1, 10)``
- ``spec["kind"] == "uniform_int"`` → V0/V1/V2 uniform integer in [low, high)

Phase B (this module):

- ``spec["kind"] == "empirical"`` → V7 empirical oracle via inverse-CDF
  sampling over a normalised histogram ``{"values": [...], "weights": [...]}``.
  See ``_build_empirical_step_policy`` and ``harvest_oracle_spec``.

References
----------
- ``worklog/threads/d-hmc-integration-steps-fn-matrix.md`` §5 — full spec
  (Path A parametric + Path B empirical).
- ``worklog/decisions/2026-05-21-step-policy-catalog.md`` — design-doc
  anchors (not yet created; Phase B task per §10 of the plan).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["build_step_policy", "harvest_oracle_spec", "harvest_oracle_spec_from_array"]

# Library-default V0 step policy (uniform integer in [1, 10)).
# Matches blackjax.mcmc.dynamic_hmc's built-in default integration_steps_fn.
_V0_LOW: int = 1
_V0_HIGH: int = 10


def _build_empirical_step_policy(spec: dict) -> Callable:
    """Build a jittable inverse-CDF sampler from an empirical spec.

    Parameters
    ----------
    spec
        Must contain:
        - ``"values"``: list of distinct positive integers (sorted, non-empty)
        - ``"weights"``: list of non-negative floats (normalised probabilities,
          same length as ``values``)

    Returns
    -------
    Callable
        A JAX-jittable function ``(key: jax.Array) -> jax.Array`` that samples
        from the empirical distribution via inverse-CDF lookup.
    """
    import jax
    import jax.numpy as jnp

    if "values" not in spec or "weights" not in spec:
        raise ValueError(
            f"empirical step_policy requires 'values' and 'weights' fields; "
            f"got spec keys: {list(spec.keys())!r}"
        )
    values_list = spec["values"]
    weights_list = spec["weights"]
    if len(values_list) == 0:
        raise ValueError("empirical step_policy 'values' must be non-empty")
    if len(values_list) != len(weights_list):
        raise ValueError(
            f"empirical step_policy 'values' and 'weights' must have the same length; "
            f"got {len(values_list)} vs {len(weights_list)}"
        )

    values = jnp.array(values_list, dtype=jnp.int32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    # Normalise defensively (handles slight floating-point drift from serialisation)
    weights = weights / weights.sum()
    cdf = jnp.cumsum(weights)  # shape (N,); cdf[-1] == 1.0

    n = len(values_list)

    def fn(key: jax.Array) -> jax.Array:
        u = jax.random.uniform(key)  # u in [0, 1)
        # searchsorted(cdf, u, side="right") gives the first index i where cdf[i] > u,
        # which implements the standard inverse-CDF for a discrete distribution.
        idx = jnp.searchsorted(cdf, u, side="right")
        # Clip to valid range [0, n-1] to guard against floating-point edge cases.
        idx = jnp.clip(idx, 0, n - 1)
        return values[idx]

    return fn


def build_step_policy(spec: dict | None) -> Callable:
    """Reconstruct ``integration_steps_fn`` from a step_policy spec dict.

    Parameters
    ----------
    spec
        A step_policy spec dict (from ``Recipe.step_policy``), or ``None``.
        ``None`` returns the library default (V0).

        Supported kinds:

        - ``None`` → V0: ``lambda key: jax.random.randint(key, (), 1, 10)``
        - ``{"kind": "uniform_int", "low": L, "high": H}`` → uniform integer
          in [L, H) (``low`` inclusive, ``high`` exclusive;
          matches ``jax.random.randint`` semantics).
        - ``{"kind": "empirical", "values": [...], "weights": [...]}`` →
          V7 empirical oracle: inverse-CDF sampling over a discrete NIS
          histogram harvested from a NUTS chain.

        Other kinds (``"log_uniform_int"``, ``"poisson"``, ``"pow2_choice"``)
        raise ``NotImplementedError`` (deferred to Phase C+).

    Returns
    -------
    Callable
        A JAX-jittable function ``(key: jax.Array) -> jax.Array`` returning
        a scalar integer step count.

    Raises
    ------
    NotImplementedError
        For kinds deferred to Phase C+ or unknown kinds.
    ValueError
        For a ``uniform_int`` spec with invalid or missing ``low``/``high``,
        or an ``empirical`` spec missing ``values``/``weights``.

    Examples
    --------
    >>> fn = build_step_policy(None)
    >>> # fn(key) returns randint in [1, 10) — library default V0

    >>> fn = build_step_policy({"kind": "uniform_int", "low": 50, "high": 200})
    >>> # fn(key) returns randint in [50, 200) — V2 long trajectory

    >>> spec = {"kind": "empirical", "values": [60, 80, 100], "weights": [0.3, 0.5, 0.2]}
    >>> fn = build_step_policy(spec)
    >>> # fn(key) samples from the discrete distribution {60:30%, 80:50%, 100:20%}
    """
    import jax

    if spec is None:
        # V0: library default — uniform integer in [1, 10)
        return lambda key: jax.random.randint(key, (), _V0_LOW, _V0_HIGH)

    kind = spec.get("kind")

    if kind == "uniform_int":
        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"uniform_int step_policy requires 'low' and 'high' fields; "
                f"got spec={spec!r}"
            )
        low = int(spec["low"])
        high = int(spec["high"])
        if low >= high:
            raise ValueError(
                f"uniform_int step_policy requires low < high; "
                f"got low={low}, high={high}"
            )
        return lambda key: jax.random.randint(key, (), low, high)

    if kind == "empirical":
        return _build_empirical_step_policy(spec)

    # All other kinds are deferred to Phase C+.
    _DEFERRED_KINDS = ("log_uniform_int", "poisson", "pow2_choice")
    if kind in _DEFERRED_KINDS:
        raise NotImplementedError(
            f"step_policy kind {kind!r} is deferred to Phase C+; "
            f"currently supported: None (V0), 'uniform_int', 'empirical'."
        )

    raise NotImplementedError(
        f"Unknown step_policy kind {kind!r}; "
        f"supported: None (V0), 'uniform_int', 'empirical'."
    )


def harvest_oracle_spec_from_array(
    nis_array: Any,
    *,
    max_values: int = 512,
) -> dict:
    """Extract a step_policy empirical spec from a raw NIS integer array.

    Core implementation used by both ``harvest_oracle_spec`` (file-based)
    and by the recipe runner (live warmup_info harvest).

    The recipe runner extracts ``warmup_info["num_integration_steps"]`` from
    the NUTS warmup call and passes it here directly — no separate file I/O
    required.  This is **Path B** in the Phase B design:

    - Path A (``harvest_oracle_spec``): harvest from a pre-existing
      ``chain_stats.npz`` file.
    - Path B (this function): harvest from the live ``warmup_info`` returned
      by the same NUTS warmup run being used to adapt (step_size, IMM).
      Cheaper: no separate run needed; oracle and warmup are co-produced.

    Parameters
    ----------
    nis_array
        Integer array of ``num_integration_steps`` values.  Any shape;
        ravelled before processing.
    max_values
        Cap on distinct L values in the returned spec.  For high-variance
        models (horseshoe, gp_regression), histogram binning reduces to
        ``max_values`` bin-centre integers.  Default 512.

    Returns
    -------
    dict
        ``{"kind": "empirical", "values": [...], "weights": [...]}``.

    Raises
    ------
    ValueError
        If the NIS array is empty or all-zero.
    """
    import numpy as np

    nis = np.asarray(nis_array).ravel().astype(int)
    if len(nis) == 0:
        raise ValueError("NIS array is empty — cannot harvest oracle spec")
    if nis.sum() == 0:
        raise ValueError(
            "All num_integration_steps values are zero — degenerate chain; "
            "cannot harvest oracle spec"
        )

    min_l, max_l = int(nis.min()), int(nis.max())
    counts = np.bincount(nis, minlength=max_l + 1)
    # Non-zero L values only
    l_values = np.where(counts > 0)[0]
    l_counts = counts[l_values]

    if len(l_values) > max_values:
        # Histogram binning for high-NIS-variance models (horseshoe, gp_regression).
        bin_edges = np.linspace(min_l, max_l + 1, max_values + 1)
        bin_counts, _ = np.histogram(nis, bins=bin_edges)
        bin_centres = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
        mask = bin_counts > 0
        l_values = bin_centres[mask]
        l_counts = bin_counts[mask]

    weights = l_counts / l_counts.sum()
    return {
        "kind": "empirical",
        "values": l_values.tolist(),
        "weights": weights.tolist(),
    }


def harvest_oracle_spec(
    chain_stats_path: Path | str,
    *,
    max_values: int = 512,
) -> dict:
    """Extract a step_policy empirical spec from a NUTS chain_stats.npz (Path A).

    Reads ``num_integration_steps`` from the chain_stats file and delegates to
    ``harvest_oracle_spec_from_array``.  This is **Path A** of the oracle
    harvest workflow — used when a pre-existing chain_stats file is available
    (e.g., from a previously-run nuts×wadapt recipe cache).

    For in-line harvest from a live warmup run, use
    ``harvest_oracle_spec_from_array`` directly (Path B).

    Parameters
    ----------
    chain_stats_path
        Path to a ``<model>/_cache/<recipe_stem>.chain_stats.npz`` file from
        any passing ``nuts × window_adapt*`` recipe for the target model.
        May also point to a groundtruth ``chain_stats.npz`` file.
    max_values
        Forwarded to ``harvest_oracle_spec_from_array``.

    Returns
    -------
    dict
        ``{"kind": "empirical", "values": [...], "weights": [...]}``.

    Raises
    ------
    KeyError
        If the chain_stats file does not contain ``num_integration_steps``.
    ValueError
        If the NIS array is empty or all-zero.
    """
    import numpy as np

    chain_stats_path = Path(chain_stats_path)
    data = np.load(str(chain_stats_path))
    # Support both key names: newer NUTS uses "num_integration_steps";
    # older caches may use "n_steps" (HMC warmup substitute per-step count).
    if "num_integration_steps" in data.files:
        nis = data["num_integration_steps"]
    elif "n_steps" in data.files:
        nis = data["n_steps"]
    else:
        raise KeyError(
            f"chain_stats file at {chain_stats_path} does not contain "
            f"'num_integration_steps' or 'n_steps'; "
            f"available keys: {list(data.files)}"
        )
    return harvest_oracle_spec_from_array(nis, max_values=max_values)
