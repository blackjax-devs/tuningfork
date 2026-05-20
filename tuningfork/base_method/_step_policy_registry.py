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

Phase A supports:

- ``spec is None`` → V0 library default: ``lambda key: jax.random.randint(key, (), 1, 10)``
- ``spec["kind"] == "uniform_int"`` → V0/V1/V2 uniform integer in [low, high)

All other kinds raise ``NotImplementedError("kind <X> deferred to Phase B")``.

References
----------
- ``worklog/threads/d-hmc-integration-steps-fn-matrix.md`` §5 — full spec
  (Path A parametric + Path B empirical).
- ``worklog/decisions/2026-05-21-step-policy-catalog.md`` — design-doc
  anchors (not yet created; Phase B task per §10 of the plan).
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["build_step_policy"]

# Library-default V0 step policy (uniform integer in [1, 10)).
# Matches blackjax.mcmc.dynamic_hmc's built-in default integration_steps_fn.
_V0_LOW: int = 1
_V0_HIGH: int = 10


def build_step_policy(spec: dict | None) -> Callable:
    """Reconstruct ``integration_steps_fn`` from a step_policy spec dict.

    Parameters
    ----------
    spec
        A step_policy spec dict (from ``Recipe.step_policy``), or ``None``.
        ``None`` returns the library default (V0).

        Supported kinds (Phase A):

        - ``None`` → V0: ``lambda key: jax.random.randint(key, (), 1, 10)``
        - ``{"kind": "uniform_int", "low": L, "high": H}`` → uniform integer
          in [L, H) (int bounds; matches ``jax.random.randint`` semantics:
          ``low`` inclusive, ``high`` exclusive).

        All other kinds raise ``NotImplementedError``.

    Returns
    -------
    Callable
        A JAX-jittable function ``(key: jax.Array) -> jax.Array`` returning
        a scalar integer step count.

    Raises
    ------
    NotImplementedError
        For any ``kind`` not yet implemented (Phase B deferred).
    ValueError
        For a ``uniform_int`` spec with invalid or missing ``low``/``high``.

    Examples
    --------
    >>> fn = build_step_policy(None)
    >>> # fn(key) returns randint in [1, 10) — library default V0

    >>> fn = build_step_policy({"kind": "uniform_int", "low": 50, "high": 200})
    >>> # fn(key) returns randint in [50, 200) — V2 long trajectory
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

    # All other kinds are deferred to Phase B.
    _DEFERRED_KINDS = ("log_uniform_int", "poisson", "pow2_choice", "empirical")
    if kind in _DEFERRED_KINDS:
        raise NotImplementedError(
            f"step_policy kind {kind!r} is deferred to Phase B; "
            f"Phase A supports only None (V0) and 'uniform_int'."
        )

    raise NotImplementedError(
        f"Unknown step_policy kind {kind!r}; "
        f"Phase A supports only None (V0) and 'uniform_int'."
    )
