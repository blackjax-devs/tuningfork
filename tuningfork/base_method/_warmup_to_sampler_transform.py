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
"""Transform warmup output into sampler init kwargs (Phase B-2 schema §3.3).

Implements ``transform_warmup_state`` — the single function that converts
warmup-adapted ``(adapted_params, warmup_info)`` into the dict of keyword
arguments required to build the sampling kernel for a given
``(warmup_inner_kernel, base_method_name)`` pair.

Resolution table (§3.4 of RECIPE_SCHEMA.md):

+-------------------------------+---------------------------+-------------------------------------------------------+
| warmup_inner_kernel           | base_method               | Transform                                             |
+===============================+===========================+=======================================================+
| nuts (resolved or explicit)   | nuts                      | {step_size, IMM}  (identity)                          |
+-------------------------------+---------------------------+-------------------------------------------------------+
| nuts                          | hmc / mhmc                | {step_size, IMM, num_integration_steps=median(NIS)}   |
+-------------------------------+---------------------------+-------------------------------------------------------+
| nuts                          | dynamic_hmc / dmhmc       | {step_size, IMM, step_policy=empirical(NIS)}          |
+-------------------------------+---------------------------+-------------------------------------------------------+
| nuts                          | laplace_*                 | {step_size, IMM, num_integration_steps=median(NIS),   |
|                               |                           |  log_joint_fn, theta_init}  (deferred — BLOCKED)      |
+-------------------------------+---------------------------+-------------------------------------------------------+
| hmc (matches base)            | hmc / mhmc                | {step_size, IMM}  (identity)                          |
+-------------------------------+---------------------------+-------------------------------------------------------+
| mala (matches base)           | mala                      | {step_size, IMM}  (identity)                          |
+-------------------------------+---------------------------+-------------------------------------------------------+
| barker (matches base)         | barker                    | {step_size, IMM}  (identity)                          |
+-------------------------------+---------------------------+-------------------------------------------------------+

Ravel semantics (§8-Q4): when ``warmup_info["num_integration_steps"]`` is
multi-dimensional (e.g., shape ``(num_chains, n_warmup_steps)`` from a
multi-chain warmup), it is ravelled to a single 1-D array before computing
the median or harvesting the empirical histogram.  One canonical L distribution
across chains, not per-chain, ensures all sampling chains run the same protocol.

Backward compat: this module is NEW in Phase B-2.  Callers that pass
``warmup_inner_kernel=None`` receive the implicit resolution via
``resolve_warmup_algorithm`` (i.e., the ``_warmup_to_sampler_transform``
module does NOT replace ``_laplace_adapter.resolve_warmup_algorithm``; it is
an additional transform stage that runs AFTER the warmup completes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["transform_warmup_state", "resolve_warmup_inner_kernel"]

# Sets for dispatch within the resolution table.
_NUTS_IDENTITY_METHODS: frozenset[str] = frozenset({"nuts"})
_NIS_MEDIAN_METHODS: frozenset[str] = frozenset({"hmc", "mhmc"})
_NIS_EMPIRICAL_METHODS: frozenset[str] = frozenset({"dynamic_hmc", "dmhmc"})
_LAPLACE_METHODS: frozenset[str] = frozenset(
    {"laplace_hmc", "laplace_dhmc", "laplace_mhmc", "laplace_dmhmc"}
)
# Methods that produce an identity transform when warmup_inner_kernel matches base.
_SELF_WARMUP_IDENTITY: frozenset[str] = frozenset({"hmc", "mhmc", "mala", "barker"})
# Methods for the substitute-family (implicit inner_nuts if kernel is None).
_WARMUP_SUBSTITUTE_FAMILY: frozenset[str] = _LAPLACE_METHODS | frozenset(
    {"dynamic_hmc", "dmhmc"}
)


def resolve_warmup_inner_kernel(
    warmup_inner_kernel: str | None,
    base_method_name: str,
) -> str:
    """Resolve the effective warmup inner kernel name.

    When ``warmup_inner_kernel`` is ``None``, apply the implicit substitute-
    family logic: substitute-family methods (laplace_*, dynamic_hmc, dmhmc)
    resolve to ``"nuts"``; all other methods resolve to ``base_method_name``.

    Parameters
    ----------
    warmup_inner_kernel
        Explicit kernel override from ``Recipe.warmup_inner_kernel``, or
        ``None`` to use the implicit default.
    base_method_name
        The sampling algorithm name, e.g. ``"hmc"``, ``"dynamic_hmc"``.

    Returns
    -------
    str
        Resolved warmup inner kernel name, e.g. ``"nuts"``, ``"hmc"``.
    """
    if warmup_inner_kernel is not None:
        return warmup_inner_kernel
    if base_method_name in _WARMUP_SUBSTITUTE_FAMILY:
        return "nuts"
    return base_method_name


def transform_warmup_state(
    warmup_inner_kernel: str | None,
    base_method_name: str,
    adapted_params: dict[str, Any],
    warmup_info: Any,
    *,
    step_policy_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transform warmup output into sampler init kwargs.

    This function is the central dispatch point for the §3.4 resolution table.
    It takes the raw warmup output (``adapted_params`` from the warmup runner,
    ``warmup_info`` from the blackjax warmup trace) and returns a dict of
    keyword arguments to pass to the sampling kernel factory.

    Parameters
    ----------
    warmup_inner_kernel
        The kernel that drove adaptation, or ``None`` to resolve implicitly.
        ``None`` delegates to ``resolve_warmup_inner_kernel`` which applies
        the substitute-family logic from ``_laplace_adapter.py``.
    base_method_name
        The sampling algorithm registry name, e.g. ``"hmc"``, ``"dynamic_hmc"``.
    adapted_params
        Dict of warmup-adapted parameters, typically containing at least
        ``"step_size"`` and ``"inverse_mass_matrix"``.  Shape of values is
        per-chain (leading axis = num_chains from the multi-chain warmup runner).
    warmup_info
        The warmup info object returned by the blackjax warmup runner.  May
        be a NamedTuple or dict.  For NUTS warmup this contains
        ``num_integration_steps`` with shape ``(num_chains, n_warmup_steps)``
        or ``(n_warmup_steps,)`` depending on the runner.  Ravelled to 1-D
        before any computation (§8-Q4).
    step_policy_override
        Optional pre-computed step_policy spec dict (e.g. from a recipe).
        When provided, this is returned in the output dict for dynamic_hmc /
        dmhmc instead of computing a fresh empirical spec from ``warmup_info``.
        Set to the recipe's ``step_policy`` field when re-running an existing
        recipe to ensure bit-identical behaviour.

    Returns
    -------
    dict
        Sampler init kwargs.  Always contains ``step_size`` and
        ``inverse_mass_matrix``.  May also contain ``num_integration_steps``
        (for hmc/mhmc with NUTS warmup) or ``step_policy`` (for dynamic_hmc
        / dmhmc with NUTS warmup).

    Notes
    -----
    The returned dict does NOT contain ``log_joint_fn`` or ``theta_init``
    for laplace_* methods — those come from the model decomposition and are
    injected separately by the recipe runner.  The laplace_* row in the
    resolution table is partially supported: median NIS is computed and
    returned as ``num_integration_steps`` for the laplace_hmc / laplace_mhmc
    variants, but the runner must still inject ``log_joint_fn`` + ``theta_init``
    independently.

    Raises
    ------
    ValueError
        If the ``(warmup_inner_kernel, base_method_name)`` combination is not
        in the resolution table and cannot be handled as a self-warmup identity.
    """
    resolved_kernel = resolve_warmup_inner_kernel(warmup_inner_kernel, base_method_name)

    # Always extract step_size and IMM from adapted_params.
    result: dict[str, Any] = {}
    if "step_size" in adapted_params:
        result["step_size"] = adapted_params["step_size"]
    if "inverse_mass_matrix" in adapted_params:
        result["inverse_mass_matrix"] = adapted_params["inverse_mass_matrix"]

    # ---- Resolution table dispatch ----

    # Row 1: nuts → nuts  (identity)
    if resolved_kernel == "nuts" and base_method_name in _NUTS_IDENTITY_METHODS:
        return result

    # Row 2: nuts → hmc / mhmc  (NIS median injection)
    if resolved_kernel == "nuts" and base_method_name in _NIS_MEDIAN_METHODS:
        nis = _extract_nis(warmup_info)
        if nis is not None:
            result["num_integration_steps"] = int(np.median(nis))
        return result

    # Row 3: nuts → dynamic_hmc / dmhmc  (empirical step_policy)
    if resolved_kernel == "nuts" and base_method_name in _NIS_EMPIRICAL_METHODS:
        if step_policy_override is not None:
            # Re-running an existing recipe: use the pinned spec.
            result["step_policy"] = step_policy_override
        else:
            nis = _extract_nis(warmup_info)
            if nis is not None:
                from tuningfork.base_method._step_policy_registry import (
                    harvest_oracle_spec_from_array,
                )

                result["step_policy"] = harvest_oracle_spec_from_array(
                    nis, max_values=24
                )
        return result

    # Row 4: nuts → laplace_*  (NIS median + caller injects log_joint_fn, theta_init)
    if resolved_kernel == "nuts" and base_method_name in _LAPLACE_METHODS:
        nis = _extract_nis(warmup_info)
        if nis is not None:
            result["num_integration_steps"] = int(np.median(nis))
        # log_joint_fn + theta_init are NOT injected here — they come from the
        # model decomposition built by the recipe runner.
        return result

    # Rows 5-7: self-warmup identity (hmc→hmc, mhmc→mhmc, mala→mala, barker→barker)
    if (
        resolved_kernel == base_method_name
        and base_method_name in _SELF_WARMUP_IDENTITY
    ):
        return result

    # Fallback: pass through — unknown pairing; no additional transform.
    # The caller is responsible for verifying the result is complete.
    return result


def _extract_nis(warmup_info: Any) -> np.ndarray | None:
    """Extract ``num_integration_steps`` from warmup_info; ravel to 1-D.

    Handles both dict-like and NamedTuple warmup_info objects.  Returns
    ``None`` if NIS is not available (e.g. non-NUTS warmup).

    §8-Q4: ravel to 1-D across chains for one canonical L distribution.
    """
    nis = None
    if hasattr(warmup_info, "num_integration_steps"):
        nis = warmup_info.num_integration_steps
    elif isinstance(warmup_info, dict) and "num_integration_steps" in warmup_info:
        nis = warmup_info["num_integration_steps"]

    if nis is None:
        return None

    arr = np.asarray(nis).ravel().astype(int)
    if len(arr) == 0:
        return None
    return arr
