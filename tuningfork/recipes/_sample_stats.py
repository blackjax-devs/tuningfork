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

"""Shared sample-stat contracts for generated programs and their evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

SAMPLE_STAT_PREFIX = "_ss_"


@dataclass(frozen=True)
class SampleStatsContract:
    """The emitted fields and semantic constraints for one sampler family."""

    fields: tuple[str, ...]
    positive_integer_fields: tuple[str, ...] = ()
    nonnegative_integer_fields: tuple[str, ...] = ()
    boolean_fields: tuple[str, ...] = ()
    unit_interval_fields: tuple[str, ...] = ()


_HMC_FIELDS = (
    "is_divergent",
    "energy",
    "num_integration_steps",
    "acceptance_rate",
    "is_accepted",
)
_LAPLACE_FIELDS = _HMC_FIELDS + (
    "lbfgs_iter_num",
    "lbfgs_error",
    "lbfgs_converged",
    "lbfgs_hit_maxiter",
)
_RW_FIELDS = ("acceptance_rate", "is_accepted")

_NUTS = SampleStatsContract(
    fields=(
        "is_divergent",
        "energy",
        "num_integration_steps",
        "num_trajectory_expansions",
        "is_turning",
        "acceptance_rate",
    ),
    positive_integer_fields=("num_integration_steps",),
    nonnegative_integer_fields=("num_trajectory_expansions",),
    boolean_fields=("is_divergent", "is_turning"),
    unit_interval_fields=("acceptance_rate",),
)
_HMC = SampleStatsContract(
    fields=_HMC_FIELDS,
    positive_integer_fields=("num_integration_steps",),
    boolean_fields=("is_divergent", "is_accepted"),
    unit_interval_fields=("acceptance_rate",),
)
_LAPLACE = SampleStatsContract(
    fields=_LAPLACE_FIELDS,
    positive_integer_fields=("num_integration_steps",),
    nonnegative_integer_fields=("lbfgs_iter_num",),
    boolean_fields=(
        "is_divergent",
        "is_accepted",
        "lbfgs_converged",
        "lbfgs_hit_maxiter",
    ),
    unit_interval_fields=("acceptance_rate",),
)
_RW = SampleStatsContract(
    fields=_RW_FIELDS,
    boolean_fields=("is_accepted",),
    unit_interval_fields=("acceptance_rate",),
)

SAMPLE_STATS_CONTRACTS: Mapping[str, SampleStatsContract] = {
    "nuts": _NUTS,
    **{
        name: _HMC
        for name in (
            "hmc",
            "mhmc",
            "dmhmc",
            "dynamic_hmc",
            "ghmc",
            "rmhmc",
            "adjusted_mclmc",
            "adjusted_mclmc_dynamic",
        )
    },
    **{
        name: _LAPLACE
        for name in (
            "laplace_hmc",
            "laplace_dhmc",
            "laplace_mhmc",
            "laplace_dmhmc",
        )
    },
    "mclmc": SampleStatsContract(
        fields=("logdensity", "kinetic_change", "energy_change", "nonans"),
        boolean_fields=("nonans",),
    ),
    **{
        name: _RW
        for name in (
            "mala",
            "barker",
            "rwm",
            "irmh",
            "additive_step_random_walk",
            "mgrad_gaussian",
        )
    },
    "orbital_hmc": SampleStatsContract(fields=("weights_mean", "weights_variance")),
    "elliptical_slice": SampleStatsContract(
        fields=("theta", "subiter"),
        nonnegative_integer_fields=("subiter",),
    ),
    "meanfield_vi": SampleStatsContract(fields=()),
    "fullrank_vi": SampleStatsContract(fields=()),
}


def sample_stats_contract(method_name: str) -> SampleStatsContract:
    """Return the single generated-artifact contract for ``method_name``."""
    try:
        return SAMPLE_STATS_CONTRACTS[method_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported sampler for sample stats: {method_name!r}"
        ) from exc


def sample_stat_fields(method_name: str) -> tuple[str, ...]:
    """Return the ordered fields emitted into a generated draws artifact."""
    return sample_stats_contract(method_name).fields


def validate_sample_stats(stats: Mapping[str, np.ndarray], method_name: str) -> None:
    """Fail closed when generated statistics do not match their typed contract."""
    contract = sample_stats_contract(method_name)
    actual = set(stats)
    expected = set(contract.fields)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "sample statistics do not match generated contract: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name in contract.positive_integer_fields:
        value = stats[name]
        if (
            np.issubdtype(value.dtype, np.bool_)
            or not np.issubdtype(value.dtype, np.integer)
            or np.any(value <= 0)
        ):
            raise ValueError(
                f"sample statistic {name!r} must contain positive integers"
            )
    for name in contract.nonnegative_integer_fields:
        value = stats[name]
        if (
            np.issubdtype(value.dtype, np.bool_)
            or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < 0)
        ):
            raise ValueError(
                f"sample statistic {name!r} must contain nonnegative integers"
            )
    for name in contract.boolean_fields:
        if not np.issubdtype(stats[name].dtype, np.bool_):
            raise ValueError(f"sample statistic {name!r} must contain booleans")
    for name in contract.unit_interval_fields:
        value = stats[name]
        if np.any(value < 0) or np.any(value > 1):
            raise ValueError(
                f"sample statistic {name!r} must be within the unit interval"
            )


__all__ = [
    "SAMPLE_STAT_PREFIX",
    "SAMPLE_STATS_CONTRACTS",
    "SampleStatsContract",
    "sample_stat_fields",
    "sample_stats_contract",
    "validate_sample_stats",
]
