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
"""Declarative model configuration shared by Laplace recipe generation."""

from __future__ import annotations

from typing import Any

LAPLACE_OPTIMIZER_KWARG_NAMES: tuple[str, ...] = (
    "maxiter",
    "maxcor",
    "gtol",
    "ftol",
    "maxls",
)

LAPLACE_PHI_THETA_SPLITS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "eight_schools_ncp": (("mu", "tau"), ("theta_raw",)),
    "gp_regression": (
        ("log_lengthscale", "log_kernel_scale", "log_noise_scale"),
        ("f_raw",),
    ),
}


def extract_laplace_optimizer_kwargs(
    primary: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Select explicitly configured optimizer arguments, preferring ``primary``."""
    result: dict[str, Any] = {}
    for key in LAPLACE_OPTIMIZER_KWARG_NAMES:
        if key in primary:
            result[key] = primary[key]
        elif fallback is not None and key in fallback:
            result[key] = fallback[key]
    return result


__all__ = [
    "LAPLACE_OPTIMIZER_KWARG_NAMES",
    "LAPLACE_PHI_THETA_SPLITS",
    "extract_laplace_optimizer_kwargs",
]
