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
"""Declarative metadata for SMC parameter-update strategies.

The registry is intentionally data-only: generated programs must not import
this module (or tuningfork) at runtime.  Strategy implementations live in the
SMC emitter, where a validated descriptor selects a standalone source snippet.

Selection guidance: ``none`` is the baseline and is required outside
``inner_kernel_tuning``.  Every update returns the full initial parameter-key
set because BlackJAX replaces the complete override pytree; callbacks receive
the underlying tempered state, and step size is adapted relative to its initial
scale.  Acceptance-rate step-size adaptation suits smooth, near-Gaussian
posteriors (logistic regression and GMM); particle-cloud IMM adaptation is
useful when scales change across tempering (funnels and hierarchical models).
IMM can lag a rapidly changing target and overshoot on smooth cells, so use
the combined strategy only when that geometry warrants it. IMM values are
broadcast to the initial inverse-mass shape.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ParameterUpdateStrategy:
    """Immutable, JSON-friendly strategy metadata used during planning."""

    name: str
    allowed_kwargs: frozenset[str]
    description: str


PARAMETER_UPDATE_STRATEGIES = MappingProxyType(
    {
        "none": ParameterUpdateStrategy(
            "none", frozenset(), "Keep all inner-kernel parameters fixed."
        ),
        "step_size_from_acceptance_rate": ParameterUpdateStrategy(
            "step_size_from_acceptance_rate",
            frozenset({"target_acceptance"}),
            "Adapt step size from mutation acceptance rates (default target 0.65).",
        ),
        "imm_from_particles": ParameterUpdateStrategy(
            "imm_from_particles",
            frozenset(),
            "Adapt diagonal inverse mass from particle-cloud variance.",
        ),
        "step_size_and_imm_from_particles": ParameterUpdateStrategy(
            "step_size_and_imm_from_particles",
            frozenset({"target_acceptance"}),
            "Adapt step size and diagonal inverse mass from acceptance and "
            "particles (default target 0.65).",
        ),
    }
)

__all__ = ["PARAMETER_UPDATE_STRATEGIES", "ParameterUpdateStrategy"]
