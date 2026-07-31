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
"""SMCMethod: registry-entry schema for Sequential Monte Carlo algorithms.

Sister abstraction to ``BaseMethod`` (parallel registry ``SMC_METHODS``).
SMC descriptors carry the metadata needed to validate and emit generated
programs. Sampling itself belongs to generated source.

See ``tuningfork/smc/__init__.py`` for the ``SMC_METHODS`` registry.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from tuningfork.base_method._base import HyperparamSpace  # REUSED

__all__ = ["SMCMethod"]


@dataclass(frozen=True)
class SMCMethod:
    """Registry entry for an SMC algorithm.

    Sister abstraction to ``BaseMethod`` (parallel registry ``SMC_METHODS``).
    Why a sister abstraction (not a 3rd specialised flag on BaseMethod):
    SMC needs (a) prior/likelihood split, (b) inner kernel composition,
    (c) particles (initial_particles array) instead of a single position
    at init time, (d) per-step extra args for some variants
    (e.g. data_mask for partial_posteriors). Forcing into BaseMethod would
    require ~5 new required factory kwargs and break the no_warmup-with-defaults
    flow.

    Compatible inner methods (statistician verdict, see WORKLOG): MH-based only.
    Excludes ``mclmc``, ``adjusted_mclmc``, ``adjusted_mclmc_dynamic``
    (microcanonical invariance broken by tempering). Default ``"rwm"``
    (statistician: "RWM/IRMH first").

    Multi-chain contract: SMC does NOT follow the multi-chain contract of MCMC
    methods (``num_chains=1`` convention; particles ARE the parallelism).

    Parameters
    ----------
    name
        Unique identifier, e.g. ``"adaptive_tempered_smc"``.
    family
        Always ``"smc"``.
    compatible_inner_methods
        Tuple of ``BASE_METHODS`` keys this SMC variant accepts as inner
        kernel.  Must be MH-based.  Non-empty.
    default_inner_method
        Default inner kernel name; must be in ``compatible_inner_methods``.
    num_particles_default
        Default particle count for tests / LOW recipes.  Default ``1000``.
    default_hp_space
        Tuple of ``HyperparamSpace`` for declared SMC-level parameters (target_ess,
        num_mcmc_steps, etc.).  Non-empty.
    step_kwargs_schema
        Names of extra kwargs the SMC algorithm's ``step_fn`` requires
        beyond ``(rng_key, state)``.  Default ``()``.  E.g.
        ``("data_mask",)`` for ``partial_posteriors_smc``.
    notes
        Free-form algorithm notes.

    Raises
    ------
    ValueError
        If ``name`` empty; if ``family != "smc"``; if
        ``compatible_inner_methods`` empty; if ``default_inner_method`` not
        in ``compatible_inner_methods``; if ``default_hp_space`` empty.
    """

    name: str
    family: Literal["smc"]
    compatible_inner_methods: tuple[str, ...]
    default_inner_method: str
    default_hp_space: tuple[HyperparamSpace, ...]
    num_particles_default: int = 1000
    step_kwargs_schema: tuple[str, ...] = field(default=())
    notes: str = ""

    _VALID_FAMILIES: ClassVar[frozenset[str]] = frozenset({"smc"})

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SMCMethod: 'name' must be a non-empty string")
        if self.family not in self._VALID_FAMILIES:
            raise ValueError(
                f"SMCMethod '{self.name}': family must be 'smc', got '{self.family}'"
            )
        if not self.compatible_inner_methods:
            raise ValueError(
                f"SMCMethod '{self.name}': 'compatible_inner_methods' must be a non-empty tuple"
            )
        if self.default_inner_method not in self.compatible_inner_methods:
            raise ValueError(
                f"SMCMethod '{self.name}': default_inner_method "
                f"'{self.default_inner_method}' not in compatible_inner_methods "
                f"{self.compatible_inner_methods}"
            )
        if not self.default_hp_space:
            raise ValueError(
                f"SMCMethod '{self.name}': 'default_hp_space' must contain "
                f"at least one HyperparamSpace entry"
            )
