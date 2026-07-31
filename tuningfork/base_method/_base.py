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
"""Base types for the tuningfork algorithm registry.

Every registered sampling method is described by a single ``BaseMethod``
frozen dataclass. Whether the method is a gradient-free random walk (RWM,
0 grads/step) or a sophisticated HMC variant with a full leapfrog integrator,
recipe planning and evaluation see the same compact descriptor surface—no
subclassing.

Grad-count aggregation uses ``entry.grad_count_per_step`` together with
``tuningfork.metrics.grad_counter.total_grad_evals``::

    n_grads = total_grad_evals(infos, entry.grad_count_per_step)

``default_hp_space`` declares deterministic defaults for generated recipe plans:
- ``"loguniform"`` → 70th percentile on the log scale
- ``"uniform"``    → arithmetic midpoint
- ``"int"``        → integer midpoint
- ``"categorical"`` → first declared choice
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from jax import Array

__all__ = ["HyperparamSpace", "BaseMethod"]

# Valid kinds — duplicated as a frozenset for the __post_init__ guard so we
# get a single source of truth (the type annotation covers static checking;
# this covers runtime).
_VALID_KINDS: frozenset[str] = frozenset(
    {"loguniform", "uniform", "int", "categorical"}
)


@dataclass(frozen=True)
class HyperparamSpace:
    """Declared parameter-space descriptor for a single algorithm hyperparameter.

    Parameters
    ----------
    name
        Hyperparameter name, e.g. ``"step_size"`` or ``"num_integration_steps"``.
    kind
        Distribution kind: ``"loguniform"``, ``"uniform"``, ``"int"``,
        or ``"categorical"``.
    low
        Lower bound (inclusive) for numeric kinds. Ignored for
        ``"categorical"``.
    high
        Upper bound (inclusive) for numeric kinds. Ignored for
        ``"categorical"``.
    choices
        Allowed values for ``"categorical"`` kind. Must be a non-empty
        tuple. Ignored for numeric kinds.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the four allowed values; if a numeric
        kind is missing ``low`` or ``high``; or if ``"categorical"`` is
        missing ``choices``.
    """

    name: str
    kind: Literal["loguniform", "uniform", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"HyperparamSpace '{self.name}': kind must be one of "
                f"{sorted(_VALID_KINDS)}, got '{self.kind}'"
            )
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(
                    f"HyperparamSpace '{self.name}': kind='categorical' requires "
                    f"a non-empty 'choices' tuple"
                )
        else:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"HyperparamSpace '{self.name}': kind='{self.kind}' requires "
                    f"both 'low' and 'high' to be set"
                )


@dataclass(frozen=True)
class BaseMethod:
    """Registry entry for a single sampling algorithm.

    Every algorithm in the zoo — from gradient-free RWM to MCLMC with a
    palindromic integrator — exposes this exact surface so recipe planners and
    evaluators can remain algorithm-agnostic.

    Parameters
    ----------
    name
        Unique identifier, e.g. ``"hmc"``, ``"mala"``, ``"mclmc"``.
    family
        Broad algorithm family: ``"mcmc"`` or ``"vi"``.
    grad_count_per_step
        A JAX-compatible callable ``(info) -> Array | int`` that maps a
        *single* per-step info NamedTuple to the number of gradient
        evaluations consumed by that step.  Must be vmappable —
        ``tuningfork.metrics.grad_counter.total_grad_evals`` wraps it with
        ``jax.vmap`` over the full chain info.
    default_hp_space
        Non-empty tuple of ``HyperparamSpace`` objects describing the
        declared parameter space used to resolve generated recipe defaults.
    needs_mass_matrix
        When ``True``, the kernel requires an inverse mass matrix (or
        metric tensor) to be provided by the applicable warmup and sampling
        path. The exact API is documented in tests. Default ``False``.
    target_acceptance_rate
        Optimal MH acceptance rate for this kernel, if applicable.  Used
        by window adaptation and recorded in generated evidence. ``None``
        for gradient-free kernels (RWM) and MCLMC (no MH step).
    notes
        Free-form string for algorithm-specific caveats, citations, or
        implementation notes.
    extra_required_kwargs
        Names of inputs the generated routine requires beyond
        ``logdensity_fn`` and the HP-space items. Empty tuple means the
        standard generated inputs are sufficient. Non-empty means codegen
        must obtain specialised inputs from model metadata or recipe
        parameters.

        Examples::

            ("prior_cov", "prior_mean")       — Gaussian-prior specialists
                                                (mgrad_gaussian, elliptical_slice)
            ("proposal_distribution",)         — IRMH-family
            ("log_joint_fn", "theta_init")     — Laplace-marginal family

        Generated emission must provide these typed inputs; methods without a
        registered emitter fail explicitly rather than inventing them.

    Raises
    ------
    ValueError
        If ``name`` is empty; if ``family`` is not one of the two valid
        values; or if ``default_hp_space`` is empty and
        ``extra_required_kwargs`` is also empty.

    Notes
    -----
    Why no inheritance / ``MCMCEntry(BaseMethod)`` subclass: registry
    consumers must not branch on type. All
    algorithm-specific logic (grad cost, HP space) is carried as
    callable/data fields, not subclass overrides.

    The ``needs_mass_matrix`` flag records that the applicable sampling path
    must thread through a mass matrix (e.g. a diagonal estimate from warmup
    draws).
    """

    # ---- identity ----
    name: str
    family: Literal["mcmc", "vi"]

    # ---- grad-cost oracle ----
    grad_count_per_step: Callable[[Any], Array | int]

    # ---- Declared parameter space ----
    default_hp_space: tuple[HyperparamSpace, ...]

    # ---- optional fields ----
    needs_mass_matrix: bool = False
    target_acceptance_rate: float | None = None
    notes: str = ""
    # ---- grad-count convention string (for headline_basis.grad_count_convention) ----
    # Short formula text describing how grad_count_per_step maps info → gradient count.
    # Example: "info.num_integration_steps" or "(NIS+1) × lbfgs_iter_num (lower bound)".
    # Defaults to empty string; populated in every BaseMethod ENTRY.
    grad_count_convention: str = ""
    # ---- specialised: generated routine requires extra kwargs beyond standard HPs ----
    extra_required_kwargs: tuple[str, ...] = ()
    """Names of kwargs the generated routine requires beyond standard HP items.

    An empty tuple means ``logdensity_fn`` plus the declared hyperparameters
    are sufficient. Specialised methods declare additional model or recipe
    inputs here.

    Examples:
      ("prior_cov", "prior_mean")        — Gaussian-prior specialists (mgrad_gaussian, elliptical_slice)
      ("proposal_distribution",)         — IRMH-family
      ("log_joint_fn", "theta_init")     — Laplace-marginal family
    """

    _VALID_FAMILIES: ClassVar[frozenset[str]] = frozenset({"mcmc", "vi"})

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BaseMethod: 'name' must be a non-empty string")
        if self.family not in self._VALID_FAMILIES:
            raise ValueError(
                f"BaseMethod '{self.name}': family must be one of "
                f"{sorted(self._VALID_FAMILIES)}, got '{self.family}'"
            )
        if not self.default_hp_space and not self.extra_required_kwargs:
            raise ValueError(
                f"BaseMethod '{self.name}': 'default_hp_space' must contain at least "
                f"one HyperparamSpace entry, or 'extra_required_kwargs' must be non-empty "
                f"(specialised generated route with additional model inputs)"
            )
