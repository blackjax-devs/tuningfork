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
"""Base types for the bjx-bench algorithm registry.

Every sampling algorithm exposed to the benchmark is described by a single
``BaseMethod`` frozen dataclass.  Whether the sampler is a gradient-free
random-walk (RWM, 0 grads/step) or a sophisticated HMC variant with a full
leapfrog integrator, the runner, Optuna tuning loop, and CLI always see the
same surface — no subclassing.

Dispatch model
--------------
The runner uses ``entry.factory`` to instantiate the BlackJAX kernel::

    kernel = entry.factory(logdensity_fn, **trial_params)
    init_state = kernel.init(position, rng_key)
    final_state, info = kernel.step(rng_key, init_state)

Grad-count aggregation uses ``entry.grad_count_per_step`` together with
``bjx_bench.metrics.grad_counter.total_grad_evals``::

    n_grads = total_grad_evals(infos, entry.grad_count_per_step)

``default_hp_space`` flows into Optuna's distribution constructors at T2.6.
Each ``HyperparamSpace`` maps 1-to-1 to an Optuna ``suggest_*`` call:
- ``"loguniform"`` → ``trial.suggest_float(name, low, high, log=True)``
- ``"uniform"``    → ``trial.suggest_float(name, low, high)``
- ``"int"``        → ``trial.suggest_int(name, low, high)``
- ``"categorical"`` → ``trial.suggest_categorical(name, choices)``
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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
    """Search space descriptor for a single algorithm hyperparameter.

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
    palindromic integrator — exposes this exact surface so the runner and
    the Optuna BO loop are algorithm-agnostic.

    Parameters
    ----------
    name
        Unique identifier, e.g. ``"hmc"``, ``"mala"``, ``"mclmc"``.
    family
        Broad algorithm family: ``"mcmc"``, ``"vi"``, or ``"smc"``.
    factory
        A callable that accepts ``(logdensity_fn, **hyperparams)`` and
        returns a BlackJAX kernel object (with ``.init`` and ``.step``
        methods).  Typically ``blackjax.hmc``, ``blackjax.mala``, etc.
        Wrappers in the per-algorithm modules adapt non-uniform BlackJAX
        signatures to this common interface.
    grad_count_per_step
        A JAX-compatible callable ``(info) -> Array | int`` that maps a
        *single* per-step info NamedTuple to the number of gradient
        evaluations consumed by that step.  Must be vmappable —
        ``bjx_bench.metrics.grad_counter.total_grad_evals`` wraps it with
        ``jax.vmap`` over the full chain info.
    default_hp_space
        Non-empty tuple of ``HyperparamSpace`` objects describing the
        recommended hyperparameter search space for Optuna BO (T2.6).
    needs_mass_matrix
        When ``True``, the kernel requires an inverse mass matrix (or
        metric tensor) to be wired in.  The Tier-B runner will construct
        and pass one; T2.2 documents the exact API.  Default ``False``.
    target_acceptance_rate
        Optimal MH acceptance rate for this kernel, if applicable.  Used
        by window adaptation and reported in tuning results.  ``None``
        for gradient-free kernels (RWM) and MCLMC (no MH step).
    notes
        Free-form string for algorithm-specific caveats, citations, or
        implementation notes.

    Raises
    ------
    ValueError
        If ``name`` is empty; if ``family`` is not one of the three
        valid values; or if ``default_hp_space`` is empty.

    Notes
    -----
    Why no inheritance / ``MCMCEntry(BaseMethod)`` subclass: registry
    consumers (runner, BO loop, CLI) must not branch on type.  All
    algorithm-specific logic (grad cost, HP space) is carried as
    callable/data fields, not subclass overrides.

    The ``needs_mass_matrix`` flag signals to the Tier-B runner that it
    must construct and thread through a mass matrix (e.g. diagonal
    estimate from warmup draws).  This is documented here for T2.2 but
    the runner wiring happens there.
    """

    # ---- identity ----
    name: str
    family: Literal["mcmc", "vi", "smc"]

    # ---- BlackJAX kernel factory ----
    factory: Callable[..., Any]

    # ---- grad-cost oracle ----
    grad_count_per_step: Callable[[Any], Array | int]

    # ---- Optuna search space ----
    default_hp_space: tuple[HyperparamSpace, ...]

    # ---- optional fields ----
    needs_mass_matrix: bool = False
    target_acceptance_rate: float | None = None
    notes: str = ""

    _VALID_FAMILIES: frozenset[str] = frozenset({"mcmc", "vi", "smc"})

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BaseMethod: 'name' must be a non-empty string")
        if self.family not in self._VALID_FAMILIES:
            raise ValueError(
                f"BaseMethod '{self.name}': family must be one of "
                f"{sorted(self._VALID_FAMILIES)}, got '{self.family}'"
            )
        if not self.default_hp_space:
            raise ValueError(
                f"BaseMethod '{self.name}': 'default_hp_space' must contain "
                f"at least one HyperparamSpace entry"
            )
