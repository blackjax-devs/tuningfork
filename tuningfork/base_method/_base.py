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
``tuningfork.metrics.grad_counter.total_grad_evals``::

    n_grads = total_grad_evals(infos, entry.grad_count_per_step)

``default_hp_space`` flows into Optuna's distribution constructors.
Each ``HyperparamSpace`` maps 1-to-1 to an Optuna ``suggest_*`` call:
- ``"loguniform"`` → ``trial.suggest_float(name, low, high, log=True)``
- ``"uniform"``    → ``trial.suggest_float(name, low, high)``
- ``"int"``        → ``trial.suggest_int(name, low, high)``
- ``"categorical"`` → ``trial.suggest_categorical(name, choices)``
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from jax import Array

# Sentinel for "not yet populated" — forces explicit assignment on every ENTRY.
# Using a named object instead of None makes missing-population detectable at
# import time (see tests/base_method/test_registry_descriptors.py).
_DESCRIPTOR_REQUIRED: tuple[str, ...] = ()

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
        ``tuningfork.metrics.grad_counter.total_grad_evals`` wraps it with
        ``jax.vmap`` over the full chain info.
    default_hp_space
        Non-empty tuple of ``HyperparamSpace`` objects describing the
        recommended hyperparameter search space for Optuna BO.
    needs_mass_matrix
        When ``True``, the kernel requires an inverse mass matrix (or
        metric tensor) to be wired in.  The BO tuning runner will construct
        and pass one; the exact API is documented in tests.  Default ``False``.
    target_acceptance_rate
        Optimal MH acceptance rate for this kernel, if applicable.  Used
        by window adaptation and reported in tuning results.  ``None``
        for gradient-free kernels (RWM) and MCLMC (no MH step).
    notes
        Free-form string for algorithm-specific caveats, citations, or
        implementation notes.
    extra_required_kwargs
        Names of kwargs the factory requires beyond ``logdensity_fn`` and the
        HP-space items.  Empty tuple (default) = standard factory.  Non-empty =
        specialised: the runner must inject these kwargs from ``Posterior``
        metadata or recipe parameters before calling ``factory(...)``.

        Examples::

            ("prior_cov", "prior_mean")       — Gaussian-prior specialists
                                                (mgrad_gaussian, elliptical_slice)
            ("proposal_distribution",)         — IRMH-family
            ("log_joint_fn", "theta_init")     — Laplace-marginal family

        The standard ``no_warmup`` path raises ``NotImplementedError`` for any
        entry with a non-empty ``extra_required_kwargs``.

    Raises
    ------
    ValueError
        If ``name`` is empty; if ``family`` is not one of the three valid
        values; or if ``default_hp_space`` is empty and
        ``extra_required_kwargs`` is also empty.

    Notes
    -----
    Why no inheritance / ``MCMCEntry(BaseMethod)`` subclass: registry
    consumers (runner, BO loop, CLI) must not branch on type.  All
    algorithm-specific logic (grad cost, HP space) is carried as
    callable/data fields, not subclass overrides.

    The ``needs_mass_matrix`` flag signals to the BO tuning runner that it
    must construct and thread through a mass matrix (e.g. diagonal
    estimate from warmup draws).  This is documented here but
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
    imm_kwarg_name: str = "inverse_mass_matrix"
    """Name of the factory kwarg that receives the adapted mass-matrix-like
    parameter.  Every kernel in the registry accepts ``inverse_mass_matrix``
    EXCEPT ``blackjax.ghmc``, which names it ``momentum_inverse_scale`` (no
    ``inverse_mass_matrix`` parameter at all, no ``**kwargs`` catch-all).

    Single source of truth for that one exception: the generic dispatch
    (``_build_vmapped_inference`` in ``_recipe_runner.py``) reads this field
    instead of hardcoding the kwarg name or special-casing
    ``base_method.name == "ghmc"``, so the translation lives in exactly one
    place rather than being duplicated at every call site that builds a
    factory call (the emit-script generator, ``_emit/_sampler.py``, has its
    own independent translation for the reproduction-script code path).

    ``batched_params`` (the warmup's raw adapted-param dict) is ALSO keyed by
    this name — e.g. MEADS's adapted_params has a ``"momentum_inverse_scale"``
    key, not ``"inverse_mass_matrix"`` — so this same field doubles as the
    dict key to read the per-chain value from, in addition to naming the
    kernel-factory kwarg.

    TODO(descriptor-driven): ``_emit/_sampler.py:258`` (the emit-script /
    reproduction-script generator) still has its own independent
    ``momentum_inverse_scale=inverse_mass_matrix`` translation for ghmc,
    predating this field. Consolidating it to read ``imm_kwarg_name`` too is
    a follow-up, not done in this pass (that code path works and is
    out of scope here).
    """
    target_acceptance_rate: float | None = None
    notes: str = ""
    # ---- grad-count convention string (for headline_basis.grad_count_convention) ----
    # Short formula text describing how grad_count_per_step maps info → gradient count.
    # Example: "info.num_integration_steps" or "(NIS+1) × lbfgs_iter_num (lower bound)".
    # Defaults to empty string; populated in every BaseMethod ENTRY.
    grad_count_convention: str = ""
    # ---- specialised: factory requires extra kwargs beyond logdensity_fn + HP-space ----
    extra_required_kwargs: tuple[str, ...] = ()
    """Names of kwargs the factory requires beyond logdensity_fn + HP-space items.

    Empty tuple = standard factory (logdensity_fn + HP-space kwargs are sufficient).
    Non-empty = specialised: the runner must inject these kwargs from Posterior
    metadata or recipe parameters before calling factory(...).

    Examples:
      ("prior_cov", "prior_mean")        — Gaussian-prior specialists (mgrad_gaussian, elliptical_slice)
      ("proposal_distribution",)         — IRMH-family
      ("log_joint_fn", "theta_init")     — Laplace-marginal family
    """

    # ---- data-driven dispatch descriptors (T2.3) ----
    per_chain_param_keys: tuple[str, ...] = ("step_size", "inverse_mass_matrix")
    """Parameter keys that are per-chain (vmapped) at step time.

    These are the keys extracted from ``batched_params`` and passed
    individually to each chain's factory call inside ``jax.vmap``.

    Typical values:
      ("step_size", "inverse_mass_matrix")  — HMC family, NUTS, MALA, Barker, GHMC, etc.
      ("step_size", "inverse_mass_matrix", "L")  — MCLMC family (L also per-chain from warmup)
      ()  — gradient-free (elliptical_slice, irmh, rwm via no_warmup): no adapted params
    """

    reinit_state: bool = False
    """Whether state must be re-initialised post-warmup via kernel.init().

    True for kernels whose state type differs from the HMCState that
    window_adaptation / mclmc_tuning produces.

    True for:
      dynamic_hmc, dmhmc       — need DynamicHMCState (random_generator_arg)
      ghmc                     — need GHMCState (momentum)
      laplace_hmc, laplace_dhmc, laplace_mhmc, laplace_dmhmc
                               — need LaplaceHMCState (theta_star warm-start)
    False for everything else (HMC, NUTS, MALA, Barker, RWM, MCLMC, etc.).

    Note: adjusted_mclmc_dynamic also needs reinit but only when batched_L
    is not None (emit path).  Its reinit_state=True flag is guarded by the
    ``batched_L is not None`` check in the runner (rerun path falls through to
    the default no-reinit branch via batched_L=None).
    """

    extra_kwarg_builder: Callable[..., dict[str, Any]] | None = None
    """Optional callable that builds extra factory kwargs from runtime context.

    Signature: ``(base_method, logdensity_fn, posterior, batched_params, ...) -> dict``
    where the dict is merged into ``shared_kwargs`` before calling factory(...).

    Used to inject extra kwargs beyond (logdensity_fn, step_size, imm, shared_kwargs)
    that cannot be computed from HP-space alone and depend on runtime context:
      - Laplace family: builds ``log_joint_fn`` + ``theta_init`` from the model
      - mgrad_gaussian / elliptical_slice: builds ``prior_cov`` + ``prior_mean``
      - irmh: builds ``proposal_distribution``
      - additive_step_random_walk: builds ``proposal_generator``

    None = no extra kwargs needed (standard HMC/NUTS/MALA/Barker/RWM/MCLMC/etc.).

    NOTE: ``extra_kwarg_builder`` is for runner-level dispatch only — the actual
    construction of laplace components, prior pytrees, etc., still requires
    model-specific logic in the runner.  The builder receives a ``context`` dict
    with all runner-available state.  See ``_recipe_runner.py`` for the calling
    convention.
    """

    _VALID_FAMILIES: frozenset[str] = frozenset({"mcmc", "vi", "smc"})

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
                f"(specialised factory that receives additional kwargs from the runner)"
            )
