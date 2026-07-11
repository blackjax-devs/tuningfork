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
"""CHEES (Change in Estimator of Step Size) warmup, wrapping
``blackjax.chees_adaptation``.

CHEES (Hoffman et al. 2022) is a **dynamic-HMC-specific** adaptation routine
that simultaneously tunes ``step_size``, ``inverse_mass_matrix``, and the
trajectory-length distribution (via ``integration_steps_fn``,
``next_random_arg_fn``, and ``integration_steps_params``).  Like MEADS, CHEES
is **multi-chain by construction**: all chains run jointly in a single
adaptation call — chains are inputs, not loop iterations.

CHEES adapts both step_size AND the trajectory-length distribution.  Like
MEADS but for dynamic-HMC instead of GHMC; both are multi-chain-by-construction
adaptation procedures.

Compatibility
-------------
CHEES is paired exclusively with ``dynamic_hmc``
(``compatible_methods=("dynamic_hmc",)``).  Passing any other base method
raises ``ValueError``.

Upstream API note
-----------------
``blackjax.chees_adaptation.run`` has a different signature from
``meads_adaptation.run``.  It requires ``step_size`` and ``optim`` (an optax
optimizer) as additional positional arguments::

    chees.run(rng_key, positions, step_size, optim, num_steps=n_warmup)

This wrapper builds an ``optax.adam(learning_rate=0.05, b1=0, b2=0.95)``
optimizer for the trajectory-length parameters and accepts ``step_size`` as a
pass-through kwarg (default ``0.5``).  The optimizer form is the canonical
CHEES/SNAPER one (Adam with ``b1=0, b2=0.95``: no first-moment averaging, so an
RMSProp-style scaled log-trajectory-length step — the same form TFP's
``GradientBasedTrajectoryLengthAdaptation`` ships).  TFP's shipped
``adaptation_rate`` is ``0.025`` ("ChEES"); Sountsov & Hoffman (SNAPER-HMC,
arXiv:2110.11576, App. D) label ``0.025`` "(ChEES)" and ``0.05`` "ChEES fast".
The ``0.05`` default here was CALIBRATED for tuningfork's ``n_warmup=2000`` cert
budget — the TFP-canonical ``0.025`` leaves the trajectory length
under-converged on stiff scale-separated targets (see issue #217 and the
``_DEFAULT_CHEES_OPTIM_LR`` note below).  ``optim_learning_rate`` stays exposed
for callers that want a different rate.

Adapted parameters returned by upstream
----------------------------------------
``chees_adaptation.run`` returns a 2-tuple ``(AdaptationResults, AdaptationInfo)``
where ``AdaptationResults.parameters`` is a dict with keys:

- ``step_size`` — scalar (shared across chains)
- ``inverse_mass_matrix`` — shape ``(d,)`` (shared across chains)
- ``next_random_arg_fn`` — Python callable (shared; not array-broadcastable)
- ``integration_steps_fn`` — Python callable (shared; not array-broadcastable)
- ``integration_steps_params`` — shape ``(1,)`` array

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, num_chains: int = 4,
            step_size: float = 0.5, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is passed directly to ``chees_adaptation.run``.
- ``init_position`` is a single pytree (one chain's worth); replicated to
  ``(num_chains, d)`` via ``_maybe_replicate`` unless pre-batched.
- ``states`` is a batched ``DynamicHMCState`` pytree with leading dim ``num_chains``.
- ``adapted_params`` contains:

  ========================= =============================================
  Key                       Shape / Type
  ========================= =============================================
  ``step_size``             ``(num_chains,)`` — scalar broadcast
  ``inverse_mass_matrix``   ``(num_chains, d)`` — broadcast from ``(d,)``
  ``next_random_arg_fn``    Python callable (passed through; shared)
  ``integration_steps_fn``  Python callable (passed through; shared)
  ``integration_steps_params`` ``(1,)`` array (passed through; shared)
  ``_chees_target_acceptance_rate`` float — value used
  ``_chees_max_leapfrog_steps``     int — cap used
  ``_chees_optim_learning_rate``    float — trajectory-optimizer LR used
  ``_chees_optim_b1``               float — Adam b1 used (0.0, canonical)
  ``_chees_optim_b2``               float — Adam b2 used (0.95, canonical)
  ========================= =============================================

References
----------
- Hoffman, M. D., Radul, A., & Sountsov, P. (2022). An adaptive-MCMC scheme
  for setting trajectory lengths in Hamiltonian Monte Carlo. In *AISTATS 2022*.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import optax

from tuningfork.warmup._base import Warmup, _maybe_replicate

__all__ = ["ENTRY"]

# Default CHEES hyperparameters (matching upstream defaults)
_DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE: float = 0.651
_DEFAULT_CHEES_MAX_LEAPFROG_STEPS: int = 1000

# Trajectory-length optimizer (Adam on log trajectory length).  The canonical
# CHEES/SNAPER form is Adam(beta1=0, beta2=0.95) — a plain RMSProp-like scaled
# gradient step with NO first-moment (momentum) averaging.  TFP's shipped
# ``GradientBasedTrajectoryLengthAdaptation`` uses ``adaptation_rate=0.025``
# ("ChEES") with these betas; Sountsov & Hoffman (SNAPER-HMC, arXiv:2110.11576,
# App. D) label 0.025 "(ChEES)" and 0.05 "ChEES fast".  The rate here (0.05) was
# CALIBRATED for tuningfork's cert budget: at ``n_warmup=2000`` on stiff
# scale-separated targets with dispersed inits (radon d=390, irt_2pl d=144,
# nc=128), the TFP-canonical 0.025 leaves the trajectory length under-converged
# (L stuck ~50, R̂≈3.6), while 0.05 lets L converge (radon×uniform L=171,
# R̂=1.004, 0 div) — the smallest paper-grounded rate that certifies robustly.
# Bigger is not better: 0.1/0.5 overshoot and destabilise (irt z-spikes, longer
# trajectories).  See tuningfork issue #217 + the 2026-07-11 GPU A/B calibration.
_DEFAULT_CHEES_OPTIM_LR: float = 0.05
_DEFAULT_CHEES_OPTIM_B1: float = 0.0
_DEFAULT_CHEES_OPTIM_B2: float = 0.95
_DEFAULT_CHEES_STEP_SIZE: float = 0.5

# Salt for deriving the init-jitter key from the caller's rng_key via fold_in,
# keeping it independent of whatever sub-keys chees_adaptation.run() itself
# splits off the SAME rng_key it is handed unchanged (see _runner below).
_INIT_JITTER_SALT = 0xC4EE5717
_DEFAULT_INIT_JITTER_SCALE: float = 0.5


def _replicate_with_init_jitter(
    position: Any, num_chains: int, rng_key: jax.Array, jitter_scale: float
) -> Any:
    """Replicate ``position`` to ``(num_chains, ...)``, jittering a bit-identical broadcast.

    Defensive symmetry with ``warmup/meads.py``'s helper of the same name:
    MEADS's first adaptation iteration provably divides by 0 on a
    bit-identical cross-chain broadcast (see meads.py docstring).  CHEES does
    not have that specific 0/0 pathology (each chain draws independent HMC
    momentum from step 1, so chains diverge immediately even from identical
    starts), but ensemble-adaptation warmups are exactly the class of routine
    where a silent identical-chains degeneracy is easy to introduce upstream
    without tuningfork noticing (no chain-count study has ever exercised
    this warmup end-to-end).  Jittering broadcast inits here costs nothing
    and keeps both ensemble-warmup wrappers behaviourally consistent.

    Only jitters when the caller did NOT pre-batch: if ``position`` already
    has a leading dim of ``num_chains`` (per ``_maybe_replicate``'s own
    detection), it is trusted to already carry meaningful per-chain
    dispersion and is passed through via ``_maybe_replicate`` unchanged.
    """
    leaves = jax.tree.leaves(position)
    is_pre_batched = (
        bool(leaves) and bool(leaves[0].shape) and (leaves[0].shape[0] == num_chains)
    )
    replicated = _maybe_replicate(position, num_chains)
    if is_pre_batched:
        return replicated

    rep_leaves, rep_treedef = jax.tree.flatten(replicated)
    leaf_keys = jax.random.split(rng_key, len(rep_leaves))
    jittered_leaves = [
        leaf + jitter_scale * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for leaf, k in zip(rep_leaves, leaf_keys)
    ]
    return jax.tree.unflatten(rep_treedef, jittered_leaves)


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    num_chains: int = 4,
    step_size: float = _DEFAULT_CHEES_STEP_SIZE,
    target_acceptance_rate: float | None = _DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE,
    max_leapfrog_steps: int = _DEFAULT_CHEES_MAX_LEAPFROG_STEPS,
    optim_learning_rate: float = _DEFAULT_CHEES_OPTIM_LR,
    init_jitter_scale: float = _DEFAULT_INIT_JITTER_SCALE,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``blackjax.chees_adaptation`` over ``num_chains`` chains jointly.

    Unlike ``window_adaptation_diag_imm`` which vmaps per-chain window adaptation independently,
    CHEES runs a **single** multi-chain adaptation call.  All chains participate
    together to adapt step size and trajectory length.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Passed directly to
        ``chees_adaptation.run``; split internally by CHEES.
    init_position
        Initial unconstrained parameter dict/array (one chain's worth).
        Replicated across ``num_chains`` via ``_maybe_replicate`` unless the
        caller pre-batches with a leading dim of ``num_chains``.
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (must be ``dynamic_hmc``; verified by compatibility
        guard).
    logdensity_fn
        BlackJAX-compatible log-density function.
    num_chains
        Number of chains for the joint CHEES adaptation.  Default ``4``.
    step_size
        Initial step size passed to CHEES.  CHEES will adapt it.  Default
        ``0.5`` (a reasonable starting point for most posteriors).
    target_acceptance_rate
        CHEES target acceptance rate.  Default ``0.651`` (upstream default).
    max_leapfrog_steps
        Maximum leapfrog steps per trajectory.  Default ``1000`` (upstream
        default).
    optim_learning_rate
        Learning rate for the optax Adam optimizer used to adapt trajectory
        length parameters.  Default ``0.05`` — SNAPER App. D "ChEES fast"; the
        smallest paper-grounded rate that certifies robustly at ``n_warmup=2000``
        (TFP-canonical ``0.025`` under-converges L at this budget on stiff
        targets — calibrated 2026-07-11, issue #217).  The optimizer betas are
        fixed to the canonical ``b1=0, b2=0.95``.
    **kwargs
        Additional keyword arguments (ignored; kept for API uniformity).

    Returns
    -------
    states
        Post-warmup ``DynamicHMCState`` pytree with leading dim ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with keys:

        - ``"step_size"``: shape ``(num_chains,)`` (scalar broadcast from
          CHEES shared estimate).
        - ``"inverse_mass_matrix"``: shape ``(num_chains, d)`` (broadcast from
          ``(d,)`` CHEES estimate).
        - ``"next_random_arg_fn"``: Python callable (CHEES-adapted; shared
          across chains; not broadcastable — passed through as-is).
        - ``"integration_steps_fn"``: Python callable (CHEES-adapted; shared
          across chains; not broadcastable — passed through as-is).
        - ``"integration_steps_params"``: shape ``(1,)`` array (CHEES-adapted
          trajectory length param; shared across chains; passed through as-is).
        - ``"_chees_target_acceptance_rate"``: Python float — value used.
        - ``"_chees_max_leapfrog_steps"``: Python int — cap used.
        - ``"_chees_optim_learning_rate"``: Python float — trajectory-optimizer
          learning rate used (default ``0.05``; issue #217 calibration).
        - ``"_chees_optim_b1"`` / ``"_chees_optim_b2"``: Python floats — Adam
          betas used (``0.0`` / ``0.95``, the canonical CHEES/SNAPER form).

    Raises
    ------
    ValueError
        If ``base_method.name != "dynamic_hmc"`` (incompatibility guard).

    Notes
    -----
    CHEES upstream signature is different from MEADS: ``chees_adaptation.run``
    requires ``step_size`` and ``optim`` (optax optimizer) as positional
    arguments after ``positions``.  This wrapper creates the optimizer
    internally as ``optax.adam(optim_learning_rate, b1=0, b2=0.95)`` — the
    canonical CHEES/SNAPER form (no first-moment averaging).  The callable
    params (``next_random_arg_fn``, ``integration_steps_fn``) are Python
    functions returned by CHEES and are passed through to the downstream kernel
    factory unchanged.
    """
    # The generic recipe-runner dispatch (_recipe_runner.py) always forwards
    # target_acceptance_rate explicitly, including None when the caller has no
    # override — the emit default is None, not "omit the kwarg".  A plain typed
    # default only helps direct callers; it does nothing once None is passed in.
    # Fall back to the CHEES default the same way _window_adaptation_common.py:92
    # falls back for window_adaptation, so a None reaching this wrapper never
    # propagates to upstream chees_adaptation.py, where
    # `target_acceptance_rate - harmonic_mean` would TypeError on None.
    _target_acceptance_rate = (
        target_acceptance_rate or _DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE
    )

    # Build CHEES adaptation object
    chees = blackjax.chees_adaptation(
        logdensity_fn,
        num_chains,
        target_acceptance_rate=_target_acceptance_rate,
        max_leapfrog_steps=max_leapfrog_steps,
    )

    # Replicate init_position to (num_chains, *leaf.shape); pass-through if pre-batched.
    # A broadcast (non-pre-batched) init is additionally jittered per chain — see
    # _replicate_with_init_jitter docstring (defensive symmetry with meads.py).
    _jitter_key = jax.random.fold_in(rng_key, _INIT_JITTER_SALT)
    init_positions = _replicate_with_init_jitter(
        init_position, num_chains, _jitter_key, init_jitter_scale
    )

    # Build optax optimizer for trajectory-length adaptation.  Canonical
    # CHEES/SNAPER form: Adam(b1=0, b2=0.95) — no first-moment averaging, so it
    # behaves as an RMSProp-style scaled log-trajectory-length step (matches
    # TFP's GradientBasedTrajectoryLengthAdaptation).  Plain optax.adam defaults
    # (b1=0.9) inject momentum the CHEES update was never designed for and were
    # observed to make the adapted L oscillate — see issue #217.
    optim = optax.adam(
        learning_rate=optim_learning_rate,
        b1=_DEFAULT_CHEES_OPTIM_B1,
        b2=_DEFAULT_CHEES_OPTIM_B2,
    )

    # Run CHEES: single call handles all num_chains chains jointly.
    # Returns (AdaptationResults, AdaptationInfo).
    # AdaptationResults.state: DynamicHMCState with leading dim num_chains.
    # AdaptationResults.parameters: dict with:
    #   step_size (scalar), inverse_mass_matrix (d,),
    #   next_random_arg_fn (callable), integration_steps_fn (callable),
    #   integration_steps_params (1,).
    (adaptation_results, _adaptation_info) = chees.run(
        rng_key, init_positions, step_size, optim, num_steps=n_warmup
    )

    states = adaptation_results.state
    raw_params = adaptation_results.parameters

    # Extract adapted scalar params and broadcast to (num_chains,) / (num_chains, d).
    # CHEES returns a single shared estimate:
    #   step_size → scalar ()
    #   inverse_mass_matrix → shape (d,)
    # Callable params (next_random_arg_fn, integration_steps_fn) cannot be
    # broadcast — they are passed through as Python functions (shared).
    step_size_scalar = jnp.asarray(raw_params["step_size"])
    raw_imm = raw_params["inverse_mass_matrix"]

    # Flatten the inverse_mass_matrix pytree to a single (d,) array.
    # For plain arrays this is a no-op.
    imm_leaves = jax.tree.leaves(raw_imm)
    if len(imm_leaves) == 1:
        imm_flat = imm_leaves[0]  # shape (d,)
    else:
        imm_flat = jnp.concatenate(
            [leaf.reshape(-1) for leaf in imm_leaves]
        )  # shape (d,)

    adapted_params: dict[str, Any] = {
        # Numeric params: broadcast to (num_chains,) / (num_chains, d)
        "step_size": jnp.broadcast_to(step_size_scalar, (num_chains,)),
        "inverse_mass_matrix": jnp.broadcast_to(
            imm_flat[None, :], (num_chains, imm_flat.shape[0])
        ),
        # Callable / non-broadcastable params: pass through as-is (shared across chains)
        "next_random_arg_fn": raw_params["next_random_arg_fn"],
        "integration_steps_fn": raw_params["integration_steps_fn"],
        "integration_steps_params": raw_params["integration_steps_params"],
        # Sidecar metadata
        "_chees_target_acceptance_rate": float(_target_acceptance_rate),
        "_chees_max_leapfrog_steps": int(max_leapfrog_steps),
        # Trajectory-length optimizer config used (provenance; underscore-prefixed
        # so it never reaches a kernel factory — see _recipe_runner filtering).
        "_chees_optim_learning_rate": float(optim_learning_rate),
        "_chees_optim_b1": float(_DEFAULT_CHEES_OPTIM_B1),
        "_chees_optim_b2": float(_DEFAULT_CHEES_OPTIM_B2),
    }

    return states, adapted_params


ENTRY = Warmup(
    name="chees",
    runner=_runner,
    compatible_methods=("dynamic_hmc",),
    notes=(
        "CHEES (Change in Estimator of Step Size) adaptation for dynamic-HMC. "
        "Adapts both step_size AND the trajectory-length distribution (integration_steps_fn, "
        "next_random_arg_fn, integration_steps_params).  Like MEADS but for dynamic-HMC "
        "instead of GHMC; both are multi-chain-by-construction adaptation procedures. "
        "Upstream API note: chees_adaptation.run() requires step_size and an optax optimizer "
        "as positional args (unlike meads_adaptation.run); this wrapper provides "
        "optax.adam(0.05, b1=0, b2=0.95) internally (canonical CHEES/SNAPER form; LR calibrated "
        "for n_warmup=2000, issue #217).  Callable params (next_random_arg_fn, integration_steps_fn) are Python "
        "functions returned by CHEES and passed through unchanged — they cannot be broadcast "
        "to (num_chains,) shape.  numeric params (step_size, inverse_mass_matrix) are "
        "broadcast from the shared CHEES estimate to (num_chains,) / (num_chains, d). "
        "dynamic_hmc-specific; not compatible with HMC, NUTS, GHMC, or any other kernel. "
        "multi-chain by default (num_chains=4); target_acceptance_rate=0.651 (CHEES default)."
    ),
)
