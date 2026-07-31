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
"""Mean-field VI warmup: init-and-IMM provider via variational surrogate.

This warmup runs a single mean-field VI optimisation (shared across all
chains), then:

1. Draws ``num_chains`` initial positions from the fitted variational
   distribution ``N(mu, diag(exp(rho))^2)`` — each chain gets an independent
   draw.
2. Returns the **diagonal inverse mass matrix** (IMM) as
   ``exp(2 * rho)`` — the per-coordinate variance of the fitted variational
   distribution, broadcast identically across all chains (the VI fit is
   shared; only init positions differ).

No step-size adaptation is performed.  A scalar default of ``1.0`` is
returned for every chain.  The downstream sampler should rely on its own
dual-averaging adaptation (e.g. NUTS window adaptation) or Bayesian
adaptation to resolve the step size; this warmup provides a better
initialisation and diagonal IMM than a flat prior sample.

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, step_size_default=1.0,
            num_chains: int = 4,
            num_optimization_steps: int = 10_000,
            optimizer=optax.adam(1e-2),
            num_samples_per_step: int = 5,
            **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate``.
- ``states`` is a batched pytree with leading dim ``num_chains``;
  ``states.position`` has shape ``(num_chains, d)``.
- ``adapted_params`` contains:

  =============================================  =================  ==========================================
  Key                                            Shape              Notes
  =============================================  =================  ==========================================
  ``step_size``                                  ``(num_chains,)``  Constant ``step_size_default`` per chain
  ``inverse_mass_matrix``                        ``(num_chains, d)``  Diagonal per-coord variance ``exp(2*rho)``
  ``_mfvi_elbo``                                 scalar             Final ELBO from the VI fit (sidecar)
  =============================================  =================  ==========================================

Sidecar keys (underscore prefix) are metadata for ``calibration_budget``
and are NOT forwarded to the base-method kernel as hyperparameters.

Compatible with: ``nuts``, ``hmc``, ``mala``, ``rwm``, ``barker``.
NOT compatible with ``mclmc`` (different geometry — microcanonical momentum,
no Gaussian inverse mass matrix in the HMC sense).

Upstream API (BlackJAX >= 0.9.x):

- ``blackjax.vi.meanfield_vi.init(position, optimizer)``
  → ``MFVIState(mu, rho, opt_state)``
- ``blackjax.vi.meanfield_vi.step(rng_key, state, logdensity_fn, optimizer, num_samples)``
  → ``(MFVIState, MFVIInfo(elbo=...))``
- ``blackjax.vi.meanfield_vi.sample(rng_key, state, num_samples)``
  → PyTree with leading dim ``num_samples``
- ``MFVIState._fields``: ``('mu', 'rho', 'opt_state')``
  where ``rho`` encodes log-scale: ``sigma = exp(rho)``,
  ``variance = exp(2 * rho)``.
"""

from typing import Any

import blackjax.vi.meanfield_vi as mf
import jax
import jax.numpy as jnp

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.warmup._base import Warmup
from tuningfork.warmup._vi_warmup_runner import _vi_warmup_runner

__all__ = ["ENTRY"]

# Algorithms that accept an inverse_mass_matrix and are therefore compatible
# with mean-field VI's init-and-IMM output.  mclmc is excluded: its geometry
# is microcanonical and its inverse_mass_matrix is a diagonal preconditioner
# for Euclidean distance, not an HMC-style covariance.
_COMPATIBLE = ("nuts", "hmc", "mala", "rwm", "barker")

# Production default for num_optimization_steps.
_DEFAULT_N_OPT_STEPS = 10_000


def _mfvi_imm_extractor(final_vi_state: mf.MFVIState, d: int) -> tuple[Any, tuple]:
    """Extract diagonal IMM from mean-field VI state.

    Returns ``(diag_imm, broadcast_shape)`` where
    ``diag_imm = exp(2 * rho_flat)`` has shape ``(d,)`` and
    ``broadcast_shape = (d,)``.
    """
    rho_flat, _ = jax.flatten_util.ravel_pytree(final_vi_state.rho)
    diag_imm = jnp.exp(2.0 * rho_flat)  # shape (d,)
    return diag_imm, (d,)


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,  # step_size dual-averaging budget (0 = skip, use step_size_default)
    base_method: Any,  # BaseMethod; not imported to avoid circular dep
    *,
    logdensity_fn: Any,
    step_size_default: float = 1.0,
    num_chains: int = 4,
    num_optimization_steps: int = _DEFAULT_N_OPT_STEPS,
    optimizer: Any = None,
    num_samples_per_step: int = 5,
    target_acceptance_rate: float = 0.8,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run mean-field VI once (shared fit); draw ``num_chains`` init positions.

    The VI optimisation is performed once and the resulting variational
    distribution is shared across all chains.  ``num_chains`` independent
    initial positions are drawn from the fitted distribution.  The diagonal
    of the per-coordinate variance (``exp(2 * rho)``) is returned as the
    ``inverse_mass_matrix``, broadcast identically across all chains.

    Parameters
    ----------
    rng_key
        JAX random key.  Split internally into a key for the VI loop and
        a key for drawing ``num_chains`` initial positions.
    init_position
        Initial unconstrained parameter pytree (one chain's worth).  The
        runner uses this as the starting point for VI optimisation, and
        replicates it to ``(num_chains, ...)`` for state initialisation
        unless the caller pre-batches it (leading dim == ``num_chains``).
    n_warmup
        Accepted for interface uniformity; not used.  The VI optimisation
        budget is controlled by ``num_optimization_steps``.
    base_method
        ``BaseMethod`` entry.  Used for kernel-state initialisation after
        the VI fit.
    logdensity_fn
        BlackJAX-compatible log-density function.
    step_size_default
        Constant step size assigned to every chain.  Default ``1.0``.
        The downstream sampler should adapt this via dual-averaging.
    num_chains
        Number of independent chains to initialise.  Default ``4``.
        The returned ``states`` have leading dim ``num_chains``.
    num_optimization_steps
        Number of Adam optimisation steps for the VI loop.
        Default ``10_000`` (production); use ``2_000`` in tests.
    optimizer
        Optax ``GradientTransformation``.  Defaults to ``optax.adam(1e-2)``.
    num_samples_per_step
        Number of Monte Carlo samples per VI gradient step.  Default ``5``
        (upstream default).
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    states
        Post-VI kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)`` (or
        ``(num_chains, ...)`` for dict/pytree positions).  Each chain's
        position is an independent draw from the fitted variational
        distribution.
    adapted_params
        Dict with:

        - ``"step_size"``: ``(num_chains,)`` array, constant ``step_size_default``.
        - ``"inverse_mass_matrix"``: ``(num_chains, d)``, per-chain copy of
          the diagonal variance ``exp(2 * rho)`` (identical across chains —
          the VI fit is shared).
        - ``"_mfvi_elbo"``: scalar, final ELBO from the VI optimisation
          (sidecar metadata; underscore prefix marks it as non-HP).

    Raises
    ------
    ValueError
        If ``base_method.name`` is not in the compatible set.
    """
    if base_method.name not in _COMPATIBLE:
        raise ValueError(
            f"meanfield_vi warmup is not compatible with base_method "
            f"{base_method.name!r}; compatible: {_COMPATIBLE}"
        )

    return _vi_warmup_runner(
        rng_key,
        init_position,
        n_warmup,
        base_method,
        vi_module=mf,
        imm_extractor_fn=_mfvi_imm_extractor,
        elbo_sidecar_key="_mfvi_elbo",
        default_n_opt_steps=_DEFAULT_N_OPT_STEPS,
        logdensity_fn=logdensity_fn,
        step_size_default=step_size_default,
        num_chains=num_chains,
        num_optimization_steps=num_optimization_steps,
        optimizer=optimizer,
        num_samples_per_step=num_samples_per_step,
        target_acceptance_rate=target_acceptance_rate,
        **kwargs,
    )


ENTRY = Warmup(
    name="meanfield_vi",
    runner=_runner,
    compatible_methods=_COMPATIBLE,
    default_hp_space=(
        # num_optimization_steps: recipe-resolved VI optimisation budget.
        # Mirror the base_method VI range for consistency (1k–50k).
        # Default value (midpoint = 25_500) is close to the production 10_000 default
        # when the recipe runner uses default_value_for_space; override by passing
        # num_optimization_steps directly via warmup_kwargs_override in emit.
        HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
    ),
    notes=(
        "Mean-field VI warmup: runs a single mean-field VI "
        "optimisation (shared across all chains) via jax.lax.scan over "
        "num_optimization_steps Adam steps.  Draws num_chains independent "
        "initial positions from the fitted variational distribution and "
        "returns the diagonal variance exp(2*rho) as the "
        "inverse_mass_matrix (shared across all chains — the fit is "
        "shared, only init positions differ).  "
        "step_size adaptation: when n_warmup > 0, runs n_warmup steps of "
        "Nesterov dual averaging with the VI IMM frozen to find the adapted "
        "step_size; when n_warmup == 0, returns step_size_default (1.0).  "
        "Sidecar: _mfvi_elbo (final ELBO scalar).  "
        "Compatible: nuts, hmc, mala, rwm, barker.  "
        "NOT compatible with mclmc (microcanonical geometry).  "
        "Production default: num_optimization_steps=10_000, "
        "optimizer=optax.adam(1e-2).  Use 2_000 in tests."
    ),
)
