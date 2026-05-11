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
"""Full-rank VI sampler-mode base method entry for tuningfork.

Wraps ``blackjax.vi.fullrank_vi`` as a dual-mode entry compatible with the
tuningfork runner contract.  The "sampler-mode" here means the full VI
optimisation loop runs inside ``init`` (via ``jax.lax.scan``), storing the
final variational state in a wrapper ``FRVISamplerState``.  Each subsequent
``step(rng_key, state)`` call draws **one** sample from the fitted full-rank
Gaussian variational distribution, returning it as the new ``position``.

This design makes the wrapper consistent with the sampler-mode contract
(``init → step → step → ... → samples``) used by every other ``BaseMethod``
in the registry, even though VI is fundamentally an optimisation algorithm.

Hyperparameter space: **empty** ``()`` — ``num_optimization_steps`` and
``optimizer`` are recipe-time constants, not Bayesian-optimisation tunable
parameters at the trial level.

Grad cost approximation: ``grad_count_per_step = lambda info: 1``.  Each VI
optimisation step requires one gradient of the log-density (via the ELBO
gradient); after the loop, each ``step`` call draws one sample.  The
approximation is intentionally simple — see ``notes`` for the full accounting.

The factory default uses ``optax.adam(1e-2)`` and
``num_optimization_steps=20000`` (production default, higher than meanfield
to allow the richer covariance parameterisation to converge).  Test suites
should override with ``num_optimization_steps=5000``.

**Applicability**: full-rank VI is recommended only for ``d <= 30``.  For
higher-dimensional problems, use ``meanfield_vi`` instead.  The cholesky
parameterisation has ``O(d^2)`` parameters which become expensive at large
dimension.
"""

from typing import Any, NamedTuple

import blackjax.vi.fullrank_vi as fr
import jax
import jax.numpy as jnp
import optax

from tuningfork.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY", "_factory", "FRVISamplerState"]

# Default optimizer (recipe-time; can be overridden via factory kwarg).
_adam_default = optax.adam(1e-2)


class FRVISamplerState(NamedTuple):
    """Wrapper state for full-rank VI in sampler mode.

    Parameters
    ----------
    position
        Current draw from the fitted variational distribution.  Shape matches
        the model's parameter pytree.
    vi_state
        Final ``FRVIState`` from the VI optimisation loop.  Stored so that
        subsequent ``step`` calls can draw new samples from the same fit.
    """

    position: Any
    vi_state: fr.FRVIState


class _FRVISamplerAlgorithm:
    """Thin wrapper that presents the full-rank VI optimiser as a sampler kernel.

    Attributes
    ----------
    init
        Callable ``(init_position) -> FRVISamplerState``.  Runs the full VI
        optimisation loop via ``jax.lax.scan`` and stores the final
        ``FRVIState``.  The ``position`` field of the returned state is set to
        the variational mean ``mu``.
    step
        Callable ``(rng_key, state) -> (FRVISamplerState, FRVIInfo)``.  Draws
        one sample from the fitted variational distribution and returns it as
        the new ``position``.
    """

    def __init__(
        self,
        logdensity_fn: Any,
        optimizer: Any,
        num_optimization_steps: int,
        num_samples_per_step: int,
    ) -> None:
        self._logdensity_fn = logdensity_fn
        self._optimizer = optimizer
        self._num_optimization_steps = num_optimization_steps
        self._num_samples_per_step = num_samples_per_step

    def init(self, init_position: Any) -> FRVISamplerState:
        """Run the full VI optimisation loop; return the final state.

        Parameters
        ----------
        init_position
            Initial unconstrained parameter pytree (one chain's worth).

        Returns
        -------
        FRVISamplerState
            Wrapper state with ``position = mu`` (variational mean) and the
            final ``vi_state`` from the optimisation loop.
        """
        vi_init = fr.init(init_position, self._optimizer)

        def one_step(carry: fr.FRVIState, rng_key: jax.Array):
            new_state, info = fr.step(
                rng_key,
                carry,
                self._logdensity_fn,
                self._optimizer,
                self._num_samples_per_step,
            )
            return new_state, info

        keys = jax.random.split(jax.random.key(0), self._num_optimization_steps)
        final_vi_state, _infos = jax.lax.scan(one_step, vi_init, keys)
        # Use the variational mean as the initial position after optimisation.
        mu_flat, unravel_fn = jax.flatten_util.ravel_pytree(final_vi_state.mu)
        position = unravel_fn(mu_flat)
        return FRVISamplerState(position=position, vi_state=final_vi_state)

    def step(
        self, rng_key: jax.Array, state: FRVISamplerState
    ) -> tuple[FRVISamplerState, fr.FRVIInfo]:
        """Draw one sample from the fitted variational distribution.

        Parameters
        ----------
        rng_key
            JAX random key for this draw.
        state
            Current ``FRVISamplerState`` containing the fitted ``vi_state``.

        Returns
        -------
        new_state
            ``FRVISamplerState`` with the drawn sample as ``position``.  The
            ``vi_state`` is unchanged (the fit is frozen after ``init``).
        info
            ``FRVIInfo`` with a placeholder ``elbo=0.0`` (the ELBO was
            computed during ``init``; no optimisation happens at step time).
        """
        samples = fr.sample(rng_key, state.vi_state, num_samples=1)
        # samples is a pytree with leading dim 1; take the first (only) draw.
        new_position = jax.tree.map(lambda x: x[0], samples)
        new_state = FRVISamplerState(position=new_position, vi_state=state.vi_state)
        return new_state, fr.FRVIInfo(elbo=jnp.asarray(0.0))


def _factory(
    logdensity_fn: Any,
    *,
    num_optimization_steps: int = 20_000,
    optimizer: Any = None,
    num_samples: int = 1000,
    **kwargs: Any,
) -> _FRVISamplerAlgorithm:
    """Build a full-rank VI sampler-mode kernel.

    The full VI optimisation loop runs during ``.init``; each ``.step`` draws
    one sample from the fitted full-rank Gaussian variational distribution.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density (log-posterior) callable ``x -> float``.
    num_optimization_steps
        Number of Adam optimisation steps to run inside ``.init``.  Default
        ``20_000`` (production).  Use ``5_000`` in tests.
    optimizer
        Optax ``GradientTransformation``.  Defaults to ``optax.adam(1e-2)``.
    num_samples
        Number of Monte Carlo samples per optimisation step for the ELBO
        gradient estimator.  Default ``1000``; lower values (e.g. 5) are
        standard for stochastic VI.  The upstream ``step`` default is 5.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    _FRVISamplerAlgorithm
        Kernel-like object with ``.init`` and ``.step`` methods.
    """
    if optimizer is None:
        optimizer = _adam_default
    # num_samples_per_step: how many MC draws per gradient step.
    # Use 5 (upstream default) for the VI loop; num_samples kwarg controls
    # the recipe-level draw count from the final distribution (not used here).
    num_samples_per_step = 5
    return _FRVISamplerAlgorithm(
        logdensity_fn=logdensity_fn,
        optimizer=optimizer,
        num_optimization_steps=num_optimization_steps,
        num_samples_per_step=num_samples_per_step,
    )


ENTRY = BaseMethod(
    name="fullrank_vi",
    family="vi",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(1),
    default_hp_space=(
        # num_optimization_steps is recipe-time by default (not BO-tuned),
        # but must be listed here to satisfy BaseMethod validation (at least
        # one HyperparamSpace required for non-specialised entries).
        # The BO loop can tune this if desired; recipe-builders should
        # override it directly via the factory kwarg.
        HyperparamSpace("num_optimization_steps", "int", low=2_000, high=100_000),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # VI is not a MH sampler
    notes=(
        "Full-rank variational inference (FRVI) in sampler mode. "
        "The full VI optimisation loop runs inside .init via jax.lax.scan "
        "over num_optimization_steps Adam steps. Each .step call draws one "
        "sample from the fitted full-rank Gaussian (N(mu, L @ L.T) where L "
        "is recovered from the flattened chol_params via _unflatten_cholesky). "
        "Hyperparameter-free at the trial level: num_optimization_steps and "
        "optimizer are recipe-time constants. "
        "grad_count_per_step=1 is an approximation: during optimisation each "
        "step consumes one ELBO gradient; at sample time no gradient is needed. "
        "Default: optax.adam(1e-2), num_optimization_steps=20_000 (production); "
        "use num_optimization_steps=5_000 in tests. "
        "Recommended ONLY for d <= 30: the cholesky parameterisation has "
        "O(d^2) parameters which become expensive and slow to converge at "
        "high dimension. Use meanfield_vi for d > 30."
    ),
)
