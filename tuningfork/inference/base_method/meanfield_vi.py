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
"""Mean-field VI sampler-mode base method entry for tuningfork.

Wraps ``blackjax.vi.meanfield_vi`` as a dual-mode entry compatible with the
tuningfork runner contract.  The "sampler-mode" here means the full VI
optimisation loop runs inside ``init`` (via ``jax.lax.scan``), storing the
final variational state in a wrapper ``MFVISamplerState``.  Each subsequent
``step(rng_key, state)`` call draws **one** sample from the fitted variational
distribution, returning it as the new ``position``.

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
``num_optimization_steps=10000`` (production default).  Test suites should
override with ``num_optimization_steps=2000`` to keep run-times reasonable.
"""

from typing import Any, NamedTuple

import blackjax.vi.meanfield_vi as mf
import jax
import jax.numpy as jnp
import optax

from tuningfork.inference.base_method._base import BaseMethod, HyperparamSpace

__all__ = ["ENTRY", "_factory", "MFVISamplerState"]

# Default optimizer (recipe-time; can be overridden via factory kwarg).
_adam_default = optax.adam(1e-2)


class MFVISamplerState(NamedTuple):
    """Wrapper state for mean-field VI in sampler mode.

    Parameters
    ----------
    position
        Current draw from the fitted variational distribution.  Shape matches
        the model's parameter pytree.
    vi_state
        Final ``MFVIState`` from the VI optimisation loop.  Stored so that
        subsequent ``step`` calls can draw new samples from the same fit.
    """

    position: Any
    vi_state: mf.MFVIState


class _MFVISamplerAlgorithm:
    """Thin wrapper that presents the VI optimiser as a sampler-mode kernel.

    Attributes
    ----------
    init
        Callable ``(init_position) -> MFVISamplerState``.  Runs the full VI
        optimisation loop via ``jax.lax.scan`` and stores the final
        ``MFVIState``.  The ``position`` field of the returned state is set to
        the variational mean ``mu``.
    step
        Callable ``(rng_key, state) -> (MFVISamplerState, MFVIInfo)``.  Draws
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

    def init(self, init_position: Any) -> MFVISamplerState:
        """Run the full VI optimisation loop; return the final state.

        Parameters
        ----------
        init_position
            Initial unconstrained parameter pytree (one chain's worth).

        Returns
        -------
        MFVISamplerState
            Wrapper state with ``position = mu`` (variational mean) and the
            final ``vi_state`` from the optimisation loop.
        """
        vi_init = mf.init(init_position, self._optimizer)

        def one_step(carry: mf.MFVIState, rng_key: jax.Array):
            new_state, info = mf.step(
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
        return MFVISamplerState(position=position, vi_state=final_vi_state)

    def step(
        self, rng_key: jax.Array, state: MFVISamplerState
    ) -> tuple[MFVISamplerState, mf.MFVIInfo]:
        """Draw one sample from the fitted variational distribution.

        Parameters
        ----------
        rng_key
            JAX random key for this draw.
        state
            Current ``MFVISamplerState`` containing the fitted ``vi_state``.

        Returns
        -------
        new_state
            ``MFVISamplerState`` with the drawn sample as ``position``.  The
            ``vi_state`` is unchanged (the fit is frozen after ``init``).
        info
            ``MFVIInfo`` with a placeholder ``elbo=0.0`` (the ELBO was
            computed during ``init``; no optimisation happens at step time).
        """
        samples = mf.sample(rng_key, state.vi_state, num_samples=1)
        # samples is a pytree with leading dim 1; take the first (only) draw.
        new_position = jax.tree.map(lambda x: x[0], samples)
        new_state = MFVISamplerState(position=new_position, vi_state=state.vi_state)
        return new_state, mf.MFVIInfo(elbo=jnp.asarray(0.0))


def _factory(
    logdensity_fn: Any,
    *,
    num_optimization_steps: int = 10_000,
    optimizer: Any = None,
    num_samples: int = 1000,
    **kwargs: Any,
) -> _MFVISamplerAlgorithm:
    """Build a mean-field VI sampler-mode kernel.

    The full VI optimisation loop runs during ``.init``; each ``.step`` draws
    one sample from the fitted variational distribution.

    Parameters
    ----------
    logdensity_fn
        Unnormalised log-density (log-posterior) callable ``x -> float``.
    num_optimization_steps
        Number of Adam optimisation steps to run inside ``.init``.  Default
        ``10_000`` (production).  Use ``2_000`` in tests.
    optimizer
        Optax ``GradientTransformation``.  Defaults to ``optax.adam(1e-2)``.
    num_samples
        Number of Monte Carlo samples per optimisation step for the ELBO
        gradient estimator.  Default ``1000``; lower values (e.g. 5) are
        standard for stochastic VI.  Here we keep 1000 for recipes, but the
        upstream ``step`` default is 5.
    **kwargs
        Accepted for interface uniformity; ignored.

    Returns
    -------
    _MFVISamplerAlgorithm
        Kernel-like object with ``.init`` and ``.step`` methods.
    """
    if optimizer is None:
        optimizer = _adam_default
    # num_samples_per_step: how many MC draws per gradient step.
    # Use 5 (upstream default) for the VI loop; num_samples kwarg controls
    # the recipe-level draw count from the final distribution (not used here).
    num_samples_per_step = 5
    return _MFVISamplerAlgorithm(
        logdensity_fn=logdensity_fn,
        optimizer=optimizer,
        num_optimization_steps=num_optimization_steps,
        num_samples_per_step=num_samples_per_step,
    )


ENTRY = BaseMethod(
    name="meanfield_vi",
    family="vi",
    factory=_factory,
    grad_count_per_step=lambda info: jnp.asarray(1),
    default_hp_space=(
        # num_optimization_steps is recipe-time by default (not BO-tuned),
        # but must be listed here to satisfy BaseMethod validation (at least
        # one HyperparamSpace required for non-specialised entries).
        # The BO loop can tune this if desired; recipe-builders should
        # override it directly via the factory kwarg.
        HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
    ),
    needs_mass_matrix=False,
    target_acceptance_rate=None,  # VI is not a MH sampler
    notes=(
        "Mean-field variational inference (MFVI) in sampler mode. "
        "The full VI optimisation loop runs inside .init via jax.lax.scan "
        "over num_optimization_steps Adam steps. Each .step call draws one "
        "sample from the fitted mean-field Gaussian (N(mu, diag(exp(rho)))). "
        "Hyperparameter-free at the trial level: num_optimization_steps and "
        "optimizer are recipe-time constants. "
        "grad_count_per_step=1 is an approximation: during the optimisation "
        "phase, each step consumes one ELBO gradient (via reparameterisation); "
        "at sample time no gradient is evaluated. The approximation slightly "
        "over-counts for the sampling phase but is correct for the dominant "
        "optimisation cost. "
        "Default: optax.adam(1e-2), num_optimization_steps=10_000 (production); "
        "use num_optimization_steps=2_000 in tests. "
        "Preferred variant for d > 30 (fullrank_vi is recommended only for d <= 30)."
    ),
)
