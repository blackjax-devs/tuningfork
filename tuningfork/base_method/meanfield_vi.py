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
import jax.numpy as jnp
import optax

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._vi_sampler_common import make_vi_sampler_algorithm

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


# Build the concrete algorithm class via the generic factory.
# _MFVISamplerAlgorithm is exported for backward-compatibility (recipes and
# tests may import it directly from this module).
_MFVISamplerAlgorithm = make_vi_sampler_algorithm(mf, MFVISamplerState)


def _factory(
    logdensity_fn: Any,
    *,
    num_optimization_steps: int = 10_000,
    optimizer: Any = None,
    num_samples: int = 1000,
    **kwargs: Any,
) -> Any:
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
    grad_count_convention="1",
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
