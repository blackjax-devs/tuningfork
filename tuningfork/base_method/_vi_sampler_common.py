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
"""Generic VI sampler-mode algorithm factory.

Both ``meanfield_vi`` and ``fullrank_vi`` present the VI optimiser as a
sampler-mode kernel (``init → step → step → ...``).  The class body is
identical: ``init`` runs a ``jax.lax.scan`` over the VI loop and stores the
final variational state; ``step`` draws one sample from the fitted
distribution.  The only difference is the VI module (``mf`` vs ``fr``) and
the ``SamplerStateClass`` (``MFVISamplerState`` vs ``FRVISamplerState``).

This module exposes ``make_vi_sampler_algorithm(vi_module, SamplerStateClass)``
which produces the concrete algorithm class.  The per-file modules keep their
exported names (``_MFVISamplerAlgorithm``, ``_FRVISamplerAlgorithm``) as thin
bindings to this factory, preserving all public symbols.
"""

from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["make_vi_sampler_algorithm"]

# Type alias to help mypy understand _vi_info_cls is always a callable type.
_InfoCls = type


def make_vi_sampler_algorithm(vi_module: Any, SamplerStateClass: type) -> type:
    """Build a VI sampler-mode algorithm class for the given VI module and state type.

    Parameters
    ----------
    vi_module
        BlackJAX VI module (``blackjax.vi.meanfield_vi`` or
        ``blackjax.vi.fullrank_vi``).  Must expose ``init``, ``step``, and
        ``sample`` functions plus ``MFVIInfo`` or ``FRVIInfo``.
    SamplerStateClass
        ``NamedTuple`` subclass used as the wrapper state (``MFVISamplerState``
        or ``FRVISamplerState``).  Must have ``position`` and ``vi_state``
        fields.

    Returns
    -------
    type
        A class with ``__init__``, ``init``, and ``step`` methods that present
        the VI optimiser as a sampler-mode kernel.
    """
    # Resolve the Info class once at factory-build time (not per step call).
    _maybe_info_cls = getattr(vi_module, "MFVIInfo", None) or getattr(
        vi_module, "FRVIInfo", None
    )
    if _maybe_info_cls is None:
        raise AttributeError(
            f"vi_module {vi_module!r} does not expose MFVIInfo or FRVIInfo"
        )
    _vi_info_cls: _InfoCls = _maybe_info_cls

    class _VISamplerAlgorithm:
        """Thin wrapper presenting a VI optimiser as a sampler-mode kernel.

        Attributes
        ----------
        init
            ``(init_position) -> SamplerStateClass``.  Runs the full VI
            optimisation loop via ``jax.lax.scan`` and stores the final
            variational state.  The ``position`` field of the returned state
            is set to the variational mean ``mu``.
        step
            ``(rng_key, state) -> (SamplerStateClass, VIInfo)``.  Draws
            one sample from the fitted variational distribution.
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

        def init(self, init_position: Any) -> Any:
            """Run the full VI optimisation loop; return the final state.

            Parameters
            ----------
            init_position
                Initial unconstrained parameter pytree (one chain's worth).

            Returns
            -------
            SamplerStateClass
                Wrapper state with ``position = mu`` (variational mean) and
                the final ``vi_state`` from the optimisation loop.
            """
            vi_init = vi_module.init(init_position, self._optimizer)

            def one_step(carry: Any, rng_key: jax.Array) -> tuple[Any, Any]:
                new_state, info = vi_module.step(
                    rng_key,
                    carry,
                    self._logdensity_fn,
                    self._optimizer,
                    self._num_samples_per_step,
                )
                return new_state, info

            keys = jax.random.split(jax.random.key(0), self._num_optimization_steps)
            final_vi_state, _infos = jax.lax.scan(one_step, vi_init, keys)
            # Use the variational mean as the initial position.
            mu_flat, unravel_fn = jax.flatten_util.ravel_pytree(final_vi_state.mu)
            position = unravel_fn(mu_flat)
            return SamplerStateClass(position=position, vi_state=final_vi_state)

        def step(self, rng_key: jax.Array, state: Any) -> tuple[Any, Any]:
            """Draw one sample from the fitted variational distribution.

            Parameters
            ----------
            rng_key
                JAX random key for this draw.
            state
                Current ``SamplerStateClass`` containing the fitted
                ``vi_state``.

            Returns
            -------
            new_state
                ``SamplerStateClass`` with the drawn sample as ``position``.
                The ``vi_state`` is unchanged (frozen after ``init``).
            info
                VI info namedtuple with a placeholder ``elbo=0.0`` (the ELBO
                was computed during ``init``; no optimisation at step time).
            """
            samples = vi_module.sample(rng_key, state.vi_state, num_samples=1)
            # samples is a pytree with leading dim 1; take the first (only) draw.
            new_position = jax.tree.map(lambda x: x[0], samples)
            new_state = SamplerStateClass(
                position=new_position, vi_state=state.vi_state
            )
            return new_state, _vi_info_cls(elbo=jnp.asarray(0.0))

    return _VISamplerAlgorithm
