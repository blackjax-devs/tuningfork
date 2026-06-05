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
"""Low-rank mass matrix adaptation via Fisher divergence minimisation.

.. note::

   **Multi-chain vmap fix landed upstream 2026-05-18.**

   Earlier versions of ``blackjax.adaptation.low_rank_adaptation.low_rank_window_adaptation``
   returned an ``inverse_mass_matrix`` carrying a Python closure
   (``Metric.momentum_generator``) that ``jax.vmap`` could not stack across
   chains.  Fixed by [blackjax#917](https://github.com/blackjax-devs/blackjax/pull/917)
   (merged at ``b094083c``), which replaces the closure with a pure-array
   ``LowRankInverseMassMatrix`` NamedTuple — vmap-compatible and pytree-flat.

   Required blackjax version: ``b094083c`` or later.

Wraps ``blackjax.adaptation.low_rank_adaptation.low_rank_window_adaptation``,
which adapts a mass matrix of the form

    M^{-1} = diag(σ) (I + U(Λ - I)U^T) diag(σ)

where σ ∈ R^d is diagonal scaling, U ∈ R^{d × k} has orthonormal columns
(k ≤ d), and Λ = diag(λ) with λ > 0.  When λ = 1, the metric reduces to
diagonal; when k approaches d, it approximates full-rank adaptation at O(dk)
cost instead of O(d²).

The adaptation minimises the Fisher divergence (matching nutpie's
implementation) and follows Stan's three-phase warmup schedule (fast → slow
windows → final fast).

Compatible with HMC-family kernels (HMC, NUTS, Barker, MALA — verified by
tripwire tests in ``tests/test_api_pins_mcmc.py``).

Runner signature (multi-chain contract)::

    _runner(rng_key, init_position, n_warmup, base_method,
            *, logdensity_fn, target_acceptance_rate=None,
            max_rank=10, num_chains: int = 4, **kwargs)
    -> (states, adapted_params)

Where:

- ``rng_key`` is a single key; split internally into ``num_chains`` keys.
- ``init_position`` is a single pytree (one chain's worth); replicated
  across chains internally via ``_maybe_replicate`` unless the caller
  pre-batches it (leading dim == ``num_chains``).
- ``states`` is a batched pytree with leading dim ``num_chains``.
- ``adapted_params`` contains ``"step_size"`` (shape ``(num_chains,)``) and
  ``"inverse_mass_matrix"`` (a ``LowRankInverseMassMatrix`` NamedTuple with
  public ``sigma`` / ``U`` / ``lam`` array fields, batched on the leading
  ``num_chains`` axis).

The low-rank IMM is represented internally by three arrays:

- ``sigma``: shape ``(d,)``, diagonal scaling.
- ``U``: shape ``(d, max_rank)``, orthonormal eigenvectors (padded with zeros
  if the actual rank is < max_rank).
- ``lam``: shape ``(max_rank,)``, eigenvalues.

These are stored in an IMM sidecar via ``Recipe.save_imm_sidecar()`` as a
structured dictionary ``{"sigma": ..., "U": ..., "lam": ...}`` (one dict per
chain, stacked).

If the ``base_method`` has a BO-tunable HP that is NOT step_size or
inverse_mass_matrix, the default value for that HP is injected into the
warmup call so the kernel can construct itself; BO trials later override
those HPs via trial_params.
"""

from typing import Any

import blackjax
import jax

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.warmup._base import Warmup
from tuningfork.warmup._window_adaptation_common import _window_adaptation_body

__all__ = ["ENTRY"]


def _runner(
    rng_key: jax.Array,
    init_position: Any,
    n_warmup: int,
    base_method: Any,  # BaseMethod; not imported to avoid circular dep at module level
    *,
    logdensity_fn: Any,
    target_acceptance_rate: float | None = None,
    max_rank: int = 10,
    num_chains: int = 4,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any], Any]:
    """Run blackjax.adaptation.low_rank_adaptation.low_rank_window_adaptation over ``num_chains`` chains via vmap.

    Parameters
    ----------
    rng_key
        JAX random key for the adaptation run.  Split internally into
        ``num_chains`` independent per-chain keys.
    init_position
        Initial unconstrained parameter dict (from the model's prior sample).
        A SINGLE pytree (one chain's worth).  The runner replicates it across
        ``num_chains`` unless the caller pre-batches it (leading dim ==
        ``num_chains``).
    n_warmup
        Number of adaptation steps.
    base_method
        ``BaseMethod`` entry (carries ``factory``, ``default_hp_space``,
        ``target_acceptance_rate``).
    logdensity_fn
        BlackJAX-compatible log-density function.
    target_acceptance_rate
        Override for the dual-averaging target.  Falls back to
        ``base_method.target_acceptance_rate``, then ``0.80``.
    max_rank
        Maximum number of eigenvectors to retain in the low-rank correction.
        Default ``10``. Use ``None`` to allow the adaptation to choose
        dynamically (currently: min(d, 10)).
    num_chains
        Number of independent chains to run in parallel via ``jax.vmap``.
        Default ``4``, matching Stan/NumPyro convention.  Pass ``num_chains=1``
        explicitly for BO trials (intentionally single-chain — chain count is
        orthogonal to HP tuning).
    **kwargs
        Additional keyword arguments forwarded to ``low_rank_window_adaptation``
        (e.g. ``num_integration_steps`` for HMC — the warmup kernel needs
        it to build its leapfrog integrator, even though BO will tune it
        later).

    Returns
    -------
    states
        Post-warmup BlackJAX kernel states, batched over ``num_chains``.
        ``states.position`` has shape ``(num_chains, d)``.
    adapted_params
        Dict with at least ``"step_size"`` and ``"inverse_mass_matrix"``.
        ``"step_size"`` has shape ``(num_chains,)``.
        ``"inverse_mass_matrix"`` is a ``LowRankInverseMassMatrix`` NamedTuple
        with public ``sigma`` / ``U`` / ``lam`` array fields (batched on the
        leading ``num_chains`` axis).  Pytree-flat / vmap-compatible by
        construction since [blackjax#917](https://github.com/blackjax-devs/blackjax/pull/917).
    """

    # NB: blackjax PR #923 (2026-05-20, on main 2026-05-20) renamed the upstream
    # function `low_rank_window_adaptation` → `window_adaptation_low_rank` with
    # no deprecation alias. The top-level symbol `blackjax.window_adaptation_low_rank`
    # is the new entry point; the legacy module path
    # `blackjax.adaptation.low_rank_adaptation.low_rank_window_adaptation` no
    # longer exists (caused the test-slow failure on main 2026-05-21).
    def _build_low_rank_warmup(
        warmup_algorithm: Any,
        logdensity_fn: Any,
        *,
        target_acceptance_rate: float,
        **warmup_kwargs: Any,
    ) -> Any:
        return blackjax.window_adaptation_low_rank(
            warmup_algorithm,
            logdensity_fn,
            max_rank=max_rank,
            target_acceptance_rate=target_acceptance_rate,
            **warmup_kwargs,
        )

    return _window_adaptation_body(
        rng_key,
        init_position,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
        num_chains=num_chains,
        warmup_builder_fn=_build_low_rank_warmup,
        **kwargs,
    )


ENTRY = Warmup(
    name="window_adaptation_low_rank_imm",
    runner=_runner,  # type: ignore[arg-type]
    compatible_methods=(
        "hmc",
        "nuts",
        "mhmc",
        "dynamic_hmc",
        "dmhmc",
        "barker",
        "mala",
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
    ),
    default_hp_space=(
        # max_rank: number of eigenvectors in the low-rank correction.
        # Declared here so the recipe runner persists it into warmup_params
        # (and emit_script can read it back via recipe.warmup_params["max_rank"]).
        # Default=10 matches the _runner default and all existing catalog recipes.
        # BO callers may widen the range; recipes use warmup_kwargs_override to pin it.
        HyperparamSpace("max_rank", "int", low=10, high=10),
    ),
    notes=(
        "Low-rank mass matrix adaptation via Fisher divergence minimisation "
        "(nutpie algorithm; :cite:`seyboldt2026preconditioning`). Metric has the "
        "form M^{-1} = diag(σ)(I + U(Λ-I)U^T)diag(σ), enabling O(dk) kernel "
        "operations when rank k << d.  Compatible with hmc, nuts, barker, mala, "
        "and laplace_* variants. Use when posterior has strong correlations but "
        "d is too large for dense adaptation.  Default max_rank=10; tune per "
        "model.  multi-chain by default (num_chains=4 via jax.vmap); per-chain "
        "adapted_params returned (step_size shape (num_chains,), Metric object "
        "encoding (sigma, U, lam))."
    ),
)
