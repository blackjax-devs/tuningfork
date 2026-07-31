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
"""Regression tests for _RECIPE_PROVENANCE_KEYS filtering in _recipe_runner.

Bug: LRD recipes store ``k_rank`` (and ``ncp_variant``) in
``base_method_params`` as provenance metadata. The certification path splatted
these into ``blackjax.mclmc(...)`` via ``_build_shared_kwargs``, causing::

    TypeError: as_top_level_api() got an unexpected keyword argument 'k_rank'

Fix: ``_RECIPE_PROVENANCE_KEYS`` is subtracted from the set forwarded to
``_build_shared_kwargs``.

Affected recipes (all carry ``k_rank`` in ``base_method_params``):
- ill_cond_50  low__mclmc_lrd__mclmc_lrd_tuning
- german_credit low__mclmc_lrd__mclmc_lrd_tuning
- stoch_vol     low__mclmc_lrd__mclmc_lrd_tuning_flatinit  (also ncp_variant)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# 1. Constant membership
# ---------------------------------------------------------------------------


def test_provenance_keys_contains_k_rank_and_ncp_variant() -> None:
    """_RECIPE_PROVENANCE_KEYS must cover both known LRD provenance fields."""
    from tuningfork.recipes._recipe_runner import _RECIPE_PROVENANCE_KEYS

    assert "k_rank" in _RECIPE_PROVENANCE_KEYS, (
        "k_rank must be in _RECIPE_PROVENANCE_KEYS — "
        "it is an LRD provenance field, not a kernel arg"
    )
    assert "ncp_variant" in _RECIPE_PROVENANCE_KEYS, (
        "ncp_variant must be in _RECIPE_PROVENANCE_KEYS — "
        "it is a stoch_vol variant tag, not a kernel arg"
    )


# ---------------------------------------------------------------------------
# 2. _build_shared_kwargs strips provenance keys
# ---------------------------------------------------------------------------


def test_build_shared_kwargs_strips_k_rank() -> None:
    """k_rank in params_override must NOT appear in the returned shared_kwargs."""
    import jax.numpy as jnp

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.recipes._recipe_runner import _build_shared_kwargs

    base_method = BASE_METHODS["mclmc"]

    # Simulate what the rerun path passes: base_method_params minus step_size/IMM,
    # but still carrying k_rank as LRD provenance.
    params_override = {"L": 5.87, "k_rank": 40}
    batched_params = {
        "step_size": jnp.ones(4) * 0.1,
        "inverse_mass_matrix": jnp.eye(2)[None].repeat(4, axis=0),
        "L": jnp.ones(4) * 5.87,
    }

    shared_kwargs, _ = _build_shared_kwargs(
        base_method,
        "mclmc",
        batched_params,
        None,  # batched_warmup_info
        None,  # warmup_inner_kernel
        None,  # step_policy
        params_override=params_override,
    )

    assert (
        "k_rank" not in shared_kwargs
    ), f"k_rank leaked into shared_kwargs: {list(shared_kwargs.keys())}"


def test_build_shared_kwargs_strips_ncp_variant() -> None:
    """ncp_variant in params_override must NOT appear in the returned shared_kwargs."""
    import jax.numpy as jnp

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.recipes._recipe_runner import _build_shared_kwargs

    base_method = BASE_METHODS["mclmc"]

    params_override = {"L": 5.87, "k_rank": 40, "ncp_variant": "stoch_vol_flatinit_ncp"}
    batched_params = {
        "step_size": jnp.ones(4) * 0.1,
        "inverse_mass_matrix": jnp.eye(2)[None].repeat(4, axis=0),
        "L": jnp.ones(4) * 5.87,
    }

    shared_kwargs, _ = _build_shared_kwargs(
        base_method,
        "mclmc",
        batched_params,
        None,
        None,
        None,
        params_override=params_override,
    )

    assert (
        "ncp_variant" not in shared_kwargs
    ), f"ncp_variant leaked into shared_kwargs: {list(shared_kwargs.keys())}"


def test_build_shared_kwargs_passes_legitimate_extra_kwargs() -> None:
    """Non-provenance extra kwargs (e.g. desired_energy_var_max_ratio) must pass through."""
    import jax.numpy as jnp

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.recipes._recipe_runner import _build_shared_kwargs

    base_method = BASE_METHODS["mclmc"]

    # desired_energy_var_max_ratio is a real mclmc kwarg (not a provenance key)
    params_override = {"L": 5.87, "desired_energy_var_max_ratio": 0.05, "k_rank": 40}
    batched_params = {
        "step_size": jnp.ones(4) * 0.1,
        "inverse_mass_matrix": jnp.eye(2)[None].repeat(4, axis=0),
        "L": jnp.ones(4) * 5.87,
    }

    shared_kwargs, _ = _build_shared_kwargs(
        base_method,
        "mclmc",
        batched_params,
        None,
        None,
        None,
        params_override=params_override,
    )

    # k_rank must be stripped; desired_energy_var_max_ratio must survive
    assert "k_rank" not in shared_kwargs
    assert (
        "desired_energy_var_max_ratio" in shared_kwargs
    ), "Legitimate kwarg 'desired_energy_var_max_ratio' was incorrectly stripped"


# ---------------------------------------------------------------------------
# 3. Kernel factory does not receive k_rank (end-to-end factory call guard)
# ---------------------------------------------------------------------------


def test_mclmc_factory_call_succeeds_with_lrd_base_method_params() -> None:
    """Simulate the rerun path with LRD base_method_params — must not raise TypeError.

    This is the direct regression guard for:
        TypeError: as_top_level_api() got an unexpected keyword argument 'k_rank'
    """
    import blackjax
    import jax.numpy as jnp

    from tuningfork.base_method import BASE_METHODS
    from tuningfork.recipes._recipe_runner import _build_shared_kwargs

    base_method = BASE_METHODS["mclmc"]
    logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)

    # Matches exactly what the rerun path builds from LRD recipe base_method_params
    # (step_size/IMM already stripped by the _recipe_params_override comprehension).
    lrd_params_override = {"L": 5.878, "k_rank": 40}
    batched_params = {
        "step_size": jnp.ones(4) * 8.066,
        "inverse_mass_matrix": jnp.ones((4, 50)),
        "L": jnp.ones(4) * 5.878,
    }

    shared_kwargs, _ = _build_shared_kwargs(
        base_method,
        "mclmc",
        batched_params,
        None,
        None,
        None,
        params_override=lrd_params_override,
    )

    # The factory call must not raise TypeError: unexpected keyword argument 'k_rank'
    step_size = 8.066
    imm = jnp.ones(50)
    kernel = blackjax.mclmc(
        logdensity_fn,
        step_size=step_size,
        inverse_mass_matrix=imm,
        **shared_kwargs,
    )
    assert kernel is not None
