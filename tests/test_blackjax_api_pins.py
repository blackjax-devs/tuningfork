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
"""Tripwire tests for BlackJAX API shapes that bjx_bench relies on.

These are defensive: if BlackJAX upstream changes the return-tuple shape or
NamedTuple fields of any kernel/adaptation we depend on, these tests fire
with a clear message pointing at the file in bjx-bench that needs an update.

Status: 1 of 3 META-004 watch threshold (see WORKLOG.md). When 2 more upstream
drifts land, promote to a Makefile target `make verify-blackjax-api`.
"""

import blackjax
import jax
import jax.numpy as jnp
import pytest
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.mclmc import MCLMCInfo

pytestmark = pytest.mark.fast


# ───────────────── 1. mclmc_find_L_and_step_size return shape ─────────────────
def test_mclmc_find_L_and_step_size_returns_3_tuple():
    """Pinned in T2.6c: bjx_bench/calibration/tier_b.py:_run_warmup unpacks
    (state, params, _n_tuning_steps). If BlackJAX changes this to a 2-tuple
    (matching its docstring) or a different shape, the unpack fails opaquely.

    Caught at T2.6c commit 55792ac. Documented in WORKLOG META-004 watch.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    key = jax.random.key(0)
    # Build the kernel and state following tier_b.py:_run_warmup line 417-419
    default_kwargs = {"L": 1.0, "step_size": 0.1}
    kernel = blackjax.mclmc(logdensity_fn, **default_kwargs)
    init_state = kernel.init({"x": jnp.zeros(5)}, key)
    # Call mclmc_find_L_and_step_size with raw build_kernel (not the wrapper)
    mclmc_kernel = blackjax.mclmc.build_kernel()
    result = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel,
        num_steps=20,
        state=init_state,
        rng_key=key,
        logdensity_fn=logdensity_fn,
        diagonal_preconditioning=True,
    )
    assert len(result) == 3, (
        f"BlackJAX changed mclmc_find_L_and_step_size return arity from 3 to {len(result)}. "
        f"Update bjx_bench/calibration/tier_b.py:_run_warmup unpack accordingly."
    )


# ───────────────── 2. MCLMCInfo NamedTuple fields ─────────────────
def test_mclmc_info_fields():
    """Pinned in T2.4: bjx_bench/algorithms/mclmc.py grad_count_per_step is a
    constant 2 (NOT 2 × info.num_integration_steps), because MCLMCInfo lacks
    that field. If BlackJAX adds num_integration_steps to MCLMCInfo, our
    constant-2 formula becomes incorrect for any integrator beyond default.
    """
    expected = ("logdensity", "kinetic_change", "energy_change", "nonans")
    assert MCLMCInfo._fields == expected, (
        f"BlackJAX MCLMCInfo fields changed from {expected} to {MCLMCInfo._fields}. "
        f"Update bjx_bench/algorithms/mclmc.py:grad_count_per_step if num_integration_steps appeared."
    )


# ───────────────── 3. MCLMCAdaptationState fields (params dict) ─────────────────
def test_mclmc_adaptation_state_fields():
    """Pinned in T2.6c: bjx_bench/calibration/tier_b.py treats the warmup
    output as a dict-like with at least step_size and L. If BlackJAX renames
    or removes either, our trial-params merge silently misbehaves.
    """
    fields = MCLMCAdaptationState._fields
    for name in ("step_size", "L", "inverse_mass_matrix"):
        assert name in fields, (
            f"MCLMCAdaptationState lost field {name!r}. Current fields: {fields}. "
            f"Update bjx_bench/calibration/tier_b.py:_run_warmup."
        )


# ───────────────── 4. HMCInfo / NUTSInfo expose num_integration_steps ─────────────────
def test_hmc_nuts_info_have_num_integration_steps():
    """Pinned in T2.1+T2.2: grad_count_per_step for HMC and NUTS reads
    info.num_integration_steps. If that field disappears, our gradient-count
    accounting breaks silently (defaulting to 0?).
    """
    from blackjax.mcmc.hmc import HMCInfo
    from blackjax.mcmc.nuts import NUTSInfo

    assert "num_integration_steps" in HMCInfo._fields, (
        "blackjax.mcmc.hmc.HMCInfo lost num_integration_steps. "
        "Update bjx_bench/algorithms/{hmc,nuts}.py:grad_count_per_step."
    )
    assert "num_integration_steps" in NUTSInfo._fields, (
        "blackjax.mcmc.nuts.NUTSInfo lost num_integration_steps. "
        "Update bjx_bench/algorithms/{hmc,nuts}.py:grad_count_per_step."
    )


# ───────────────── 5. window_adaptation works for HMC/NUTS/Barker/MALA ─────────────────
def test_window_adaptation_constructs_for_supported_kernels():
    """Pinned in T2.6b: bjx_bench/calibration/tier_b.py:_run_warmup uses
    blackjax.window_adaptation for kernels with needs_mass_matrix=True (and
    structurally for MALA in T2.6c, even though MALA's needs_mass_matrix=False
    — sanity check). Verifies the construction path stays valid.
    """

    def logdensity_fn(x):
        return -0.5 * jnp.sum(x["x"] ** 2)

    for kernel_factory in (blackjax.hmc, blackjax.nuts, blackjax.barker, blackjax.mala):
        # window_adaptation construction (no .run) — fast smoke
        try:
            wa = blackjax.window_adaptation(kernel_factory, logdensity_fn)
            assert hasattr(wa, "run"), (
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                f"returned object lacking .run; surface changed."
            )
        except Exception as exc:
            raise AssertionError(
                f"blackjax.window_adaptation(blackjax.{kernel_factory.__name__}) "
                f"failed at construction: {type(exc).__name__}: {exc}"
            ) from exc
