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
"""Extended coverage: all 4 laplace_* × all 3 window-adaptation warmups.

Verifies that the laplace-marginal warmup pathway (Phase 2a, Decision 1)
composes correctly for every valid combination of:

- **4 laplace_* base methods**: laplace_hmc, laplace_dhmc, laplace_mhmc,
  laplace_dmhmc
- **3 warmup strategies**: window_adaptation_diag_imm,
  window_adaptation_dense_imm, window_adaptation_low_rank_imm

Testing on ``eight_schools_ncp`` — the canonical hierarchical model with
phi=(mu, tau) [dim=2] and theta=(theta_raw,) [dim=8].

Each test asserts:
1. Warmup completes without exception (C2).
2. Adapted (step_size, IMM) are finite (C3a).
3. IMM has shape ``(num_chains, dim_phi)`` for diag / ``(num_chains,
   dim_phi, dim_phi)`` for dense (C3b — critical for correctness).

n_warmup=200 keeps the test suite under 3 minutes total across all 12 cells.

Upstream vmap fix landed 2026-05-18
------------------------------------
Earlier blackjax versions returned a ``Metric`` whose ``momentum_generator``
attribute was a Python closure that ``jax.vmap`` could not stack across
chains.  Fixed by [blackjax#917](https://github.com/blackjax-devs/blackjax/pull/917)
at ``b094083c``: the new ``LowRankInverseMassMatrix`` NamedTuple is
pytree-flat and vmap-compatible.  All 12 cells (3 warmups × 4 laplace_*
samplers) are expected to pass on blackjax ``b094083c`` or later.
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax.mcmc.laplace_marginal import laplace_marginal_factory

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.warmup import WARMUPS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared model setup (built once per test — parametrize reuses the same
# model object, so L-BFGS warm-starting is not shared, but that's fine for
# unit tests at this scale)
# ---------------------------------------------------------------------------

EIGHT_SCHOOLS = "eight_schools_ncp"
DIM_PHI = 2  # phi = {mu, tau}

LAPLACE_METHODS = [
    "laplace_hmc",
    "laplace_dhmc",
    "laplace_mhmc",
    "laplace_dmhmc",
]

WARMUP_STRATEGIES = [
    "window_adaptation_diag_imm",
    "window_adaptation_dense_imm",
    "window_adaptation_low_rank_imm",
]


def _build_eight_schools_marginal(seed=42):
    """Return (phi_init, theta_init, marginal_logdensity_fn, log_joint_fn)."""
    model = MODELS[EIGHT_SCHOOLS]
    key = jax.random.key(seed)
    init_position, joint_logdensity_fn, _model_data = build_logdensity_fn(key, model)

    phi_init = {"mu": init_position["mu"], "tau": init_position["tau"]}
    theta_init = {"theta_raw": init_position["theta_raw"]}

    def log_joint_fn(theta, phi):
        return joint_logdensity_fn({"theta_raw": theta["theta_raw"], **phi})

    laplace = laplace_marginal_factory(log_joint_fn, theta_init)

    def marginal_logdensity_fn(phi):
        lp, _theta_star = laplace(phi)
        return lp

    return phi_init, theta_init, marginal_logdensity_fn, log_joint_fn


@pytest.mark.parametrize("laplace_name", LAPLACE_METHODS)
@pytest.mark.parametrize("warmup_name", WARMUP_STRATEGIES)
def test_laplace_marginal_warmup_composition(warmup_name, laplace_name):
    """All 4 laplace_* × all 3 warmups compose on eight_schools_ncp.

    Assertions:
    - Warmup runs without exception.
    - step_size and IMM are finite.
    - IMM shape equals (num_chains, dim_phi) [diag] or
      (num_chains, dim_phi, dim_phi) [dense/low-rank projected diagonal].

    Design note: window_adaptation_low_rank_imm returns a
    ``LowRankInverseMassMatrix`` NamedTuple (not a plain array) for
    inverse_mass_matrix.  We check finiteness on sigma/U/lam components
    via ``jax.tree.leaves`` when the result is not a plain ndarray.
    """
    phi_init, _theta_init, marginal_logdensity_fn, _log_joint_fn = (
        _build_eight_schools_marginal()
    )

    warmup = WARMUPS[warmup_name]
    base_method = BASE_METHODS[laplace_name]

    n_warmup = 200
    num_chains = 2

    _states, adapted_params, *_ = warmup.runner(
        jax.random.key(0),
        phi_init,
        n_warmup,
        base_method,
        logdensity_fn=marginal_logdensity_fn,
        num_chains=num_chains,
    )

    step_size = adapted_params["step_size"]
    imm = adapted_params["inverse_mass_matrix"]

    # C3a: step_size is always a plain array — must be finite
    assert jnp.all(
        jnp.isfinite(step_size)
    ), f"[{warmup_name} x {laplace_name}] step_size NaN/Inf: {step_size}"

    # C3b: IMM shape check.
    # For diag: expect (num_chains, dim_phi).
    # For dense: expect (num_chains, dim_phi, dim_phi).
    # For low_rank: returns a Metric pytree — check finiteness on all leaves.
    if warmup_name == "window_adaptation_diag_imm":
        assert hasattr(imm, "shape"), f"Expected array IMM for diag, got {type(imm)}"
        assert imm.shape == (num_chains, DIM_PHI), (
            f"[{warmup_name} x {laplace_name}] IMM shape {imm.shape} "
            f"!= ({num_chains}, {DIM_PHI})"
        )
        assert jnp.all(
            jnp.isfinite(imm)
        ), f"[{warmup_name} x {laplace_name}] IMM NaN/Inf: {imm}"
    elif warmup_name == "window_adaptation_dense_imm":
        assert hasattr(imm, "shape"), f"Expected array IMM for dense, got {type(imm)}"
        assert imm.shape == (num_chains, DIM_PHI, DIM_PHI), (
            f"[{warmup_name} x {laplace_name}] IMM shape {imm.shape} "
            f"!= ({num_chains}, {DIM_PHI}, {DIM_PHI})"
        )
        assert jnp.all(
            jnp.isfinite(imm)
        ), f"[{warmup_name} x {laplace_name}] IMM NaN/Inf: {imm}"
    elif warmup_name == "window_adaptation_low_rank_imm":
        # low_rank IMM is a LowRankInverseMassMatrix NamedTuple (sigma, U, lam
        # arrays, batched on the leading num_chains axis).  Check finiteness
        # on all leaves; verify the structural fields exist.
        assert hasattr(imm, "sigma") and hasattr(imm, "U") and hasattr(imm, "lam"), (
            f"[{warmup_name} x {laplace_name}] IMM should be a "
            f"LowRankInverseMassMatrix NamedTuple with sigma/U/lam, got {type(imm)}"
        )
        # sigma leading dim is num_chains
        assert imm.sigma.shape[0] == num_chains, (
            f"[{warmup_name} x {laplace_name}] sigma leading dim {imm.sigma.shape[0]} "
            f"!= num_chains {num_chains}"
        )
        imm_leaves = jax.tree.leaves(imm)
        for leaf in imm_leaves:
            assert jnp.all(
                jnp.isfinite(leaf)
            ), f"[{warmup_name} x {laplace_name}] low-rank IMM leaf NaN/Inf"
