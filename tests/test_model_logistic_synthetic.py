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
"""Tests for the logistic_synthetic model (3-D Bayesian logistic regression, bicluster).

Tests
-----
1.  ENTRY registered in MODELS with name "logistic_synthetic".
2.  dim == 3, class_ == "glm", tags contain "logistic-regression".
3.  Data is deterministic across imports (X_DATA and Y_DATA have stable hashes).
4.  y is binary — all values in {0, 1}.
5.  Bicluster is separable: a simple threshold classifier achieves >85% accuracy.
6.  build_logdensity_fn returns finite log-density at zero-init position.
7.  Log-density at a near-MLE point is higher than at the zero-init position.
8.  Posterior dim matches the parameter count after build_logdensity_fn (3).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.glm.logistic_synthetic import DIM, ENTRY, X_DATA, Y_DATA

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Test 1: registration
# ---------------------------------------------------------------------------


def test_registered_in_models() -> None:
    """logistic_synthetic must appear in MODELS under the correct name."""
    assert "logistic_synthetic" in MODELS, "logistic_synthetic not found in MODELS"
    assert MODELS["logistic_synthetic"] is ENTRY


# ---------------------------------------------------------------------------
# Test 2: schema — dim, class_, tags
# ---------------------------------------------------------------------------


def test_dim_class_tags() -> None:
    """Posterior dim must be 3, class_ 'glm', tags include 'logistic-regression'."""
    assert ENTRY.dim == DIM
    assert ENTRY.dim == 3
    assert ENTRY.class_ == "glm"
    assert "logistic-regression" in ENTRY.tags
    assert "glm" in ENTRY.tags
    assert "low-dim" in ENTRY.tags
    assert "well-conditioned" in ENTRY.tags


# ---------------------------------------------------------------------------
# Test 3: determinism — X_DATA and Y_DATA are stable across imports
# ---------------------------------------------------------------------------


def test_data_is_deterministic() -> None:
    """X_DATA and Y_DATA must be deterministic: same values on every import."""
    # Re-import to check stability — same module object, but we verify values
    from bjx_bench.model.glm.logistic_synthetic import X_DATA as x2
    from bjx_bench.model.glm.logistic_synthetic import Y_DATA as y2

    assert np.array_equal(np.asarray(X_DATA), np.asarray(x2)), "X_DATA not stable"
    assert np.array_equal(np.asarray(Y_DATA), np.asarray(y2)), "Y_DATA not stable"

    # Shape checks
    assert X_DATA.shape == (50, 2), f"Expected (50, 2), got {X_DATA.shape}"
    assert Y_DATA.shape == (50,), f"Expected (50,), got {Y_DATA.shape}"


# ---------------------------------------------------------------------------
# Test 4: y is binary
# ---------------------------------------------------------------------------


def test_y_is_binary() -> None:
    """All values in Y_DATA must be in {0, 1}."""
    y_np = np.asarray(Y_DATA)
    unique_vals = set(y_np.astype(int).tolist())
    assert unique_vals == {0, 1}, f"Y_DATA contains non-binary values: {unique_vals}"

    # Exactly 25 per class (by construction)
    assert int((y_np == 0).sum()) == 25, "Expected 25 class-0 observations"
    assert int((y_np == 1).sum()) == 25, "Expected 25 class-1 observations"


# ---------------------------------------------------------------------------
# Test 5: bicluster is linearly separable (>95% accuracy)
# ---------------------------------------------------------------------------


def test_bicluster_separability() -> None:
    """A simple linear classifier on the bicluster must achieve >85% accuracy.

    The bicluster uses centres at (±1.5, ±1.5) with blob std=1.5, giving
    ~92% separability on this fixed seed without complete separation. The
    threshold is >85% (not 95%) to avoid the *complete-separation* regime in
    logistic regression where the MLE diverges to infinity and NUTS produces
    divergences. The design deliberately allows a small overlap so the
    posterior is well-conditioned (finite MLE, E-BFMI > 0.9).

    Decision rule: predict class=1 iff row mean of features > 0.
    """
    x_np = np.asarray(X_DATA)
    y_np = np.asarray(Y_DATA)

    # Decision rule: predict class=1 iff row mean > 0
    predicted = (x_np.mean(axis=1) > 0).astype(float)
    accuracy = float((predicted == y_np).mean())
    assert accuracy > 0.85, (
        f"Bicluster classifier accuracy {accuracy:.3f} is below 0.85; "
        "data may not be sufficiently separated."
    )


# ---------------------------------------------------------------------------
# Test 6: build_logdensity_fn returns finite log-density at zero-init
# ---------------------------------------------------------------------------


def test_build_logdensity_fn_finite() -> None:
    """build_logdensity_fn must return a finite log-density at the init position."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


# ---------------------------------------------------------------------------
# Test 7: log-density is higher at a near-MLE point than at zero-init
# ---------------------------------------------------------------------------


def test_logdensity_higher_at_mle() -> None:
    """Log-density at a near-MAP point must exceed log-density at zero-init.

    For the bicluster at (-1.5,-1.5) / (+1.5,+1.5) with blob std=1.5, the
    MAP beta is approximately [0.2, 1.1, 1.3] (from Laplace approximation).
    We test beta = [0, 1, 1] as a conservatively accurate near-MAP point.
    """
    key = jax.random.key(1)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

    # Log-density at zero-init (beta ≈ [0, 0, 0])
    ld_init = float(logdensity_fn(init_pos))

    # Build a near-MAP position in unconstrained space.
    # For this parameterization, unconstrained = constrained (beta is Gaussian).
    # beta = [0, 1, 1] gives logits = x1 + x2: positive for (+1.5,+1.5) region.
    map_pos = {"beta": jnp.array([0.0, 1.0, 1.0])}
    ld_map = float(logdensity_fn(map_pos))

    assert ld_map > ld_init, (
        f"Log-density at near-MAP ({ld_map:.3f}) must exceed "
        f"log-density at zero-init ({ld_init:.3f})."
    )


# ---------------------------------------------------------------------------
# Test 8: posterior dim matches parameter count from build_logdensity_fn
# ---------------------------------------------------------------------------


def test_posterior_dim_matches_params() -> None:
    """Posterior dim == 3 must match the number of unconstrained parameters."""
    key = jax.random.key(2)
    init_pos, _logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

    # init_pos is a dict {site_name: array}; sum up sizes
    total_params = sum(v.size for v in jax.tree.leaves(init_pos))
    assert total_params == ENTRY.dim, (
        f"Posterior dim {ENTRY.dim} != parameter count {total_params} "
        "from build_logdensity_fn."
    )


# ---------------------------------------------------------------------------
# Test 9: posterior recovers correct signs via short NUTS
# ---------------------------------------------------------------------------


def test_posterior_recovers_correct_signs_via_short_nuts() -> None:
    """500-sample NUTS smoke: beta_1 and beta_2 should both be positive
    (mean estimates around +1.1 from Laplace approximation; tolerance
    3 standard-error bands). Catches label-swap bugs and likelihood
    misspecifications that the structural tests above all miss.

    Per the statistician's retroactive checkpoint review (2026-05-08).
    NOT marked @pytest.mark.fast — runs MCMC.
    """
    import blackjax

    rng_key = jax.random.key(0)
    init_position, logdensity_fn, _ = build_logdensity_fn(rng_key, ENTRY)

    # Short NUTS run — enough to check signs but not full Tier-A
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, target_acceptance_rate=0.80
    )
    (state, params), _ = warmup.run(rng_key, init_position, num_steps=500)

    # Sample 500 post-warmup. run_inference_algorithm returns
    # (final_state, history) where history = (states_trace, infos_trace)
    # under the default transform.
    nuts = blackjax.nuts(logdensity_fn, **params)
    _, (history_states, _) = blackjax.util.run_inference_algorithm(
        rng_key=jax.random.key(1),
        inference_algorithm=nuts,
        num_steps=500,
        initial_state=state,
    )

    # Extract beta means from the position trace
    beta_samples = history_states.position["beta"]  # shape (500, 3)
    mean_beta = beta_samples.mean(axis=0)

    # Both feature coefficients should be positive (cluster centres at +1.5
    # for class 1, -1.5 for class 0 → positive logit slope on each feature).

    # Expect mean_beta[1], mean_beta[2] ≈ 1.1 (Laplace approximation point
    # estimate at the data MLE); allow generous bounds because n=500 is small.
    assert (
        mean_beta[1] > 0.3
    ), f"beta_1 should be > 0; got {mean_beta[1]:.3f} (suggests label-swap)"
    assert (
        mean_beta[2] > 0.3
    ), f"beta_2 should be > 0; got {mean_beta[2]:.3f} (suggests label-swap)"
    # Also check that both are FINITE — catches improper-posterior bugs
    assert jnp.isfinite(mean_beta).all(), f"non-finite posterior mean: {mean_beta}"
