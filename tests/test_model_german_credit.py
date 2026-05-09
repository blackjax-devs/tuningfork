"""Tests for the german_credit model (26-D Bayesian logistic regression, real UCI data).

Tests
-----
1.  ENTRY registered in MODELS with name "german_credit".
2.  dim == 26, class_ == "glm", tags contain the required labels.
3.  Data loaded from CSV; shape (1000, 25); y is binary; X is standardized.
4.  Data is deterministic across imports.
5.  build_logdensity_fn returns finite log-density at zero-init position.
6.  Log-density at a near-MLE point (sklearn LogisticRegression estimate) is
    higher than at the zero-init position.
7.  Posterior dim matches parameter count after build_logdensity_fn (26).
8.  Sampling test: 500-sample NUTS smoke; ≥70% of beta coefficients have the
    same sign as the sklearn-LogisticRegression MLE (catches label-swap /
    likelihood misspecification bugs — per P4.4 retro-checkpoint precedent).

Notes
-----
- Tests 1–7 are @pytest.mark.fast (no MCMC).
- Test 8 (NUTS smoke) is NOT @pytest.mark.fast — runs MCMC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bjx_bench.model import MODELS, build_logdensity_fn
from bjx_bench.model.glm.german_credit import DIM, ENTRY, X_DATA, Y_DATA

pytestmark = pytest.mark.fast  # tests 1–7 are fast; test 8 has its own decorator

# ---------------------------------------------------------------------------
# Test 1: registration
# ---------------------------------------------------------------------------


def test_registered_in_models() -> None:
    """german_credit must appear in MODELS under the correct name."""
    assert "german_credit" in MODELS, "german_credit not found in MODELS"
    assert MODELS["german_credit"] is ENTRY


# ---------------------------------------------------------------------------
# Test 2: schema — dim, class_, tags
# ---------------------------------------------------------------------------


def test_dim_class_tags() -> None:
    """Posterior dim must be 26, class_ 'glm', tags include required labels."""
    assert ENTRY.dim == DIM
    assert ENTRY.dim == 26
    assert ENTRY.class_ == "glm"
    assert "glm" in ENTRY.tags
    assert "logistic-regression" in ENTRY.tags
    assert "real-data" in ENTRY.tags
    assert "ill-conditioned" in ENTRY.tags
    assert "medium-dim" in ENTRY.tags


# ---------------------------------------------------------------------------
# Test 3: data shape, y binary, X standardized
# ---------------------------------------------------------------------------


def test_data_shape_and_properties() -> None:
    """X_DATA shape (1000, 25); Y_DATA binary; X standardized to ~0 mean / ~1 std."""
    assert X_DATA.shape == (1000, 25), f"Expected (1000, 25), got {X_DATA.shape}"
    assert Y_DATA.shape == (1000,), f"Expected (1000,), got {Y_DATA.shape}"

    # y is binary
    y_np = np.asarray(Y_DATA)
    unique_vals = set(y_np.astype(int).tolist())
    assert unique_vals == {0, 1}, f"Y_DATA contains non-binary values: {unique_vals}"

    # Class balance: ~70/30 (good / bad)
    pct_good = float(y_np.mean())
    assert (
        0.65 < pct_good < 0.75
    ), f"Expected ~70% good credit, got {pct_good * 100:.1f}%"

    # X is standardized: column means ≈ 0, stds ≈ 1
    x_np = np.asarray(X_DATA)
    col_means = x_np.mean(axis=0)
    col_stds = x_np.std(axis=0)
    assert (
        np.abs(col_means).max() < 1e-5
    ), f"X_DATA column means not ~0: max |mean| = {np.abs(col_means).max():.2e}"
    # stds should be near 1 for continuous columns; binary-encoded dummies
    # have stds in [0.3, 0.5] — so we check > 0.2 and < 1.1 instead of strict ~1.
    assert col_stds.min() > 0.2, f"X_DATA col std below 0.2: min = {col_stds.min():.4f}"
    assert col_stds.max() < 1.1, f"X_DATA col std above 1.1: max = {col_stds.max():.4f}"


# ---------------------------------------------------------------------------
# Test 4: determinism — X_DATA and Y_DATA are stable across imports
# ---------------------------------------------------------------------------


def test_data_is_deterministic() -> None:
    """X_DATA and Y_DATA must be deterministic: same values on every import."""
    from bjx_bench.model.glm.german_credit import X_DATA as x2
    from bjx_bench.model.glm.german_credit import Y_DATA as y2

    assert np.array_equal(np.asarray(X_DATA), np.asarray(x2)), "X_DATA not stable"
    assert np.array_equal(np.asarray(Y_DATA), np.asarray(y2)), "Y_DATA not stable"


# ---------------------------------------------------------------------------
# Test 5: build_logdensity_fn returns finite log-density at zero-init
# ---------------------------------------------------------------------------


def test_build_logdensity_fn_finite() -> None:
    """build_logdensity_fn must return a finite log-density at the init position."""
    key = jax.random.key(0)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)
    ld = logdensity_fn(init_pos)
    assert jnp.isfinite(ld), f"Expected finite log-density at init, got {ld}"


# ---------------------------------------------------------------------------
# Test 6: log-density is higher at near-MLE than at zero-init
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_logdensity_higher_at_mle() -> None:
    """Log-density at a near-MAP point (sklearn MLE) must exceed log-density at zero-init.

    Uses sklearn LogisticRegression to find a quick MLE estimate; constructs a
    beta dict in unconstrained space (no bijector for N(0,5) priors) and compares.
    """
    from sklearn.linear_model import LogisticRegression

    # Quick MLE estimate (no regularization to get close to the Bayesian MAP)
    lr = LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")
    lr.fit(np.asarray(X_DATA), np.asarray(Y_DATA).astype(int))

    # sklearn gives intercept_ and coef_ (shape 1 × n_features for binary)
    intercept = float(lr.intercept_[0])
    coef = lr.coef_[0]  # shape (25,)
    beta_mle = np.concatenate([[intercept], coef]).astype(np.float32)

    key = jax.random.key(1)
    init_pos, logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

    ld_init = float(logdensity_fn(init_pos))

    mle_pos = {"beta": jnp.array(beta_mle)}
    ld_mle = float(logdensity_fn(mle_pos))

    assert ld_mle > ld_init, (
        f"Log-density at near-MLE ({ld_mle:.3f}) must exceed "
        f"log-density at zero-init ({ld_init:.3f})."
    )


# ---------------------------------------------------------------------------
# Test 7: posterior dim matches parameter count from build_logdensity_fn
# ---------------------------------------------------------------------------


def test_posterior_dim_matches_params() -> None:
    """Posterior dim == 26 must match the number of unconstrained parameters."""
    key = jax.random.key(2)
    init_pos, _logdensity_fn, _ = build_logdensity_fn(key, ENTRY)

    total_params = sum(v.size for v in jax.tree.leaves(init_pos))
    assert total_params == ENTRY.dim, (
        f"Posterior dim {ENTRY.dim} != parameter count {total_params} "
        "from build_logdensity_fn."
    )


# ---------------------------------------------------------------------------
# Test 8: sampling smoke — ≥70% sign agreement with sklearn MLE
# NOT @pytest.mark.fast — runs MCMC.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_posterior_signs_via_short_nuts() -> None:
    """500-sample NUTS smoke: at least 70% of beta coefficients should have
    signs matching the sklearn-LogisticRegression MLE. Catches label-swap and
    likelihood misspecification bugs.

    Per P4.4 retro-checkpoint precedent (statistician review 2026-05-08):
    this sampling test catches bugs that structural tests (tests 1-7) cannot.
    NOT marked @pytest.mark.fast — runs MCMC.
    """
    import blackjax
    from sklearn.linear_model import LogisticRegression

    # Quick sklearn MLE for sign reference
    lr = LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")
    lr.fit(np.asarray(X_DATA), np.asarray(Y_DATA).astype(int))
    beta_mle = np.concatenate([lr.intercept_, lr.coef_[0]])  # shape (26,)

    rng_key = jax.random.key(0)
    init_position, logdensity_fn, _ = build_logdensity_fn(rng_key, ENTRY)

    # Short NUTS warmup
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, target_acceptance_rate=0.80
    )
    (state, params), _ = warmup.run(rng_key, init_position, num_steps=500)

    # 500 post-warmup samples
    nuts = blackjax.nuts(logdensity_fn, **params)
    _, (history_states, _) = blackjax.util.run_inference_algorithm(
        rng_key=jax.random.key(1),
        inference_algorithm=nuts,
        num_steps=500,
        initial_state=state,
    )

    # Extract beta means from the position trace; shape (500, 26)
    beta_samples = history_states.position["beta"]
    mean_beta = np.asarray(beta_samples.mean(axis=0))  # shape (26,)

    # Sign agreement: fraction of coefficients where sign matches MLE
    # (excludes the intercept, index 0, which can be negative due to
    # 70/30 imbalance and prior centering — focus on the feature coefs)
    signs_mle = np.sign(beta_mle[1:])  # shape (25,) feature coefs
    signs_post = np.sign(mean_beta[1:])

    # Ignore components where MLE is near 0 (ambiguous sign)
    non_trivial = np.abs(beta_mle[1:]) > 0.05
    if non_trivial.sum() > 0:
        agreement = float((signs_mle[non_trivial] == signs_post[non_trivial]).mean())
    else:
        agreement = 1.0  # no non-trivial coefs → trivially pass

    assert agreement >= 0.70, (
        f"Only {agreement * 100:.1f}% of feature coefs have signs matching sklearn MLE "
        f"(expected ≥70%). Possible label-swap or likelihood misspecification. "
        f"MLE signs: {signs_mle.tolist()}, Posterior signs: {signs_post.tolist()}"
    )

    # Also verify finite posterior mean
    assert np.isfinite(mean_beta).all(), f"Non-finite posterior mean beta: {mean_beta}"
