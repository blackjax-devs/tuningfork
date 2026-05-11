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
"""Bayesian logistic regression on German Credit — 26-D (1000 borrowers, 25 standardized features)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = ["ENTRY", "X_DATA", "Y_DATA"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Path to the committed, preprocessed CSV.
_CSV_PATH: Path = Path(__file__).parent.parent.parent / "data" / "german_credit.csv"

# Intercept + 25 feature coefficients = 26 unconstrained parameters.
DIM = 26

# ---------------------------------------------------------------------------
# Load dataset from committed CSV (deterministic, no network call)
# ---------------------------------------------------------------------------

_raw = np.loadtxt(_CSV_PATH, delimiter=",", skiprows=1)  # shape (1000, 26)

Y_DATA: jnp.ndarray = jnp.array(_raw[:, 0], dtype=jnp.float32)  # shape (1000,)
X_DATA: jnp.ndarray = jnp.array(_raw[:, 1:], dtype=jnp.float32)  # shape (1000, 25)

# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------


def _model(X: jnp.ndarray, y: jnp.ndarray) -> None:
    """NumPyro model: 26-D logistic regression on German Credit.

    Parameters
    ----------
    X
        Standardized feature matrix of shape (n, 25).
    y
        Binary response of shape (n,); 1 = good credit, 0 = bad.
    """
    n_features = X.shape[1]
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(n_features + 1), 5.0))
    logits = beta[0] + X @ beta[1:]
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)


# Design notes (German Credit is a follow-on GLM variant):
#     - German Credit follows the same prior structure as logistic_synthetic.
#     - n_warmup=2000 for in-spawn reference-certification (higher than logistic_synthetic's 1000
#       because dim is ~8× larger; 25-D posterior needs more warmup for window
#       adaptation to converge; in practice 2000 steps suffices per a-priori
#       expectation for a well-conditioned logistic regression at this scale).
#     Preprocessing provenance (deterministic, bundled in bjx_bench/data/german_credit.csv):
#         Source: UCI ML Repository German Credit dataset (scikit-learn fetch_openml
#         name='credit-g', version=1, 1000 borrowers × 20 original attributes).
#         Feature selection (hits ~25-D posterior target):
#             Numerical (7): duration, credit_amount, installment_commitment,
#                 residence_since, age, existing_credits, num_dependents.
#             Categorical, one-hot encoded with drop_first=True:
#                 checking_status (4 levels → 3 dummies)
#                 credit_history  (5 levels → 4 dummies)
#                 savings_status  (5 levels → 4 dummies)
#                 employment      (5 levels → 4 dummies)
#                 personal_status (4 levels → 3 dummies)
#             Total dummies: 18.
#             Total features: 7 + 18 = 25 → posterior dim = 26 (intercept + 25 coefs).
#         Standardization: sklearn StandardScaler fit on the full 1000-row set
#             (zero mean, unit std). Binary target: 1 = good credit, 0 = bad credit
#             (70% / 30% class balance).
#         The CSV was generated via tools/prepare_german_credit.py and is committed at
#         bjx_bench/data/german_credit.csv.
#     Model description:
#         - Data: 1000 borrowers, 25 standardized features (7 numerical +
#           18 categorical dummies), binary target (good / bad credit).
#         - Likelihood: y_i ~ Bernoulli(sigmoid(beta_0 + X_i @ beta[1:])).
#         - Priors: beta_k ~ N(0, 5) for k = 0 .. 25 (weakly informative,
#           matching logistic_synthetic). 5-unit std is wide on standardized
#           features; the data are informative enough to dominate the prior.
#         - Posterior dim = 26 (unconstrained; no bijector transforms needed).
#     Discrimination claim:
#         HMC vs MALA vs RWM on a 26-D near-Gaussian posterior with mild
#         ill-conditioning from mixed numerical / binary dummy-encoded features.
#         Real-data class imbalance (70/30) and collinearity between categorical
#         dummies provides more realistic geometry than logistic_synthetic.
#         Extends the GLM discrimination ladder: logistic_synthetic (3-D baseline)
#         → german_credit (26-D, real data) → horseshoe (103-D, sparse).
# References:
#     UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
#     Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="german_credit",
    dim=DIM,
    class_="glm",
    tags=(
        "glm",
        "logistic-regression",
        "real-data",
        "ill-conditioned",
        "medium-dim",
    ),
    numpyro_model=_model,
    model_args=(X_DATA, Y_DATA),
    description=(
        "26-D Bayesian logistic regression on the German Credit dataset "
        "(UCI ML, 1000 borrowers; 7 numerical + 18 categorical dummies, "
        "standardized; 70/30 class balance). "
        "Priors: beta_k ~ N(0, 5). "
        "Discriminates HMC vs MALA vs RWM at medium dimensionality with "
        "real-data ill-conditioning."
    ),
)
