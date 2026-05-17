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
"""IRT 2PL model — 144-D NCP hierarchical psychometric model (J=100 students, I=20 items)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = [
    "ENTRY",
    "RESPONSE",
    "N_STUDENTS",
    "N_ITEMS",
    "DIM",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Path to the committed, preprocessed CSV.
_CSV_PATH: Path = Path(__file__).parent.parent / "data" / "irt_2pl.csv"

#: Number of students (J = 100)
N_STUDENTS: int = 100

#: Number of items (I = 20)
N_ITEMS: int = 20

#: Unconstrained dimensionality:
#: sigma_theta(1) + theta_raw(N_STUDENTS) + sigma_a(1) + log_a_raw(N_ITEMS)
#: + mu_b(1) + sigma_b(1) + b_raw(N_ITEMS) = 144
DIM: int = 1 + N_STUDENTS + 1 + N_ITEMS + 1 + 1 + N_ITEMS  # = 144

# ---------------------------------------------------------------------------
# Load dataset from committed CSV (deterministic, no network call at runtime)
# ---------------------------------------------------------------------------
#
# Columns: student_id (0-indexed, 0..99), item_id (0-indexed, 0..19), response (0/1)
# We load and reshape into a (J, I) 2-D array RESPONSE[j, i].

_raw = np.loadtxt(_CSV_PATH, delimiter=",", skiprows=1)  # shape (2000, 3)
_student_ids = _raw[:, 0].astype(np.int32)
_item_ids = _raw[:, 1].astype(np.int32)
_responses_flat = _raw[:, 2].astype(np.float32)

# Reshape to (J, I) matrix
_response_matrix = np.zeros((N_STUDENTS, N_ITEMS), dtype=np.float32)
_response_matrix[_student_ids, _item_ids] = _responses_flat

#: Binary response matrix of shape (J=100, I=20). RESPONSE[j, i] = 1 if student j
#: answered item i correctly, 0 otherwise.
RESPONSE: jnp.ndarray = jnp.array(_response_matrix)

# Validate shapes
assert _raw.shape == (
    N_STUDENTS * N_ITEMS,
    3,
), f"Expected {N_STUDENTS * N_ITEMS} rows, got {_raw.shape[0]}"
assert RESPONSE.shape == (
    N_STUDENTS,
    N_ITEMS,
), f"Expected RESPONSE shape ({N_STUDENTS}, {N_ITEMS}), got {RESPONSE.shape}"
assert set(np.unique(_responses_flat).tolist()) <= {
    0.0,
    1.0,
}, f"Response values must be binary {{0, 1}}, got {set(np.unique(_responses_flat).tolist())}"

# ---------------------------------------------------------------------------
# NumPyro model (NCP IRT 2PL)
# ---------------------------------------------------------------------------


def irt_2pl(
    response: jnp.ndarray,
    n_students: int = N_STUDENTS,
    n_items: int = N_ITEMS,
) -> None:
    """NumPyro model: 144-D NCP 2-parameter logistic IRT.

    Non-centered parameterization on all three hierarchical groups (theta/b/log_a)
    to avoid geometry funnels when hyperparameters approach zero. Posterior
    dim = 144 for the irt_2pl dataset (J=100 students, I=20 items).

    Parameters
    ----------
    response
        Binary response matrix of shape (J, I). ``response[j, i]`` is 1 if
        student j answered item i correctly, 0 otherwise.
    n_students
        Number of students J (100 for irt_2pl dataset).
    n_items
        Number of items I (20 for irt_2pl dataset).

    Notes
    -----
    NCP structure matches the Stan posteriordb model (sigma_theta is a free
    parameter, not fixed at 1). Three funnels are suppressed via NCP:

    - Student abilities: theta_raw ~ N(0,1)^J; theta = sigma_theta * theta_raw.
    - Item difficulties: b_raw ~ N(0,1)^I; b = mu_b + sigma_b * b_raw.
    - Item discriminations: log_a_raw ~ N(0,1)^I; log_a = sigma_a * log_a_raw;
      a = exp(log_a) (positive). Equivalent to Stan's a ~ lognormal(0, sigma_a).

    Likelihood: y[j,i] ~ Bernoulli(logit = a[i] * (theta[j] - b[i])).
    """
    # --- student ability hyperprior ---
    sigma_theta = numpyro.sample("sigma_theta", dist.HalfCauchy(2.0))

    # NCP student abilities
    theta_raw = numpyro.sample(
        "theta_raw", dist.Normal(jnp.zeros(n_students), jnp.ones(n_students))
    )
    theta = numpyro.deterministic("theta", sigma_theta * theta_raw)  # shape (J,)

    # --- item difficulty hyperpriors ---
    mu_b = numpyro.sample("mu_b", dist.Normal(0.0, 5.0))
    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(2.0))

    # NCP item difficulties
    b_raw = numpyro.sample("b_raw", dist.Normal(jnp.zeros(n_items), jnp.ones(n_items)))
    b = numpyro.deterministic("b", mu_b + sigma_b * b_raw)  # shape (I,)

    # --- item discrimination hyperprior ---
    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(2.0))

    # NCP item discriminations (lognormal via NCP)
    log_a_raw = numpyro.sample(
        "log_a_raw", dist.Normal(jnp.zeros(n_items), jnp.ones(n_items))
    )
    log_a = numpyro.deterministic("log_a", sigma_a * log_a_raw)  # shape (I,)
    a = numpyro.deterministic("a", jnp.exp(log_a))  # shape (I,), positive

    # --- likelihood ---
    # logits[j, i] = a[i] * (theta[j] - b[i])
    logits = a[None, :] * (theta[:, None] - b[None, :])  # shape (J, I)
    numpyro.sample("response", dist.Bernoulli(logits=logits), obs=response)


# Statistician verdict (TL-orchestrated, 2026-05-08):
#     Approve-with-modifications. Use Long-NUTS self-check only (split-R̂ gate).
#     CRITICAL CORRECTION 1: plan said dim ~230; correct dim = 144.
#         Decomposition: sigma_theta(1) + theta_raw(100) + sigma_a(1)
#         + log_a_raw(20) + mu_b(1) + sigma_b(1) + b_raw(20) = 144.
#     CRITICAL CORRECTION 2: posteriordb_id = None. The posteriordb posterior
#         'irt_2pl-irt_2pl' has reference_posterior_name: null (no reference
#         draws). Cross-checking against Stan reference is impossible.
#     CRITICAL CORRECTION 3: Stan model uses sigma_theta ~ Cauchy(0, 2) as a
#         FREE parameter (not fixed at 1). Match this for consistency.
#     NCP on all three hierarchical groups (theta/b/log_a) eliminates three
#     potential geometry funnels (sigma_theta, sigma_b, sigma_a → 0).
#     Parameterization (NCP on all three hierarchical groups):
#         sigma_theta ~ HalfCauchy(2.0)
#         theta_raw   ~ Normal(0, 1)^J         J=100 students
#         theta       = sigma_theta * theta_raw  (deterministic)
#         mu_b        ~ Normal(0, 5)
#         sigma_b     ~ HalfCauchy(2.0)
#         b_raw       ~ Normal(0, 1)^I         I=20 items
#         b           = mu_b + sigma_b * b_raw  (deterministic)
#         sigma_a     ~ HalfCauchy(2.0)
#         log_a_raw   ~ Normal(0, 1)^I
#         log_a       = sigma_a * log_a_raw     (Stan: a ~ lognormal(0, sigma_a))
#         a           = exp(log_a)              (deterministic, positive)
#         logits = a[None, :] * (theta[:, None] - b[None, :])   # shape (J, I)
#         y      ~ Bernoulli(logits=logits)
#     Unconstrained dimensionality:
#         sigma_theta(1) + theta_raw(100) + sigma_a(1) + log_a_raw(20)
#         + mu_b(1) + sigma_b(1) + b_raw(20) = 144.
#         (plan had ~230 — corrected by statistician).
#     Data: posteriordb dataset 'irt_2pl' (J=100 students, I=20 items, 2000 responses).
#         Source: https://raw.githubusercontent.com/stan-dev/posteriordb/master/
#                 posterior_database/data/data/irt_2pl.json.zip
#         Generated via tools/fetch_irt_2pl.py; committed as tuningfork/data/irt_2pl.csv
#     posteriordb_id = None:
#         'irt_2pl-irt_2pl' has reference_posterior_name: null (no reference draws).
#         reference-certification uses Long-NUTS self-check (split-R̂ < 1.01) only; no xcheck.
#     Stan model reference:
#         https://raw.githubusercontent.com/stan-dev/posteriordb/master/
#         posterior_database/models/stan/irt_2pl.stan
#     reference-certification budget:
#         In-spawn verification: n_warmup=1000, n_samples=2000, 4 chains.
#         Production cache: n_warmup=2000, n_samples=20000.
#     Discrimination claim:
#         Defensible (CP vs NCP genuine — 3 free sigma hyperparameters → 3 funnels;
#         Pathfinder vs window-adaptation also defensible at 144-D).
#         Extends the hierarchical ladder: eight_schools_ncp (10-D) → radon (390-D)
#         → irt_2pl (144-D, distinct psychometric structure).
# References:
#     Stan User's Guide § 1.11 "Item Response Theory Models".
#     Baker, F.B. & Kim, S.-H. (2004). Item Response Theory: Parameter
#         Estimation Techniques (2nd ed.). Marcel Dekker.
#     posteriordb: https://github.com/stan-dev/posteriordb (dataset: irt_2pl).
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="irt_2pl",
    dim=DIM,
    class_="hierarchical",
    tags=(
        "hierarchical",
        "psychometric",
        "ncp",
        "scale_identifiable",
        "real_data",
    ),
    numpyro_model=irt_2pl,
    model_args=(RESPONSE,),
    model_kwargs={"n_students": N_STUDENTS, "n_items": N_ITEMS},
    posteriordb_id=None,  # irt_2pl-irt_2pl has no reference posterior draws upstream
    citations=(
        "Stan User's Guide § 1.11 'Item Response Theory Models'.",
        "Baker, F.B. & Kim, S.-H. (2004). Item Response Theory: Parameter "
        "Estimation Techniques (2nd ed.). Marcel Dekker.",
        "posteriordb: https://github.com/stan-dev/posteriordb (dataset: irt_2pl)",
    ),
    description=(
        "144-D NCP 2-parameter logistic IRT model "
        "(irt_2pl: J=100 students, I=20 items, 2000 binary responses). "
        "Priors: sigma_theta~HC(2), theta_raw~N(0,1)^100, "
        "mu_b~N(0,5), sigma_b~HC(2), b_raw~N(0,1)^20, "
        "sigma_a~HC(2), log_a_raw~N(0,1)^20. "
        "NCP on all three hierarchical groups (theta, b, log_a). "
        "posteriordb_id=None (no reference draws; Long-NUTS self-check only). "
        "Plan dim ~230 corrected to 144 by statistician (2026-05-08)."
    ),
)
