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
"""Hierarchical radon model — 390-D NCP varying-intercept linear regression."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from bjx_bench.model._base import Posterior

__all__ = [
    "ENTRY",
    "FLOOR_X",
    "COUNTY_IDX",
    "LOG_URANIUM",
    "LOG_RADON",
    "N_COUNTIES",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Path to the committed, preprocessed CSV.
_CSV_PATH: Path = Path(__file__).parent.parent.parent / "data" / "radon.csv"

#: Number of counties (J = 386 for radon_all)
N_COUNTIES: int = 386

#: Unconstrained dimensionality:
#: alpha_raw (N_COUNTIES) + mu_a (1) + log_sigma_a (1) + beta (1) + log_sigma_y (1)
DIM: int = N_COUNTIES + 4  # = 390

# ---------------------------------------------------------------------------
# Load dataset from committed CSV (deterministic, no network call at runtime)
# ---------------------------------------------------------------------------
#
# Columns: county_idx (0-indexed), floor_measure, log_radon, log_uppm
# county_idx is converted from 1-indexed (posteriordb) to 0-indexed on write.

_raw = np.loadtxt(_CSV_PATH, delimiter=",", skiprows=1)  # shape (N, 4)

COUNTY_IDX: np.ndarray = _raw[:, 0].astype(np.int32)  # shape (N,), 0-indexed
FLOOR_X: jnp.ndarray = jnp.array(_raw[:, 1], dtype=jnp.float32)  # shape (N,)
LOG_RADON: jnp.ndarray = jnp.array(_raw[:, 2], dtype=jnp.float32)  # shape (N,)
LOG_URANIUM: jnp.ndarray = jnp.array(_raw[:, 3], dtype=jnp.float32)  # shape (N,)

# Verify county_idx bounds against expected J=386
_county_min = int(COUNTY_IDX.min())
_county_max = int(COUNTY_IDX.max())
assert _county_min == 0, f"Expected county_idx min=0, got {_county_min}"
assert (
    _county_max == N_COUNTIES - 1
), f"Expected county_idx max={N_COUNTIES - 1}, got {_county_max}"

# ---------------------------------------------------------------------------
# NumPyro model (NCP varying-intercept hierarchical radon)
# ---------------------------------------------------------------------------


def radon_hierarchical(
    floor_x: jnp.ndarray,
    county_idx: np.ndarray,
    log_radon: jnp.ndarray,
    n_counties: int = N_COUNTIES,
) -> None:
    """NumPyro model: 390-D NCP varying-intercept hierarchical radon.

    Non-centered parameterization of the varying-intercept hierarchical
    model from Gelman & Hill (2007) ch. 12. Posterior dim = 390 for
    radon_all (J=386 US counties, N=12573 observations).

    Parameters
    ----------
    floor_x
        Floor indicator (0=basement, 1=first floor, + rare codes) of shape (N,).
    county_idx
        0-indexed county identifiers of shape (N,).
    log_radon
        Log radon measurements of shape (N,).
    n_counties
        Number of counties J (386 for radon_all).

    Notes
    -----
    NCP structure: alpha_raw ~ N(0, 1)^J; alpha = mu_a + sigma_a * alpha_raw.
    This avoids the CP funnel geometry (sigma_a → 0 collapses alpha → mu_a).
    log_uranium is not used as a predictor in this model variant to match the
    variable-intercept-only posteriordb model (radon_variable_intercept_*).
    """
    # Hyperpriors on county intercepts
    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, 10.0))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(5.0))

    # NCP base variable: alpha_raw ~ N(0, 1)^J
    alpha_raw = numpyro.sample(
        "alpha_raw", dist.Normal(jnp.zeros(n_counties), jnp.ones(n_counties))
    )

    # Deterministic county intercepts (exposed for cross-check)
    alpha = numpyro.deterministic("alpha", mu_a + alpha_raw * sigma_a)

    # Floor slope (shared across counties)
    beta = numpyro.sample("beta", dist.Normal(0.0, 10.0))

    # Observation noise
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(5.0))

    # Likelihood
    mu = alpha[county_idx] + beta * floor_x
    numpyro.sample("log_radon", dist.Normal(mu, sigma_y), obs=log_radon)


# Statistician verdict (TL-orchestrated, 2026-05-08):
#     Approve-with-modifications. NCP varying-intercept hierarchical radon,
#     on the radon_all dataset (J=386 counties), 390-D total.
#     CRITICAL CORRECTION: plan said 'MN subset, ~170 dim'; this is wrong.
#     posteriordb uses radon_all (all US counties, J=386), and cross-checking
#     against a different dataset is indefensible. Use radon_all. Final
#     dim = 390 (= 386 alpha_raw + 1 mu_a + 1 log_sigma_a + 1 beta + 1 log_sigma_y; statistician initially said 391, arithmetic correction by TL).
#     WARN: Verify NCP log_sigma_a prior is not inadvertently centered.
#     WARN: If ESS for log_sigma_a < 200 after 1000 post-warmup, escalate n_warmup to 2000.
#     Parameterization (NCP varying-intercept, no varying-slopes):
#         mu_a       ~ Normal(0, 10)
#         log_sigma_a ~ HalfNormal(5)            # via NumPyro bijector on sigma_a
#         alpha_raw  ~ Normal(0, 1)^J             # NCP base variable, J=386
#         alpha      = mu_a + alpha_raw * sigma_a  (deterministic; exposed for cross-check)
#         beta       ~ Normal(0, 10)
#         log_sigma_y ~ HalfNormal(5)
#         mu         = alpha[county_idx] + beta * floor_measure[i]
#         y          ~ Normal(mu, sigma_y)
#     J=386 correction:
#         The original plan used 'MN subset (~170 dim)'. The posteriordb cross-check
#         posterior 'radon-radon_hierarchical_centered' uses radon_all (J=386 US
#         counties). We match radon_all exactly to enable valid cross-checking.
#     NCP vs posteriordb CP note:
#         This implementation uses NCP (non-centered parameterization). The posteriordb
#         reference posterior 'radon-radon_hierarchical_centered' uses CP (centered).
#         Cross-checking compares constrained-space marginal posteriors of alpha, beta,
#         mu_a, sigma_a, sigma_y — these quantities are parameterization-invariant.
#         The NCP-vs-CP difference affects sampling efficiency but not the target
#         distribution or the posterior marginals.
#     Unconstrained dimensionality:
#         alpha_raw (386) + mu_a (1) + log_sigma_a (1) + beta (1) + log_sigma_y (1) = 390.
#     Data provenance:
#         Source: posteriordb GitHub (stan-dev/posteriordb), dataset radon_all.
#         URL: https://raw.githubusercontent.com/stan-dev/posteriordb/master/
#              posterior_database/data/data/radon_all.json.zip
#         N=12573 observations, J=386 US counties.
#         CSV committed at bjx_bench/data/radon.csv (generated by tools/fetch_radon.py).
#         Fields: county_idx (0-indexed, 0..385), floor_measure (0/1 + rare codes),
#                 log_radon, log_uppm (log uranium ppm per observation).
#     Tier-A budget:
#         In-spawn verification: n_warmup=1000, n_samples=1000, 4 chains.
#         Production cache: n_warmup=1000, n_samples=20000.
#     Discrimination claim:
#         Defensible (Stan-window vs MEADS warmup) iff both use same NCP model +
#         same data + same post-warmup count, and metric is ESS/s.
#         Extends the hierarchical discrimination ladder:
#         eight_schools_ncp (10-D) → radon (390-D, real data).
# References:
#     Gelman, A. & Hill, J. (2007). Data Analysis Using Regression and
#     Multilevel/Hierarchical Models. Cambridge University Press. Ch. 12.
#     posteriordb: https://github.com/stan-dev/posteriordb
#     Dataset: radon_all (J=386 counties, N=12573 observations).
#     PLAN_bjx_bench_phase4.md § "Block C", row P4.7 (radon).
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="radon",
    dim=DIM,
    class_="hierarchical",
    tags=(
        "hierarchical",
        "regression",
        "ncp",
        "real_data",
        "varying_intercept",
        "high_dim",
    ),
    numpyro_model=radon_hierarchical,
    model_args=(FLOOR_X, COUNTY_IDX, LOG_RADON),
    model_kwargs={"n_counties": N_COUNTIES},
    posteriordb_id="radon-radon_hierarchical_centered",
    citations=(
        "Gelman, A. & Hill, J. (2007). Data Analysis Using Regression and "
        "Multilevel/Hierarchical Models. Cambridge University Press. Ch. 12.",
        "posteriordb: https://github.com/stan-dev/posteriordb (dataset: radon_all)",
    ),
    description=(
        "390-D NCP varying-intercept hierarchical radon model "
        "(radon_all: J=386 US counties, N=12573 observations). "
        "Priors: mu_a~N(0,10), sigma_a~HN(5), alpha_raw~N(0,1)^386, "
        "beta~N(0,10), sigma_y~HN(5). "
        "NCP on county intercepts (alpha = mu_a + sigma_a * alpha_raw). "
        "Cross-check against posteriordb radon_all CP reference on "
        "constrained-space marginals (parameterization-invariant). "
        "Gelman & Hill (2007) ch. 12."
    ),
)
