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
"""Stochastic volatility model — 503-D NCP recursive AR(1) (Kim-Shephard-Chib 1998).

Data: first 500 mean-centered daily returns of the S&P 500 (real financial
data), matching the canonical Stan/NumPyro stoch_vol example data class.
Source: ``numpyro.examples.datasets.SP500``. Replaced 2026-05-12 from
the prior synthetic (KSC truth parameters) version per the user direction
to match Stan's setup more fully.

Priors match the Stan User's Guide § 2.5 NCP form (primary / generic AR(1)):
    mu    ~ Cauchy(0, 10)
    phi   ~ Uniform(-1, 1)
    sigma ~ HalfCauchy(5)
    h_std ~ Normal(0, 1) i.i.d.
"""

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

__all__ = [
    "ENTRY",
    "RETURNS",
    "T_LENGTH",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Number of time steps (T = 500 — first 500 daily returns of SP500)
T_LENGTH: int = 500

#: Unconstrained dimensionality: mu(1) + phi(1) + log_sigma(1) + h_raw(500)
DIM: int = 3 + T_LENGTH  # = 503

#: Path to committed CSV data file (SP500 first 500 mean-centered daily returns).
#: Single column `returns`; 500 rows.
_CSV_PATH: Path = Path(__file__).parent.parent / "data" / "stoch_vol_returns.csv"

# ---------------------------------------------------------------------------
# Load real SP500 returns (shape: (500,), float32, mean-centered)
# ---------------------------------------------------------------------------


def _load_returns(path: Path) -> np.ndarray:
    """Load a single-column 'returns' CSV (header on first line)."""
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["returns"], f"Expected header ['returns'], got {header}"
        return np.array([float(row[0]) for row in reader], dtype=np.float32)


RETURNS: jnp.ndarray = jnp.array(_load_returns(_CSV_PATH), dtype=jnp.float32)

# Validate shape
assert RETURNS.shape == (
    T_LENGTH,
), f"Expected RETURNS shape ({T_LENGTH},), got {RETURNS.shape}"

# ---------------------------------------------------------------------------
# NumPyro model (NCP recursive AR(1) stochastic volatility)
# ---------------------------------------------------------------------------


def stoch_vol_model(returns: jnp.ndarray, T: int = T_LENGTH) -> None:
    """NumPyro model: 503-D NCP recursive AR(1) stochastic volatility.

    Kim-Shephard-Chib (1998) stochastic volatility model with non-centered
    parameterization and recursive AR(1) latent log-volatility via
    ``jax.lax.scan``. Posterior dim = 503 for T=500.

    Parameters
    ----------
    returns
        Observed returns array of shape (T,).
    T
        Number of time steps (500 — first 500 SP500 daily returns,
        mean-centered).

    Priors (Stan User's Guide § 2.5 canonical NCP form, primary variant):
        mu       ~ Cauchy(0, 10)             # wide, no scale assumption
        phi      ~ Uniform(-1, 1)            # generic AR(1) stationarity
        sigma    ~ HalfCauchy(5)             # wide positive-real
        h_std    ~ Normal(0, 1) i.i.d.       # NCP standardised innovations

    History note (2026-05-12): an experiment briefly swapped phi to Stan's
    *daily-financial-vol* alternative ``2·Beta(20, 1.5) - 1`` after a
    divergence-cluster analysis suggested the chain was visiting the
    unit-root region (constrained phi ≈ 0.9999) under Uniform. The swap
    made cert WORSE (362 divs / 0.91% vs 105 / 0.26% under Uniform) because
    the Beta-shifted prior concentrates mass at high persistence (near the
    sharp-geometry region) rather than suppressing trips to it. Reverted
    same day. Lesson: the cluster diagnosis was correct (divergences
    cluster at extreme phi); the proposed fix was not. The remaining
    divergence rate is a *structural* feature of the diagonal-IMM NUTS on
    a 503-D AR(1) state space — see worklog thread
    ``phase0-statistician-es-ncp-stoch-vol.md`` for the full diagnosis.

    Data: real SP500 daily returns from ``numpyro.examples.datasets.SP500``,
    first 500 entries, mean-centered. Replaces the prior synthetic data
    (KSC ``MU_TRUE=-10, PHI_TRUE=0.95, SIGMA_TRUE=0.25``) per user direction
    2026-05-12: real data is preferred for matching Stan's reference setup.
    No analytic posterior available, so cross-checking moments uses Stan
    reference samples (when available via posteriordb_xcheck) or a
    long-NUTS reference run as ground truth.

    Latent log-volatility is initialised at the stationary distribution:
        h[0] = mu + (sigma / sqrt(1 - phi^2)) * h_std[0]

    Subsequent steps follow the AR(1) recursion:
        h[t] = mu + phi * (h[t-1] - mu) + sigma * h_std[t]

    Likelihood: returns[t] ~ Normal(0, exp(h[t] / 2)).
    """
    # Stan-canonical priors (Stan User's Guide § 2.5, primary form).
    # See docstring "History note" for the briefly-tested Beta(20,1.5)-shifted
    # variant that was reverted on 2026-05-12.
    mu = numpyro.sample("mu", dist.Cauchy(0.0, 10.0))
    phi = numpyro.sample("phi", dist.Uniform(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))

    # NCP latent innovations (renamed h_raw → h_std to match Stan's naming).
    # h_raw kept as the NumPyro sample name for backwards-compat with prior cache;
    # the variable name in code is h_std for readability.
    h_std = numpyro.sample("h_raw", dist.Normal(jnp.zeros(T), 1.0))

    def step(h_prev, h_std_t):
        h_t = mu + phi * (h_prev - mu) + sigma * h_std_t
        return h_t, h_t

    h0 = mu + (sigma / jnp.sqrt(1.0 - phi**2)) * h_std[0]
    _, h_rest = jax.lax.scan(step, h0, h_std[1:])
    h = jnp.concatenate([h0[None], h_rest])
    h = numpyro.deterministic("h", h)

    numpyro.sample("returns", dist.Normal(0.0, jnp.exp(h / 2.0)), obs=returns)


# Statistician verdict (TL-orchestrated, 2026-05-08):
#     Approve-with-modifications. NCP-recursive AR(1) via lax.scan;
#     phi Beta(20,1.5)-shifted; posteriordb_id=None.
#     CRITICAL CORRECTION 1: No posteriordb cross-check available.
#         Full posteriordb directory scan confirms no stochastic-volatility
#         posterior with reference draws exists upstream (only GARCH(1,1),
#         which is a different model class). Set posteriordb_id=None. Use
#         Long-NUTS self-check only (split-R̂ gate).
#     CRITICAL CORRECTION 2: phi prior must be Beta(20, 1.5) shifted to (-1, 1),
#         NOT Uniform(-1, 1). Uniform places mass near the unit-root boundary
#         causing sigma/sqrt(1-phi^2) geometry blowup. Beta(20, 1.5) shifted
#         concentrates mass at phi in [0.87, 0.99] (Kim-Shephard-Chib 1998
#         canonical prior for daily financial vol).
#     CRITICAL CORRECTION 3: T = 500 exactly. Dim = 503
#         = mu(1) + phi(1) + log_sigma(1) + h_raw(500).
#     CRITICAL CORRECTION 4: Recursive NCP via jax.lax.scan is correct —
#         algebraically equivalent to Stan's transformed_parameters block.
#     Parameterization (NCP recursive AR(1)):
#         mu       ~ Normal(-10, 5)                      # mean log-volatility
#         phi      ~ 2 * Beta(20, 1.5) - 1               # via TransformedDistribution
#         sigma    ~ HalfNormal(0.5)                     # innovation scale
#         h_raw    ~ Normal(0, 1)^T                      # NCP innovations
#         h[0]  = mu + (sigma / sqrt(1 - phi^2)) * h_raw[0]   # stationary init
#         h[t]  = mu + phi * (h[t-1] - mu) + sigma * h_raw[t]  # t >= 1
#         returns[t] ~ Normal(0, exp(h[t] / 2))
#     Unconstrained dimensionality = 503:
#         mu(1) + phi(1, softplus-warped in (-1,1)) + log_sigma(1) + h_raw(500)
#     Data: synthetic, generated by tools/generate_stoch_vol.py.
#         Parameters: mu=-10.0, phi=0.95, sigma=0.25, T=500, seed=20260508.
#         Committed as tuningfork/data/stoch_vol_returns.npy (float32, shape (500,)).
#     posteriordb_id = None:
#         No upstream stochastic-volatility posterior with reference draws exists.
#         reference-certification uses Long-NUTS self-check (split-R̂ < 1.01) only; no xcheck.
#     reference-certification budget:
#         In-spawn verification: n_warmup=2000, n_samples=20000, 4 chains.
#         Production cache: n_warmup=5000, n_samples=100000.
#         (Escalated from radon's 1k/1k due to AR(1) autocorrelation; phi=0.95
#         implies effective memory ~20 steps.)
#     Discrimination claim:
#         NUTS vs MCLMC at 503-D AR(1). AR(1) creates tridiagonal precision in
#         the latent space; MCLMC's continuous momentum may handle this more
#         efficiently than diagonal-mass NUTS. Frame as "may favor MCLMC for
#         long-memory AR chains" rather than guaranteed win. E-BFMI gate
#         informative; if <0.3, may need to tighten phi prior further.
# References:
#     Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility:
#         Likelihood Inference and Comparison with ARCH Models. Review of
#         Economic Studies, 65(3), 361–393.
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="stoch_vol",
    dim=DIM,
    class_="latent_gaussian",
    tags=(
        "latent_gaussian",
        "state_space",
        "ar1",
        "ncp",
        "synthetic",
    ),
    numpyro_model=stoch_vol_model,
    model_args=(RETURNS,),
    model_kwargs={"T": T_LENGTH},
    posteriordb_id=None,  # no upstream SV posterior with reference draws
    citations=(
        "Kim, S., Shephard, N., & Chib, S. (1998). Stochastic Volatility: "
        "Likelihood Inference and Comparison with ARCH Models. "
        "Review of Economic Studies, 65(3), 361–393.",
    ),
    description=(
        "503-D NCP recursive AR(1) stochastic volatility (KSC 1998). "
        "T=500 real SP500 mean-centered daily returns (numpyro.examples.datasets.SP500, "
        "first 500 entries; CSV at data/stoch_vol_returns.csv). "
        "Stan User's Guide § 2.5 priors (primary form): "
        "mu~Cauchy(0,10), phi~Uniform(-1,1), sigma~HalfCauchy(5), h_raw~N(0,1)^500 (NCP). "
        "posteriordb_id=None (no upstream reference draws; Long-NUTS self-check). "
        "Dim=503: mu(1)+phi(1)+log_sigma(1)+h_raw(500)."
    ),
    # Per-model divergence-rate override: 0.005 (= 0.5%, vs the global 0.1%).
    # Justification (TL ↔ user, 2026-05-12): under the canonical NUTS + stan_window
    # cert (n_warmup=5000, n_samples=40000, ta=0.99, max_num_doublings=15, seed=42),
    # stoch_vol produces ~105 divergences (0.26%) with R̂=1.0006, min-bulk-ESS=1992,
    # E-BFMI=0.93 — sampling is correct; the residual divergences cluster at extreme
    # phi (constrained ≈ 0.9999, the AR(1) unit root) where sigma²/(1-phi²) blows up
    # the stationary-init geometry. Per the model-freeze-post-groundtruth policy,
    # the right response is gate relaxation backed by the cluster diagnosis, NOT a
    # prior swap (the Beta(20,1.5)-shifted alternative was tried 2026-05-12 and made
    # cert WORSE, see worklog/threads/phase0-statistician-es-ncp-stoch-vol.md).
    # MCLMC (Recipe Phase 2) remains the principled long-term fix; this override
    # closes Phase 0 in the meantime.
    divergence_rate_tolerance=0.005,
)
