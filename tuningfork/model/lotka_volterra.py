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
"""Lotka-Volterra ODE inverse problem — 7-D posterior via ProbDiffEq probabilistic solver.

REQUIRES JAX_ENABLE_X64=1 at cert time (and recipe runs).

Under float32, probdiffeq's lax.scan gradient path accumulates rounding errors at
extreme parameter values (prior-region init: grad_norm ~25 k vs posterior ~15).
Under JAX 0.10.0's XLA fusion changes the float32 noise is enough to cause
NUTS dual-averaging to collapse the step_size → 41 % divergences from any
default init.  Under float64 the 1752× gradient-scale mismatch is handled
exactly → clean warmup from default prior init across all tested seeds.

Strictly analogous to the gp_regression x64 fix (dense-Cholesky precision).
Ratified @tl 2026-05-26T16:56.  See worklog/threads/lotka-volterra-cert-regression.md §9.
"""

import functools as ft
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from tuningfork.model._base import Posterior

try:
    from probdiffeq import ivpsolve
    from probdiffeq import probdiffeq as _pdq
except ImportError as e:
    raise ImportError(
        "probdiffeq is required for the lotka_volterra model. "
        "Install with: uv add probdiffeq"
    ) from e

__all__ = [
    "ENTRY",
    "OBSERVATIONS",
    "OBSERVATION_TIMES",
    "T_OBS",
    "MU_TRUE",
]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Number of observation time points (T = 40 per statistician verdict)
T_OBS: int = 40

#: Unconstrained dimensionality:
#: alpha(1) + beta(1) + gamma(1) + delta(1) + u0(1) + v0(1) + sigma_obs(1)
DIM: int = 7

#: Path to committed .npz data file
_NPZ_PATH: Path = Path(__file__).parent / "_data" / "lotka_volterra.npz"

# ---------------------------------------------------------------------------
# Load synthetic data (shape: (40, 2), float32)
# ---------------------------------------------------------------------------

_data = np.load(_NPZ_PATH)

# Dtype selection: requires_x64=True is set on ENTRY (see bottom of file).
# At cert/recipe time (x64=True): arrays load as float64 → ODE arithmetic
# stable at all parameter values the chain might visit.
# At test-import time (x64 typically False): they load as float32. The model
# is not invoked at import; the cert assertion in certify_reference_nuts
# catches any actual cert attempt without x64. This avoids JAX UserWarning
# during pytest collection (treated as error in CI).
_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

OBSERVATIONS: jnp.ndarray = jnp.array(_data["observations"], dtype=_dtype)
OBSERVATION_TIMES: jnp.ndarray = jnp.array(_data["observation_times"], dtype=_dtype)

# Validate shapes
assert OBSERVATIONS.shape == (
    T_OBS,
    2,
), f"Expected OBSERVATIONS shape ({T_OBS}, 2), got {OBSERVATIONS.shape}"
assert OBSERVATION_TIMES.shape == (
    T_OBS,
), f"Expected OBSERVATION_TIMES shape ({T_OBS},), got {OBSERVATION_TIMES.shape}"

#: Ground-truth parameters used to generate synthetic data
MU_TRUE: dict[str, float] = {
    "alpha": float(_data["alpha_true"]),
    "beta": float(_data["beta_true"]),
    "gamma": float(_data["gamma_true"]),
    "delta": float(_data["delta_true"]),
    "u0": float(_data["u0_true"]),
    "v0": float(_data["v0_true"]),
    "sigma_obs": float(_data["sigma_obs_true"]),
}


# ---------------------------------------------------------------------------
# ProbDiffEq ODE solver
# ---------------------------------------------------------------------------


def _lotka_volterra_vf(
    u: jnp.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    *,
    t: float,
) -> jnp.ndarray:
    """Lotka-Volterra vector field [du/dt, dv/dt].

    Parameters
    ----------
    u
        State vector [prey, predator].
    alpha
        Prey birth rate.
    beta
        Predation rate.
    gamma
        Predator death rate.
    delta
        Predator growth from predation.
    t
        Time (unused — autonomous ODE).
    """
    du = alpha * u[0] - beta * u[0] * u[1]
    dv = delta * u[0] * u[1] - gamma * u[1]
    return jnp.array([du, dv])


def _solve_lv(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    u0: float,
    v0: float,
    observation_times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve Lotka-Volterra ODE via ProbDiffEq on a fixed time grid.

    Uses isotropic filter (TS0 correction, MLE output-scale calibration).
    Returns the zeroth-order Taylor coefficient (position mean/std).

    Parameters
    ----------
    alpha, beta, gamma, delta
        ODE parameters.
    u0, v0
        Initial conditions (prey, predator).
    observation_times
        1-D array of time points including t0.

    Returns
    -------
    u_mean : jnp.ndarray, shape (T, 2)
        Posterior mean trajectory.
    u_std : jnp.ndarray, shape (T, 1)
        Posterior standard deviation (solver uncertainty, isotropic scalar per
        time step; shape (T, 1) so it broadcasts correctly with u_mean in the
        likelihood).

    Notes
    -----
    ProbDiffEq 0.9 API vs 0.8:

    - ``ivpsolvers`` module removed; prior/strategy/correction/solver now live
      in ``probdiffeq.probdiffeq`` (imported as ``_pdq``).
    - ``prior_wiener_integrated(tcoeffs, ssm_fact="isotropic")`` (returns 3-tuple)
      → ``ssm.prior_wiener_integrated(tcoeffs)`` where
        ``ssm = _pdq.state_space_model_isotropic()`` (returns prior directly).
    - ``strategy_filter(ssm=ssm)`` → ``_pdq.strategy_filter()`` (no ssm kwarg).
    - ``correction_ts0(vf, ssm=ssm)`` → ``ssm.constraint_ode_ts0(ode)`` where
        ``ode = _pdq.ode_autonomous(vf_autonomous)``.
    - ``taylor.odejet_padded_scan(vf, (u,), num=N)`` → returns tcoeffs directly;
      in 0.9: ``_pdq.jetexpand_ode_padded_scan(num=N)(ode, [u], t=t0)`` returns
      ``(tcoeffs, metadata)``.
    - ``solver_mle(strategy, correction=..., prior=..., ssm=...)``
      → ``_pdq.solver_mle(constraint=constraint, strategy=strategy)``.
    - ``ivpsolve.solve_fixed_grid(init, grid=..., solver=..., ssm=...)``
      → ``ivpsolve.solve_fixed_grid(solver=slvr)`` returns a callable;
        call it as ``solve_fn(prior, grid=observation_times)``.
    - ``solution.u`` / ``solution.u_std``: were list[array] in 0.8;
      in 0.9, ``solution.u`` is ``IsotropicNormal`` with ``.mean`` (list[array])
      and ``.std`` (list[array]). Index 0 = position coefficient.
      **Shape change**: std was (T, 2) in 0.8; is (T,) in 0.9 (isotropic scalar
      per time step). We reshape to (T, 1) so the likelihood broadcast is correct.

    Numeric equivalence: u_mean max abs diff (0.8.2 vs 0.9.2) = 1.1e-14 (float64
    machine epsilon). Verdict: trajectories are identical within floating-point
    precision; no re-cert required for the mean. u_std values differ due to the
    representation change (per-component vs isotropic scalar) but both encode
    the same underlying uncertainty.
    """
    u_init = jnp.array([u0, v0])
    # Autonomous vector field (no time kwarg) required by 0.9 ode_autonomous
    vf_autonomous = ft.partial(
        _lotka_volterra_vf, alpha=alpha, beta=beta, gamma=gamma, delta=delta, t=0.0
    )
    # Wrap as a probdiffeq 0.9 JetOdeAutonomous for Taylor expansion
    ode = _pdq.ode_autonomous(vf_autonomous)
    # Compute 2nd-order Taylor coefficients at t0
    expand = _pdq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = expand(ode, [u_init], t=observation_times[0])
    # Build isotropic SSM, prior, strategy, and TS0 correction
    ssm = _pdq.state_space_model_isotropic()
    prior = ssm.prior_wiener_integrated(tcoeffs)
    strategy = _pdq.strategy_filter()
    constraint = ssm.constraint_ode_ts0(ode)
    slvr = _pdq.solver_mle(constraint=constraint, strategy=strategy)
    # Build and run the fixed-grid solve (0.9 returns a callable)
    solve_fn = ivpsolve.solve_fixed_grid(solver=slvr)
    solution = solve_fn(prior, grid=observation_times)
    # solution.u is IsotropicNormal; .mean/.std are lists of Taylor coefficients
    # Index 0 = position (zeroth-order) coefficient
    u_mean = jnp.array(solution.u.mean[0])  # shape (T, 2)
    u_std = jnp.array(solution.u.std[0])[:, None]  # shape (T, 1) — isotropic
    return u_mean, u_std


# ---------------------------------------------------------------------------
# NumPyro model (7-D ODE inverse, likelihood form B)
# ---------------------------------------------------------------------------


def lotka_volterra_inverse(
    observations: jnp.ndarray,
    observation_times: jnp.ndarray,
) -> None:
    """NumPyro model: 7-D Lotka-Volterra ODE inverse via ProbDiffEq.

    Posteriorises over 4 ODE parameters, 2 initial conditions, and 1
    observation noise scale. Likelihood form B: Normal(u_mean, sqrt(u_std^2 +
    sigma_obs^2)) incorporates both solver uncertainty and measurement noise.

    Parameters
    ----------
    observations
        Observed prey/predator time series of shape (T, 2).
    observation_times
        Observation times of shape (T,); first element is t0.

    Notes
    -----
    All four ODE parameters and both initial conditions are positive, so we
    use LogNormal priors in the unconstrained space. sigma_obs uses HalfNormal.

    Prior centres are chosen to be within 1 sigma of the synthetic truth:
        alpha ~ LogNormal(log(0.5), 0.5)   # truth = 0.5
        beta  ~ LogNormal(log(0.05), 0.5)  # truth = 0.05
        gamma ~ LogNormal(log(0.5), 0.5)   # truth = 0.5
        delta ~ LogNormal(log(0.05), 0.5)  # truth = 0.05
        u0    ~ LogNormal(log(10.0), 0.3)  # truth = 10.0
        v0    ~ LogNormal(log(5.0), 0.3)   # truth = 5.0
        sigma_obs ~ HalfNormal(1.0)        # truth = 0.5

    The ODE is solved on the fixed grid ``observation_times`` via
    ``ivpsolve.solve_fixed_grid`` with a 2nd-order isotropic filter (TS0
    correction, MLE calibration). Solver std is incorporated into the
    likelihood as form B.
    """
    alpha = numpyro.sample("alpha", dist.LogNormal(jnp.log(jnp.array(0.5)), 0.5))
    beta = numpyro.sample("beta", dist.LogNormal(jnp.log(jnp.array(0.05)), 0.5))
    gamma = numpyro.sample("gamma", dist.LogNormal(jnp.log(jnp.array(0.5)), 0.5))
    delta = numpyro.sample("delta", dist.LogNormal(jnp.log(jnp.array(0.05)), 0.5))
    u0 = numpyro.sample("u0", dist.LogNormal(jnp.log(jnp.array(10.0)), 0.3))
    v0 = numpyro.sample("v0", dist.LogNormal(jnp.log(jnp.array(5.0)), 0.3))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))

    # Solve ODE to get probabilistic trajectory
    u_mean, u_std = _solve_lv(alpha, beta, gamma, delta, u0, v0, observation_times)

    # Likelihood form B: incorporate solver uncertainty
    # obs[t] ~ Normal(u_mean[t], sqrt(u_std[t]^2 + sigma_obs^2))
    scale = jnp.sqrt(u_std**2 + sigma_obs**2)
    numpyro.sample(
        "obs",
        dist.Normal(u_mean, scale),
        obs=observations,
    )


# Statistician verdict (TL-orchestrated, 2026-05-08):
#     Approve-with-modifications. 7-D ODE inverse via ProbDiffEq probabilistic
#     solver (not plan's "4-D"). Likelihood form B (incorporates solver
#     uncertainty). posteriordb_id=None. Stable limit cycle with ~3 oscillations.
#     Dimensionality correction (plan said 4-D, actual is 7-D):
#         4 ODE params (alpha, beta, gamma, delta)
#         + 2 initial conditions (u0, v0)
#         + 1 observation noise (sigma_obs)
#         = 7 unconstrained parameters
#     ProbDiffEq integration (ported to 0.9.x API, 2026-06-23):
#         Uses ``_pdq.state_space_model_isotropic().prior_wiener_integrated`` +
#         ``_pdq.strategy_filter`` + ``ssm.constraint_ode_ts0`` +
#         ``_pdq.solver_mle`` + ``ivpsolve.solve_fixed_grid``.
#         ``solution.u`` is now an ``IsotropicNormal``; ``.mean[0]`` gives the
#         mean trajectory of shape (T, 2); ``.std[0]`` gives the isotropic std
#         of shape (T,), reshaped to (T, 1) for likelihood broadcasting.
#         u_mean max abs diff (0.8.2 vs 0.9.2) = 1.1e-14 (float64 eps); no
#         re-cert required.
#     Likelihood form B (solver uncertainty included):
#         obs[t] ~ Normal(u_mean[t], sqrt(u_std[t]^2 + sigma_obs^2))
#         where u_mean / u_std come from the probabilistic ODE solver.
#         Form A (ignoring solver uncertainty) defeats the purpose of ProbDiffEq.
#         Form C (full multivariate) is overkill at 7-D.
#     Synthetic data design (KSC-style):
#         alpha=0.5, beta=0.05, gamma=0.5, delta=0.05, u0=10.0, v0=5.0,
#         sigma_obs=0.5, T_obs=40 points on linspace(0, 20), seed=1234.
#         Stable limit cycle producing ~3 oscillations over 20 time units.
#         Committed as tuningfork/model/_data/lotka_volterra.npz.
#     posteriordb_id = None:
#         Stan's lotka_volterra uses a different solver/likelihood structure.
#         No upstream cross-check available.
#     reference-certification budget:
#         In-spawn verification: n_warmup=500, n_samples=500, 4 chains.
#         (Lighter than radon/stoch_vol — 7-D but each likelihood call = ODE solve.)
#         Production cache: n_warmup=1000, n_samples=10000.
# References:
#     Lotka, A. J. (1925). Elements of Physical Biology. Williams & Wilkins.
#     Volterra, V. (1926). Fluctuations in the abundance of a species.
# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

ENTRY = Posterior(
    name="lotka_volterra",
    dim=DIM,
    class_="ode",
    tags=(
        "ode",
        "nonlinear",
        "expensive_likelihood",
        "synthetic",
    ),
    requires_x64=True,
    numpyro_model=lotka_volterra_inverse,
    model_args=(OBSERVATIONS, OBSERVATION_TIMES),
    model_kwargs={},
    posteriordb_id=None,  # Stan's LV uses a different solver/likelihood structure
    citations=(
        "Lotka, A. J. (1925). Elements of Physical Biology. Williams & Wilkins.",
        "Volterra, V. (1926). Fluctuations in the Abundance of a Species considered "
        "Mathematically. Nature, 118(2972), 558–560.",
    ),
    description=(
        "7-D Lotka-Volterra ODE inverse via ProbDiffEq. "
        "T=40 synthetic observations over 20 time units (~3 oscillations). "
        "True params: alpha=0.5, beta=0.05, gamma=0.5, delta=0.05, "
        "u0=10.0, v0=5.0, sigma_obs=0.5. "
        "Likelihood form B: Normal(u_mean, sqrt(u_std^2 + sigma_obs^2)). "
        "posteriordb_id=None. Dim=7: alpha+beta+gamma+delta+u0+v0+sigma_obs."
    ),
    headline_params=("alpha", "beta", "gamma", "delta"),
    headline_coords=None,
)
