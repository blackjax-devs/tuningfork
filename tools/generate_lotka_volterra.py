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
"""Generate synthetic Lotka-Volterra observations and persist as .npz.

Provenance:
    Synthetic data — NOT from an external source.
    Model: Lotka-Volterra ODE inverse via ProbDiffEq probabilistic solver.
    Parameters (ground truth):
        alpha_true    = 0.5    (prey birth rate)
        beta_true     = 0.05   (predation rate)
        gamma_true    = 0.5    (predator death rate)
        delta_true    = 0.05   (predator growth from predation)
        u0_true       = 10.0   (initial prey population)
        v0_true       = 5.0    (initial predator population)
        sigma_obs_true = 0.5   (observation noise std)
    T_obs = 40 time points on linspace(0, 20).
    Seed: jax.random.key(1234).

Output:
    tuningfork/model/_data/lotka_volterra.npz — compressed numpy array with:
        observations: float32 array of shape (40, 2) — [prey, predator]
        observation_times: float32 array of shape (40,)
        alpha_true, beta_true, gamma_true, delta_true,
        u0_true, v0_true, sigma_obs_true: scalar float32

Usage:
    cd tuningfork
    uv run python tools/generate_lotka_volterra.py
"""

import functools as ft
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from probdiffeq import ivpsolve
from probdiffeq import probdiffeq as _pdq

# Ground-truth parameters (Statistician verdict: stable limit-cycle with ~3 oscillations)
ALPHA_TRUE: float = 0.5
BETA_TRUE: float = 0.05
GAMMA_TRUE: float = 0.5
DELTA_TRUE: float = 0.05
U0_TRUE: float = 10.0
V0_TRUE: float = 5.0
SIGMA_OBS_TRUE: float = 0.5
T_OBS: int = 40
T_END: float = 20.0

_OUT = (
    Path(__file__).parent.parent
    / "tuningfork"
    / "model"
    / "_data"
    / "lotka_volterra.npz"
)


def lotka_volterra_vf(
    u: jnp.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    *,
    t: float,
) -> jnp.ndarray:
    """Lotka-Volterra vector field.

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

    Returns
    -------
    jnp.ndarray of shape (2,): [du/dt, dv/dt].
    """
    du = alpha * u[0] - beta * u[0] * u[1]
    dv = delta * u[0] * u[1] - gamma * u[1]
    return jnp.array([du, dv])


def solve_lv_fixed_grid(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    u0: float,
    v0: float,
    observation_times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve Lotka-Volterra ODE on a fixed grid via ProbDiffEq.

    Uses an isotropic filter (TS0 correction) with MLE calibration.
    Returns mean trajectory and per-step standard deviation for the
    zeroth-order (position) Taylor coefficient.

    Parameters
    ----------
    alpha, beta, gamma, delta
        ODE parameters.
    u0, v0
        Initial conditions.
    observation_times
        1-D array of time points; first element is t0.

    Returns
    -------
    u_mean : jnp.ndarray, shape (T, 2)
        Mean trajectory at each observation time.
    u_std : jnp.ndarray, shape (T, 2)
        Solver-estimated standard deviation at each observation time.
    """
    u_init = jnp.array([u0, v0])
    # Autonomous vector field (no time kwarg) required by probdiffeq 0.9
    vf_autonomous = ft.partial(
        lotka_volterra_vf, alpha=alpha, beta=beta, gamma=gamma, delta=delta, t=0.0
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
    # solution.u is IsotropicNormal; .mean/.std are lists of Taylor coefficients.
    # Index 0 = position (zeroth-order) coefficient.
    # Shape change vs 0.8: std is now (T,) isotropic scalar, not (T, 2).
    u_mean = jnp.array(solution.u.mean[0])  # shape (T, 2)
    u_std = jnp.array(solution.u.std[0])  # shape (T,) — isotropic scalar
    return u_mean, u_std


def simulate(
    alpha: float = ALPHA_TRUE,
    beta: float = BETA_TRUE,
    gamma: float = GAMMA_TRUE,
    delta: float = DELTA_TRUE,
    u0: float = U0_TRUE,
    v0: float = V0_TRUE,
    sigma_obs: float = SIGMA_OBS_TRUE,
    t_obs: int = T_OBS,
    t_end: float = T_END,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate noisy Lotka-Volterra observations.

    Parameters
    ----------
    alpha, beta, gamma, delta
        ODE parameters.
    u0, v0
        Initial prey/predator populations.
    sigma_obs
        Observation noise standard deviation.
    t_obs
        Number of observation time points.
    t_end
        End time (observations span [0, t_end]).
    seed
        JAX random seed.

    Returns
    -------
    observations : np.ndarray, shape (t_obs, 2), dtype float32
        Noisy prey/predator observations.
    observation_times : np.ndarray, shape (t_obs,), dtype float32
    """
    observation_times = jnp.linspace(0.0, t_end, t_obs)
    u_mean, _u_std = solve_lv_fixed_grid(
        alpha, beta, gamma, delta, u0, v0, observation_times
    )
    key = jax.random.key(seed)
    noise = jax.random.normal(key, shape=u_mean.shape) * sigma_obs
    observations = u_mean + noise
    return (
        np.array(observations, dtype=np.float32),
        np.array(observation_times, dtype=np.float32),
    )


def main() -> None:
    observations, observation_times = simulate()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _OUT,
        observations=observations,
        observation_times=observation_times,
        alpha_true=np.float32(ALPHA_TRUE),
        beta_true=np.float32(BETA_TRUE),
        gamma_true=np.float32(GAMMA_TRUE),
        delta_true=np.float32(DELTA_TRUE),
        u0_true=np.float32(U0_TRUE),
        v0_true=np.float32(V0_TRUE),
        sigma_obs_true=np.float32(SIGMA_OBS_TRUE),
    )

    print(f"Saved to {_OUT}")
    print(f"  observations.shape: {observations.shape}")
    print(
        f"  prey  range: [{observations[:, 0].min():.2f}, {observations[:, 0].max():.2f}]"
    )
    print(
        f"  pred  range: [{observations[:, 1].min():.2f}, {observations[:, 1].max():.2f}]"
    )
    print(f"  all finite: {np.all(np.isfinite(observations))}")
    print(
        f"  ground truth: alpha={ALPHA_TRUE}, beta={BETA_TRUE}, "
        f"gamma={GAMMA_TRUE}, delta={DELTA_TRUE}, "
        f"u0={U0_TRUE}, v0={V0_TRUE}, sigma_obs={SIGMA_OBS_TRUE}"
    )


if __name__ == "__main__":
    main()
