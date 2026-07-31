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

"""Explicit descriptor registry for tuningfork warmups."""

from tuningfork.base_method._base import HyperparamSpace
from tuningfork.warmup._base import Warmup

_COMMON = ("nuts", "hmc", "mala", "rwm", "barker")

WARMUPS: dict[str, Warmup] = {
    "adjusted_mclmc_trajectory_tuning": Warmup(
        name="adjusted_mclmc_trajectory_tuning",
        compatible_methods=("adjusted_mclmc_dynamic",),
        default_hp_space=(
            HyperparamSpace("n_pilot", "int", low=500, high=500),
            HyperparamSpace("avg_grid", "categorical", choices=((1.0, 2.0, 4.0),)),
        ),
        notes=(
            "Extended adjusted-MCLMC warmup that escapes the MALA-collapse artifact. "
            "Step 1: runs blackjax.adjusted_mclmc_find_L_and_step_size (static kernel) "
            "per chain (vmap) to produce (step_size, L_tuned, diagonal IMM). "
            "Step 2: grid-searches avg ∈ {1.0, 2.0, 4.0} via short pilot "
            "(n_pilot=500 samples × num_chains) using "
            "blackjax.diagnostics.effective_sample_size (JAX-native, no ArviZ "
            "dependency, jit-safe). Step 3: outputs L = avg_star × step_size "
            "(per-chain). Diagnostic sidecars: _avg_star (float), "
            "_avg_search_ess_per_grad (dict), _total_tuning_steps (warmup + pilot "
            "grads). Only compatible with adjusted_mclmc_dynamic (not static "
            "adjusted_mclmc). Canonical target acceptance rate: 0.9. Validated "
            "(n_warmup=500, n_pilot=500, 4 chains): escapes MALA where a longer "
            "trajectory helps — mvn_10 picks avg=2 (2.9× the avg=1 ess/grad), "
            "german_credit picks avg=4 (8.3×). On ill_cond_50 the diagonal IMM "
            "cannot whiten the rotated kappa=1000 geometry, so avg=1 is correctly "
            "selected (pair with an LRD/dense IMM warmup to benefit from longer "
            "trajectories)."
        ),
    ),
    "adjusted_mclmc_tuning": Warmup(
        name="adjusted_mclmc_tuning",
        compatible_methods=("adjusted_mclmc", "adjusted_mclmc_dynamic"),
        notes=(
            "Adjusted-MCLMC-specific adaptation via "
            "blackjax.adjusted_mclmc_find_L_and_step_size. Finds L, step_size, "
            "and a diagonal inverse_mass_matrix jointly. Uses static "
            "adjusted_mclmc kernel for tuning (adapter is integrator-agnostic). "
            "Canonical target acceptance rate: 0.9 (per upstream adjusted-MCLMC "
            "tests). Generated telemetry records the summed per-chain tuning cost "
            "as two gradients per integrator step. "
            "Compatible with both adjusted_mclmc (static) and "
            "adjusted_mclmc_dynamic (random N). multi-chain by default (num_chains=4 "
            "via jax.vmap); per-chain L/step_size/IMM returned as "
            "(num_chains,)/(num_chains,)/(num_chains, d) arrays."
        ),
    ),
    "chees": Warmup(
        name="chees",
        compatible_methods=("dynamic_hmc",),
        notes=(
            "CHEES (Change in Estimator of Step Size) adaptation for dynamic-HMC. "
            "Adapts both step_size AND the trajectory-length distribution "
            "(integration_steps_fn, next_random_arg_fn, integration_steps_params). "
            "Like MEADS but for dynamic-HMC instead of GHMC; both are "
            "multi-chain-by-construction adaptation procedures. Upstream API note: "
            "chees_adaptation.run() requires step_size and an optax optimizer as "
            "positional args (unlike meads_adaptation.run); the generated protocol "
            "supplies optax.adam(0.05, b1=0, b2=0.95) (canonical CHEES/SNAPER form; "
            "LR calibrated for n_warmup=2000, issue #217). Callable params "
            "(next_random_arg_fn, integration_steps_fn) are Python functions returned "
            "by CHEES and passed through unchanged. Numeric outputs remain shared "
            "ensemble estimates: scalar step_size and (d,) inverse_mass_matrix. "
            "dynamic_hmc-specific; not compatible with HMC, NUTS, GHMC, or any "
            "other kernel. multi-chain by default (num_chains=4); "
            "target_acceptance_rate=0.651 (CHEES default)."
        ),
    ),
    "fullrank_vi": Warmup(
        name="fullrank_vi",
        compatible_methods=_COMMON,
        default_hp_space=(
            HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
        ),
        notes=(
            "Full-rank VI warmup: runs a single full-rank VI optimisation (shared "
            "across all chains) via jax.lax.scan over num_optimization_steps Adam "
            "steps. Draws num_chains independent initial positions from the fitted "
            "variational distribution N(mu, L@L.T) and returns the dense covariance "
            "L@L.T as the inverse_mass_matrix (shared across all chains — the fit is "
            "shared, only init positions differ). When n_warmup > 0, runs Nesterov "
            "dual averaging with the VI IMM frozen to adapt step_size; when "
            "n_warmup == 0, returns step_size_default (1.0). Sidecar: _frvi_elbo "
            "(final ELBO scalar). Compatible: nuts, hmc, mala, rwm, barker. NOT "
            "compatible with mclmc (microcanonical geometry). Recommended ONLY for "
            "d <= 30: the "
            "Cholesky parameterisation has O(d^2) parameters which become expensive "
            "at high dimension. Use meanfield_vi warmup for d > 30. Production "
            "default: num_optimization_steps=20_000, optimizer=optax.adam(1e-2). "
            "Use 5_000 in tests."
        ),
    ),
    "mclmc_lrd_tuning": Warmup(
        name="mclmc_lrd_tuning",
        compatible_methods=("mclmc",),
        notes=(
            "Generated NUTS-pilot LRD-preconditioned MCLMC warmup. Pipeline: "
            "(1) single-chain NUTS window adaptation and pilot draws; "
            "(2) standardisation plus rank-k SVD extraction of "
            "LowRankInverseMassMatrix, clamped to the available SVD modes; "
            "(3) multi-chain unadjusted LRD tuning via vmapped "
            "mclmc_find_L_and_step_size. Returns per-chain L/step_size and a shared "
            "LRD IMM broadcast across chains. This NUTS-pilot route preserves the "
            "provenance of the committed LRD recipes. Do not silently replace it "
            "with blackjax.mclmc_lrd_warmup (upstream PR #937, SHA 359205da): that "
            "routine uses a diagonal-MCLMC pilot, an ESS-based rank guard, and "
            "averaged L/step_size, so it is a different algorithm. Recommended for "
            "ill-conditioned targets where diagonal mclmc_tuning fails."
        ),
    ),
    "mclmc_tuning": Warmup(
        name="mclmc_tuning",
        compatible_methods=("mclmc",),
        notes=(
            "MCLMC-specific adaptation via blackjax.mclmc_find_L_and_step_size. "
            "Finds L, step_size, and a diagonal inverse_mass_matrix jointly. Generated "
            "telemetry records the summed per-chain tuning cost as two gradients per "
            "integrator step. Not compatible with any other kernel (HMC/NUTS use "
            "window_adaptation). multi-chain by default (num_chains=4 via jax.vmap); "
            "per-chain L/step_size/IMM returned as "
            "(num_chains,)/(num_chains,)/(num_chains, d) arrays."
        ),
    ),
    "meads": Warmup(
        name="meads",
        compatible_methods=("ghmc",),
        notes=(
            "MEADS (Maximum-Eigenvalue Adapted Dual-Averaging Step-size) warmup for "
            "GHMC. Unlike window_adaptation_diag_imm which vmaps per-chain "
            "adaptation, MEADS runs a single multi-chain adaptation that cross-"
            "validates across num_folds folds; chains are inputs, not loop iterations. "
            "Requires num_chains ≥ num_folds (default 4). Adapts step_size, "
            "momentum_inverse_scale, alpha, and delta jointly. GHMC-specific; not "
            "compatible with HMC, NUTS, or any other kernel. multi-chain by default "
            "(num_chains=4). CAUTION: at the DEFAULT "
            "num_chains=4/num_folds=4 (n_per_fold=1), MEADS's cross-chain std within "
            "a fold is 0/0-NaN by construction (a single-sample std has no variance "
            "to estimate, independent of init_jitter_scale) -- this is distinct from "
            "the identical-init NaN that init_jitter_scale fixes. MEADS needs "
            "num_chains large enough that num_chains // num_folds >= 2 "
            "(recipe-generation practice: num_chains>=16) to adapt at all. MEADS "
            "returns shared ensemble estimates: scalar step_size/alpha/delta and "
            "(d,) momentum_inverse_scale; generated execution rejects low-rank MEADS "
            "until that metric can be represented losslessly."
        ),
    ),
    "meanfield_vi": Warmup(
        name="meanfield_vi",
        compatible_methods=_COMMON,
        default_hp_space=(
            HyperparamSpace("num_optimization_steps", "int", low=1_000, high=50_000),
        ),
        notes=(
            "Mean-field VI warmup: runs a single mean-field VI optimisation (shared "
            "across all chains) via jax.lax.scan over num_optimization_steps Adam "
            "steps. Draws num_chains independent initial positions from the fitted "
            "variational distribution and returns the diagonal variance exp(2*rho) "
            "as the inverse_mass_matrix (shared across all chains — the fit is "
            "shared, only init positions differ). step_size adaptation: when n_warmup "
            "> 0, runs n_warmup steps of Nesterov dual averaging with the VI IMM frozen "
            "to find the adapted step_size; when n_warmup == 0, returns "
            "step_size_default (1.0). Sidecar: _mfvi_elbo (final ELBO scalar). "
            "Compatible: nuts, hmc, mala, rwm, barker. NOT compatible with mclmc "
            "(microcanonical geometry). Production default: "
            "num_optimization_steps=10_000, optimizer=optax.adam(1e-2). Use 2_000 "
            "in tests."
        ),
    ),
    "multipathfinder": Warmup(
        name="multipathfinder",
        compatible_methods=_COMMON,
        notes=(
            "Multi-path Pathfinder warmup: generated protocol using "
            "blackjax.pathfinder_adaptation(num_chains, n_paths>=2, "
            "imm_estimator='lbfgs_psis_mixture'). Multi-path Pathfinder fit from "
            "n_paths independent starting positions (default n_paths == num_chains). "
            "PSIS-resamples num_chains init positions. Derives shared dense (d, d) "
            "IMM via the PSIS-weighted L-BFGS mixture covariance (law of total "
            "variance). Dual-averaging runs over the configured chains, then generated "
            "execution uses the mean step_size scalar and shared (d, d) IMM. Sidecar: "
            "_multipathfinder_psis_pareto_k (PSIS Pareto-k diagnostic). IMM shape "
            "is dense (d, d); the superseded direct implementation "
            "returned a diagonal (num_chains, d) matrix. Compatible: nuts, hmc, "
            "mala, rwm, barker. NOT "
            "compatible with mclmc (microcanonical geometry)."
        ),
    ),
    "multipathfinder_window_adaptation": Warmup(
        name="multipathfinder_window_adaptation",
        compatible_methods=_COMMON,
        notes=(
            "Paper-canonical composed warmup (Zhang et al. 2022 § 4): multi-path "
            "Pathfinder as init-strategy preceding window adaptation. Runs "
            "multipathfinder (n_paths=num_chains by default) to derive a shared dense "
            "(d, d) IMM via the PSIS-weighted L-BFGS mixture covariance. PSIS-"
            "resamples num_chains init positions. Passes the dense IMM as "
            "initial_inverse_mass_matrix to single-chain window_adaptation, with "
            "imm_shrinkage_to_previous=20.0 (medium persistence) so the "
            "multipathfinder IMM seed remains influential across windows. Generated "
            "execution broadcasts distinct PSIS-resampled positions into chain states "
            "while keeping the adapted step_size scalar and dense (d, d) IMM shared. "
            "Preserves the PSIS Pareto-k diagnostic. Compatible: nuts, hmc, "
            "mala, rwm, barker. NOT compatible with mclmc (microcanonical geometry)."
        ),
    ),
    "no_warmup": Warmup(
        name="no_warmup",
        compatible_methods=("*",),
        notes=(
            "Identity warmup: returns the kernel's init state with default params and "
            "an empty adapted_params dict. Zero gradient evaluations. Used for LOW-"
            "effort recipes, gradient-free kernels (RWM), and isolated no-warmup "
            "baselines. MCLMC is handled specially: kernel.init(position, rng_key) "
            "rather than kernel.init(position). multi-chain by default (num_chains=4 "
            "via jax.vmap); states batched with leading dim num_chains (never squeezed)."
        ),
    ),
    "pathfinder": Warmup(
        name="pathfinder",
        compatible_methods=_COMMON,
        notes=(
            "Single-path Pathfinder warmup (generated protocol invoking "
            "blackjax.pathfinder_adaptation with num_chains and n_paths=1). Adapts "
            "step size via dual-averaging over n_warmup steps, then uses the mean "
            "step_size scalar and shared dense (d, d) L-BFGS inverse Hessian. "
            "The superseded direct implementation returned a diagonal "
            "(num_chains, d) matrix; generated execution intentionally preserves the "
            "full dense geometry. Compatible: nuts, hmc, mala, rwm, barker. NOT "
            "compatible with mclmc (microcanonical geometry)."
        ),
    ),
    "window_adaptation_dense_imm": Warmup(
        name="window_adaptation_dense_imm",
        compatible_methods=(
            "hmc",
            "nuts",
            "mhmc",
            "dynamic_hmc",
            "dmhmc",
            "barker",
            "mala",
            "laplace_hmc",
            "laplace_dhmc",
            "laplace_mhmc",
            "laplace_dmhmc",
        ),
        notes=(
            "Stan-style window adaptation with dense (full-rank) inverse mass matrix. "
            "Compatible with hmc, nuts, mhmc, barker, mala, and laplace_* variants "
            "(all kernels that accept inverse_mass_matrix; mhmc verified by "
            "RECIPE_GENERATION.md Table 1A note + needs_mass_matrix=True in registry). "
            "For dynamic_hmc/dmhmc, the native init requires random_generator_arg "
            "and cannot be called directly by blackjax.window_adaptation; the "
            "generated protocol substitutes a compatible warmup kernel. This direct "
            "route is not compatible with their native init. Use when posterior "
            "correlation is the dominant pathology. multi-chain by default "
            "(num_chains=4 via jax.vmap); per-chain adapted_params returned "
            "(step_size shape (num_chains,), dense IMM shape (num_chains, d, d))."
        ),
    ),
    "window_adaptation_diag_imm": Warmup(
        name="window_adaptation_diag_imm",
        compatible_methods=(
            "hmc",
            "nuts",
            "mhmc",
            "rmhmc",
            "dynamic_hmc",
            "dmhmc",
            "barker",
            "mala",
            "laplace_hmc",
            "laplace_dhmc",
            "laplace_mhmc",
            "laplace_dmhmc",
        ),
        notes=(
            "Standard Stan window adaptation: dual-averaging step_size + diagonal "
            "mass matrix. Compatible with hmc, nuts, mhmc, rmhmc, barker, mala, and "
            "laplace_* variants (all kernels that accept inverse_mass_matrix; mhmc "
            "verified by RECIPE_GENERATION.md Table 1A note + needs_mass_matrix=True "
            "in registry). For rmhmc, the generated protocol dispatches a compatible "
            "kernel route so window_adaptation can call build_kernel(integrator) and "
            "init(position, logdensity_fn); the underlying kernel reuses hmc.build_kernel "
            "so inverse_mass_matrix is passed correctly despite rmhmc's user-facing API "
            "taking mass_matrix. For dynamic_hmc/dmhmc, the native init requires "
            "random_generator_arg and cannot be called directly by "
            "blackjax.window_adaptation; the generated protocol substitutes a compatible "
            "warmup kernel. This direct route is not compatible with their native init. "
            "multi-chain by default (num_chains=4 via jax.vmap); per-chain adapted_params "
            "returned (step_size shape (num_chains,), IMM shape (num_chains, d) or "
            "(num_chains, d, d))."
        ),
    ),
    "window_adaptation_low_rank_imm": Warmup(
        name="window_adaptation_low_rank_imm",
        compatible_methods=(
            "hmc",
            "nuts",
            "mhmc",
            "dynamic_hmc",
            "dmhmc",
            "barker",
            "mala",
            "laplace_hmc",
            "laplace_dhmc",
            "laplace_mhmc",
            "laplace_dmhmc",
        ),
        default_hp_space=(HyperparamSpace("max_rank", "int", low=10, high=10),),
        notes=(
            "Low-rank mass matrix adaptation via Fisher divergence minimisation "
            "(nutpie algorithm; :cite:`seyboldt2026preconditioning`) and Stan's "
            "fast-slow-fast schedule. Metric has the form "
            "M^{-1}=diag(σ)(I+U(Λ-I)U^T)diag(σ): sigma has shape (d,), U has "
            "orthonormal shape (d,k), and positive lam has shape (k,). lam=1 reduces "
            "to diagonal and k approaching d approximates full rank at O(dk) cost. "
            "Requires BlackJAX b094083c or later (#917 removed the non-vmappable "
            "metric closure). Compatible with hmc, nuts, barker, mala, and laplace_* "
            "variants. Use when correlations are strong but d is too large for dense "
            "adaptation. Default max_rank=10. multi-chain generated adaptation returns "
            "per-chain step_size and batched sigma/U/lam fields; structured IMM "
            "sidecars preserve those three arrays."
        ),
    ),
}
