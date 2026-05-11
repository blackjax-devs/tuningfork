# tuningfork

A BlackJAX-native benchmark library for comparing MCMC, VI, and SMC sampling algorithms — modeled after [`inference-gym`](https://pypi.org/project/inference-gym/) and [`posteriordb`](https://github.com/stan-dev/posteriordb), but designed around **calibrated, gradient-counted comparisons** over a curated 14-model suite.

## Why

BlackJAX has 16+ MCMC kernels, 6 VI methods, 6 SMC variants, and 8 adaptation strategies. None are currently benchmarked together with calibrated configurations, gradient-budget accounting, or posteriordb-style certified reference draws. `tuningfork` answers questions like:

- *"What is the best calibrated HMC config for Neal's funnel, and how many leapfrog steps does it cost per effective sample?"*
- *"Does Pathfinder→HMC dominate Stan-window→HMC on hierarchical models, or only on well-conditioned ones?"*
- *"Is MCLMC actually worth it on a 500-D state-space model when both algorithms are tuned to their best?"*

## Status

**Phase 0 — scaffold only.** No working code yet. See the plan document (in the parent `blackjax-devs/` directory) for the full design.

## Suite (14 models)

| # | Name | Dim | Class |
|---|------|-----|-------|
| 1 | Standard MVN (diagonal) | 10 | Gaussian baseline |
| 2 | Ill-conditioned correlated Gaussian | 50 | Ill-conditioned (κ≈1000) |
| 3 | Eight Schools (NCP) | 10 | Hierarchical |
| 4 | Neal's Funnel | 10 | Funnel |
| 5 | Banana (Rosenbrock) | 2–10 | Curved manifold |
| 6 | Radon hierarchical | ~170 | Hierarchical+funnel |
| 7 | Synthetic logistic regression | 3 | GLM baseline |
| 8 | German Credit logistic regression | 25 | GLM real data |
| 9 | Sparse horseshoe linear regression | 103 | Sparse / heavy-tailed |
| 10 | IRT (2PL) | ~230 | Hierarchical, scale-identifiability |
| 11 | 25-mode Gaussian mixture | 2 | Multimodal |
| 12 | Stochastic volatility | ~500 | Latent-Gaussian / state-space |
| 13 | Lotka–Volterra ODE inverse | 4 | Nonlinear, expensive likelihood |
| 14 | GP regression (1D) | ~200 | Latent-Gaussian |

## Calibration tiers

- **reference-certification — Gold reference draws**: 1 chain × 100 000 samples (NUTS + Stan window adaptation), reshape into 10 chunks for split-R̂. Multimodal exception for #11 (parallel-tempered SMC + multi-restart).
- **BO tuning — Per-algorithm tuning**: Optuna BO maximizing `min-bulk-ESS / total_grad_evals`, with per-algorithm acceptance targets.
- **warmup-only — Warmup isolated**: cross-product of (Stan-window, MEADS, ChEES, Pathfinder, MCLMC tuning, no-op) × samplers.

## Headline metric

`primary = min_over_dimensions(bulk_ESS) / total_gradient_evaluations`

## Setup

```bash
make install      # uv sync --group bench
make test         # run tests (default: skip e2e suite)
make test-fast    # inner-loop dev (fast tests only)
make test-full    # merge gate (everything)
make lint         # pre-commit
```

For GPU: `uv pip install "jax[cuda12]"` after `make install`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for a complete guide to test markers (`fast`, `slow`, `e2e`), folder layout, and adding new tests.

## License

Apache 2.0
