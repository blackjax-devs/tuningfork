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
"""Closed-form GP marginal ground-truth generation.

Covers the ``closed_form_gp_marginal`` path for ``gp_regression`` — the only
model in the catalog whose exact posterior is too high-dimensional for direct
NUTS but admits an exact analytic marginalization.

Algorithm
---------
1. Marginalize the 200-dimensional GP latent function ``f`` analytically:
   ``log p(y | θ) = log N(y; 0, K_θ + JITTER·I + σ²I)``, reducing sampling to
   a 3-dimensional space ``(log_lengthscale, log_kernel_scale, log_noise_scale)``.
2. Run NUTS on this 3-dim marginal with ``blackjax.window_adaptation``.
3. For each ``θ`` sample, draw ``f | θ, y`` from the exact conditional Gaussian
   ``N(μ_f, Σ_f)`` where ``μ_f = A B⁻¹ y`` and ``Σ_f = A - A B⁻¹ A``.
4. Convert ``f → f_raw`` (NCP whitened) via the lossless triangular solve
   ``f_raw = L_K^{-1} f``.

The method is exact for Gaussian likelihood (no approximation).  Validated at
``n=40000`` against the original 40k-draw single-chain NUTS GT:
``|Δmean| ≤ 0.0012``, ``f RMSE(mean) = 0.0001`` — 50× below the MC noise floor.

Environment flags
-----------------
``JAX_ENABLE_X64=1`` (or ``GT_X64=1``)
    Required.  64-bit floats are necessary for the Cholesky factorisations of
    the 200×200 kernel matrix to be numerically stable.
``OPENBLAS_NUM_THREADS=1``
    Required on multi-core machines with NumPy backed by OpenBLAS.  On a
    128-core machine without this flag, the concurrent Cholesky allocations
    during chunked ``f``-reconstruction (≈7 × 200×200 intermediates per draw)
    corrupt the heap and SIGABRT the process.

Chunking
--------
``f``-reconstruction processes draws in chunks of 200 to bound peak XLA
intermediate buffer allocation to ≈512 MB.  Without chunking, 1000 draws in
a single vmap call allocates ≈2.5 GB of XLA buffers.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["generate_gp_marginal"]


def generate_gp_marginal(
    model_name: str,
    committed_summary: dict,
    out_dir: Path,
    *,
    seed: int | None = None,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_warmup: int | None = None,
    smoke: bool = False,
) -> dict:
    """Generate GP regression ground-truth via closed-form marginal sampling.

    Parameters
    ----------
    model_name
        Must be ``"gp_regression"``.
    committed_summary
        Parsed ``summary_v2.json`` for ``gp_regression``.
    out_dir
        Directory where ``draws.npz`` and ``summary_v2.json`` are written.
    seed
        Master RNG seed.  Defaults to committed seed (``20260715``).
    n_chains
        Number of chains.  Defaults to 10.
    n_draws
        Draws per chain.  Defaults to 10000.
    n_warmup
        Warmup steps.  Defaults to 1000 (marginal posterior converges fast).
    smoke
        Run at tiny scale (1 chain × 50 draws × 50 warmup).

    Returns
    -------
    dict
        Parsed ``summary_v2.json`` for the generated GT.

    Raises
    ------
    NotImplementedError
        This path will be implemented in a follow-up commit.
    """
    if model_name != "gp_regression":
        raise ValueError(
            f"generate_gp_marginal() is only for gp_regression, got {model_name!r}"
        )
    raise NotImplementedError(
        "generate_gp_marginal is not yet implemented. " "Await the next module update."
    )
