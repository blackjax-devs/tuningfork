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
"""Ground-truth regeneration for tuningfork's 16-model benchmark suite.

Every model in the catalog ships committed GT draws in
``tuningfork/catalog/<model>/groundtruth_samples/blackjax/``.  This module
provides a reproducible, self-contained way to regenerate those draws locally.

The generation method for each model is determined automatically from its
committed ``summary_v2.json``; no additional configuration is required.

Quick start (CLI)::

    # Analytic models — seconds
    python -m tuningfork.groundtruth generate mvn_10

    # NUTS models — minutes to hours depending on the model
    python -m tuningfork.groundtruth generate radon

    # Models requiring 64-bit floats
    GT_X64=1 python -m tuningfork.groundtruth generate gp_regression
    GT_X64=1 python -m tuningfork.groundtruth generate lotka_volterra

    # Quick smoke run (2 chains × 50 draws)
    python -m tuningfork.groundtruth generate mvn_10 --smoke

Python API::

    from tuningfork.groundtruth import generate_groundtruth
    summary = generate_groundtruth("mvn_10", out_dir="/tmp/gt_out")

Reproducibility caveat
-----------------------
Output is statistically equivalent to the committed GT (in-spec) but NOT
bit-identical.  CUDA/XLA nondeterminism means float-level values will differ
while being statistically indistinguishable from the committed draws.  Use
``verify_groundtruth`` (or ``--verify`` on the CLI) to confirm equivalence.

Environment flags
-----------------
``GT_X64=1``
    Enable 64-bit JAX.  Required for ``gp_regression`` and
    ``lotka_volterra``.  Must be set before the process starts (sets
    ``JAX_ENABLE_X64=1`` before JAX is imported).

``OPENBLAS_NUM_THREADS=1``
    Required for ``gp_regression`` on machines with 64+ CPU cores to prevent
    heap corruption from concurrent OpenBLAS Cholesky allocations.

``PYTHONUNBUFFERED=1``
    Recommended for long-running models so progress lines reach the log
    without 4 KB buffering.
"""

from tuningfork.groundtruth._dispatch import (
    GTMethod,
    _resolve_gt_method,
    committed_gt_dir,
    load_committed_summary,
)

__all__ = [
    "GTMethod",
    "committed_gt_dir",
    "load_committed_summary",
    "_resolve_gt_method",
    "generate_groundtruth",
]


def generate_groundtruth(
    model_name: str,
    out_dir: str | None = None,
    *,
    seed: int | None = None,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_warmup: int | None = None,
    sequential: bool = False,
    smoke: bool = False,
) -> dict:
    """Generate ground-truth draws for ``model_name``.

    The generation method is determined automatically from the model's
    committed ``summary_v2.json``.

    Parameters
    ----------
    model_name
        Registry model name (e.g. ``"radon"``, ``"mvn_10"``).
    out_dir
        Output directory.  Defaults to a temporary directory.
    seed, n_chains, n_draws, n_warmup
        Override committed defaults; ``None`` = use committed value.
    sequential
        Run NUTS chains sequentially instead of via vmap (NUTS models only).
    smoke
        Tiny-scale run for fast validation.

    Returns
    -------
    dict
        Parsed ``summary_v2.json`` for the generated GT.
    """
    import tempfile
    from pathlib import Path

    summary = load_committed_summary(model_name)
    method = _resolve_gt_method(summary)

    _out = (
        Path(out_dir)
        if out_dir is not None
        else Path(tempfile.mkdtemp(prefix=f"tuningfork_gt_{model_name}_"))
    )

    if method is GTMethod.ANALYTIC_IID:
        from tuningfork.groundtruth._analytic_iid import generate_analytic_iid

        return generate_analytic_iid(
            model_name,
            summary,
            _out,
            seed=seed,
            n_chains=n_chains,
            n_draws=n_draws,
            smoke=smoke,
        )
    elif method in (GTMethod.STANDARD_MULTICHAIN_NUTS, GTMethod.EXPLICIT_POSITIONS):
        from tuningfork.groundtruth._nuts_multichain import generate_nuts_multichain

        return generate_nuts_multichain(
            model_name,
            summary,
            _out,
            seed=seed,
            n_chains=n_chains,
            n_draws=n_draws,
            n_warmup=n_warmup,
            sequential=sequential,
            smoke=smoke,
        )
    elif method is GTMethod.CLOSED_FORM_GP_MARGINAL:
        from tuningfork.groundtruth._gp_marginal import generate_gp_marginal

        return generate_gp_marginal(
            model_name,
            summary,
            _out,
            seed=seed,
            n_chains=n_chains,
            n_draws=n_draws,
            n_warmup=n_warmup,
            sequential=sequential,
            smoke=smoke,
        )
    else:
        raise ValueError(f"Unhandled GT method {method!r} for model {model_name!r}")
