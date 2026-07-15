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
"""CLI for ground-truth regeneration.

Invocation::

    python -m tuningfork.groundtruth generate <model> [options]

For each model, the generation method is determined automatically from the
committed ``summary_v2.json``.  The output is written to ``--out-dir`` (default:
a temporary directory under ``/tmp``).

The tool is self-contained: each model's committed ``summary_v2.json`` embeds
all parameters needed to reproduce the original run (seed, number of chains,
number of draws, warmup steps, target acceptance, etc.).

Environment flags
-----------------
``GT_X64=1``
    Enable 64-bit JAX.  Required for ``gp_regression`` and ``lotka_volterra``.
    Must be set *before* the process starts (sets ``JAX_ENABLE_X64=1`` before
    JAX is imported).  Alternatively: ``JAX_ENABLE_X64=1 python -m ...``.
``OPENBLAS_NUM_THREADS=1``
    Required for ``gp_regression`` on machines with many CPU cores
    (OpenBLAS multi-thread heap corruption on 64+ cores).  Set it in the
    shell before running: ``OPENBLAS_NUM_THREADS=1 python -m ...``.
``PYTHONUNBUFFERED=1``
    Recommended for long-running models (``stoch_vol``, ``lotka_volterra``)
    to ensure per-step progress lines reach the log without buffering.

Reproducibility caveat
----------------------
Regeneration produces statistically equivalent draws (within the same
standard error as the committed GT) but NOT bit-identical copies.
CUDA/XLA nondeterminism (cuBLAS GEMM ordering, GPU clock state) means
float-level values will differ while being statistically indistinguishable.
Use ``--verify`` to confirm equivalence: expect a passing gate and a per-site
coherence z-score below 3.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tuningfork.groundtruth",
        description=(
            "Reproduce ground-truth draws for any tuningfork model.\n\n"
            "The generation method is selected automatically from the model's\n"
            "committed summary_v2.json.  Output is draws.npz + summary_v2.json\n"
            "in the gt_v2_multichain schema.\n\n"
            "Environment flags:\n"
            "  GT_X64=1                Required for gp_regression, lotka_volterra\n"
            "  OPENBLAS_NUM_THREADS=1  Required for gp_regression on many-core machines\n"
            "  PYTHONUNBUFFERED=1      Recommended for long-running models\n\n"
            "Reproducibility: output is statistically equivalent (in-spec) but\n"
            "NOT bit-identical — use --verify to confirm coherence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="Generate ground-truth draws for a model.",
        description=(
            "Generate ground-truth draws for MODEL.\n\n"
            "By default uses the same seed, n_chains, n_draws, and warmup steps\n"
            "as the committed GT so the default invocation reproduces the original\n"
            "configuration.  Use --seed to generate an independent sample set."
        ),
    )
    gen.add_argument(
        "model",
        help=(
            "Model name as it appears in the registry "
            "(e.g. radon, gp_regression, mvn_10)."
        ),
    )
    gen.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for draws.npz and summary_v2.json. "
            "Default: a temporary directory printed to stdout. "
            "To overwrite the committed catalog GT: "
            "tuningfork/tuningfork/catalog/<model>/groundtruth_samples/blackjax/"
        ),
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Master RNG seed. Default: the committed GT seed "
            "(reproduces the original configuration)."
        ),
    )
    gen.add_argument(
        "--n-chains",
        type=int,
        default=None,
        metavar="N",
        help="Number of chains. Default: committed value (10).",
    )
    gen.add_argument(
        "--n-draws",
        type=int,
        default=None,
        metavar="N",
        help="Draws per chain. Default: committed value (10000).",
    )
    gen.add_argument(
        "--n-warmup",
        type=int,
        default=None,
        metavar="N",
        help="Warmup steps per chain (NUTS models). Default: committed value.",
    )
    gen.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Run NUTS chains sequentially instead of in parallel via vmap. "
            "Useful for heterogeneous-cost models. Ignored for analytic models."
        ),
    )
    gen.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny-scale run for fast validation "
            "(2 chains × 50–100 draws × 50–100 warmup). "
            "Overrides --n-chains, --n-draws, --n-warmup."
        ),
    )
    gen.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After generation, check the gate and per-site coherence "
            "vs the committed GT. Prints PASS/FAIL."
        ),
    )
    return parser


def _run_generate(args: argparse.Namespace) -> None:
    from tuningfork.groundtruth._dispatch import (
        GTMethod,
        _resolve_gt_method,
        load_committed_summary,
    )

    model = args.model

    # Validate the model is known before doing any heavy imports
    try:
        summary = load_committed_summary(model)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    method = _resolve_gt_method(summary)

    out_dir: Path
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        _tmp = tempfile.mkdtemp(prefix=f"tuningfork_gt_{model}_")
        out_dir = Path(_tmp)

    print(
        f"[generate] model={model} method={method.value} out_dir={out_dir}", flush=True
    )

    if method is GTMethod.ANALYTIC_IID:
        from tuningfork.groundtruth._analytic_iid import generate_analytic_iid

        result = generate_analytic_iid(
            model,
            summary,
            out_dir,
            seed=args.seed,
            n_chains=args.n_chains,
            n_draws=args.n_draws,
            smoke=args.smoke,
        )
    elif method in (GTMethod.STANDARD_MULTICHAIN_NUTS, GTMethod.EXPLICIT_POSITIONS):
        from tuningfork.groundtruth._nuts_multichain import generate_nuts_multichain

        result = generate_nuts_multichain(
            model,
            summary,
            out_dir,
            seed=args.seed,
            n_chains=args.n_chains,
            n_draws=args.n_draws,
            n_warmup=args.n_warmup,
            sequential=args.sequential,
            smoke=args.smoke,
        )
    elif method is GTMethod.CLOSED_FORM_GP_MARGINAL:
        from tuningfork.groundtruth._gp_marginal import generate_gp_marginal

        result = generate_gp_marginal(
            model,
            summary,
            out_dir,
            seed=args.seed,
            n_chains=args.n_chains,
            n_draws=args.n_draws,
            n_warmup=args.n_warmup,
            smoke=args.smoke,
        )
    else:
        print(f"[error] unhandled method {method!r}", file=sys.stderr)
        sys.exit(1)

    gate = result.get("quality_gate", {})
    gate_pass = gate.get("passed", False)
    print(
        f"[gate] {'PASS' if gate_pass else 'FAIL'} "
        f"max_rhat={gate.get('max_rhat', 'n/a'):.5f} "
        f"min_bulk_ess={gate.get('min_bulk_ess', 'n/a'):.0f} "
        f"total_div={gate.get('total_divergences', 0)}",
        flush=True,
    )
    print(f"[done] draws.npz and summary_v2.json written to {out_dir}", flush=True)

    if args.verify:
        from tuningfork.groundtruth._verify import verify_groundtruth

        draws_path = out_dir / "draws.npz"
        ok = verify_groundtruth(model, result, draws_path)
        if not ok:
            sys.exit(2)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        _run_generate(args)
    else:
        parser.print_help()
        sys.exit(1)
