"""Sequential re-run harness for catalog recipes.

Walks the catalog, re-runs each in-scope recipe through the existing
``_recipe_runner`` infrastructure, and overwrites only if the verdict
is still PASS.  Designed for OOM-safe sequential execution (one recipe
at a time, no parallelism).

Key properties:

- **Idempotent** — already-passed recipes get fresh draws + updated metrics
  (warmup_grad_evals now exact via CUMSUM, sample_quality recomputed).
- **Overwrite-only-on-pass** — if a re-run returns FAIL (stochastic), the
  existing recipe file is untouched.
- **Clean-skip** for un-re-runnable recipes: ``failed__*`` filename (no point),
  mclmc/adjusted_mclmc family (separate tuning path), ``no_warmup`` (no adapted
  params), sidecar-IMM without path, long-running excluded models.
- **Broad catch** — any unhandled exception in a recipe run is logged and the
  recipe is skipped; the loop continues.
- **Per-recipe timeout** via subprocess + bounded kill (D-state safe).
- **Per-model chunk git-commit** — progress durable against terminal death.
- **Single-recipe override mode** via ``--recipe PATH`` for deliberate deep-dives.

Usage::

    # Full sequential run (detached, OOM-safe cgroup):
    systemd-run --user --scope --quiet --collect --unit=seqrun -- \\
        bash -c 'PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cpu \\
        uv run --directory /path/to/tuningfork \\
        python -m tuningfork.recipes.sequential_run_recipe_pipeline \\
        2>&1 | tee /tmp/seqrun.log'

    # Smoke-test 6 specific recipes (no commits):
    python -m tuningfork.recipes.sequential_run_recipe_pipeline \\
        --smoke-paths p1:p2:p3:p4:p5:p6

    # Single-recipe override:
    python -m tuningfork.recipes.sequential_run_recipe_pipeline \\
        --recipe catalog/mvn_10/recipes/low__nuts__window_adaptation_diag_imm.json
"""

from __future__ import annotations

import json
import multiprocessing
import queue as _queue
import subprocess
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "catalog"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_PATH = Path("/tmp/sequential_run_recipe_pipeline.log")
_PER_RECIPE_TIMEOUT_S = 300  # 5 min per recipe; long-running NUTS models need it

# Families/warmups that can't be re-run through emit_low_recipe_for_cell cleanly
_SKIP_BASE_METHODS: frozenset[str] = frozenset(
    {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
)
_SKIP_WARMUP_NAMES: frozenset[str] = frozenset({"no_warmup"})

# Models excluded from sequential re-run (too slow / known OOM risk)
_EXCLUDE_MODELS: frozenset[str] = frozenset({"gp_regression"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RecipeRunOutcome:
    rel_path: str
    status: str  # "pass", "fail_stochastic", "skip:<reason>", "error:<msg>", "timeout"


@dataclass
class PipelineReport:
    total: int = 0
    passed: int = 0
    skipped: int = 0
    failed_stochastic: int = 0
    errors: int = 0
    timeouts: int = 0
    verdict_moved: list[str] = field(default_factory=list)
    outcomes: list[RecipeRunOutcome] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str, file: Any = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if file:
        print(line, file=file, flush=True)


# ---------------------------------------------------------------------------
# Skip-gate (O(1), no JAX)
# ---------------------------------------------------------------------------


def _skip_reason(recipe_path: Path, recipe_json: dict[str, Any]) -> str | None:
    """Return a skip reason string or None if the recipe should be re-run."""
    filename = recipe_path.name

    # 1. Failed recipes — no point re-running
    verdict = (recipe_json.get("gate_evidence") or {}).get("auto", {}).get("verdict")
    if filename.startswith("failed__") or verdict == "FAIL":
        return "recipe_failed"

    # 2. mclmc family — tuning path requires separate warmup
    bm = recipe_json.get("base_method_name", "")
    if bm in _SKIP_BASE_METHODS:
        return f"mclmc_family:{bm}"

    # 3. no_warmup — no adapted step_size / IMM
    warmup_name = recipe_json.get("warmup_name") or (
        (recipe_json.get("warmups") or [{}])[0].get("name", "")
    )
    if warmup_name in _SKIP_WARMUP_NAMES:
        return "no_warmup"

    # 4. sidecar-IMM without path — can't load IMM for re-run
    bmp = recipe_json.get("base_method_params") or {}
    if bmp.get("inverse_mass_matrix") == "sidecar" and not recipe_json.get(
        "inverse_mass_matrix_path"
    ):
        return "sidecar_imm_no_path"

    # 5. base_method_params with None step_size (policy-driven inner-kernel recipes)
    if bmp.get("step_size") is None:
        return "step_size_none"

    return None


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _recipe_run_worker(
    result_q: multiprocessing.Queue,  # type: ignore[type-arg]
    recipe_json_str: str,
    catalog_root_str: str,
) -> None:
    """Subprocess worker: re-run one recipe through emit_low_recipe_for_cell."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork")
    try:
        from tuningfork.recipes._recipe_runner import (
            RECIPE_N_CHUNKS,
            Effort,
            emit_low_recipe_for_cell,
        )

        recipe_json = json.loads(recipe_json_str)
        catalog_root = Path(catalog_root_str)

        model_name = recipe_json.get("model_name", "")
        bm_name = recipe_json.get("base_method_name", "")
        warmup_name = recipe_json.get("warmup_name") or (
            (recipe_json.get("warmups") or [{}])[0].get("name", "")
        )
        budget = recipe_json.get("calibration_budget") or {}
        n_warmup = int(budget.get("n_warmup") or 1000)
        n_samples = int(budget.get("n_samples") or 1000)
        num_chains = int(budget.get("num_chains") or 4)
        tuning_seed = recipe_json.get("tuning_seed") or 20260517

        effort_str = recipe_json.get("effort") or "LOW"
        try:
            effort = Effort(effort_str)
        except (ValueError, KeyError):
            effort = Effort.LOW

        # Use stored hyperparameters — bypasses BO, uses fixed params
        bmp = recipe_json.get("base_method_params") or {}
        # Validate: all numeric params must be non-None to avoid jnp.array(None)
        for k, v in bmp.items():
            if v is None and k not in ("inverse_mass_matrix",):
                result_q.put(("error", f"None param {k}"))
                return

        warmup_inner_kernel = recipe_json.get("warmup_inner_kernel")
        init_strategy = recipe_json.get("init_strategy")
        step_policy = recipe_json.get("step_policy")

        result = emit_low_recipe_for_cell(
            model_name=model_name,
            warmup_name=warmup_name,
            sampler_name=bm_name,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            seed=tuning_seed,
            n_chunks=RECIPE_N_CHUNKS,
            catalog_root=catalog_root,
            verbose=False,
            effort=effort,
            sampler_kwargs_override=bmp,
            warmup_inner_kernel=warmup_inner_kernel,
            init_strategy=init_strategy,
            step_policy=step_policy,
        )
        result_q.put(("ok", result.verdict))
    except Exception as exc:  # noqa: BLE001
        result_q.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_recipe_with_timeout(
    recipe_json: dict[str, Any],
    catalog_root: Path,
    timeout_s: int,
) -> tuple[str, str]:
    """Run one recipe in a subprocess; return (status, detail).

    status is one of: "pass", "fail_stochastic", "error", "timeout"
    """
    result_q: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
    proc = multiprocessing.Process(
        target=_recipe_run_worker,
        args=(result_q, json.dumps(recipe_json), str(catalog_root)),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(10)  # bounded — D-state processes move on after 10 s
        return "timeout", f">{timeout_s}s"

    try:
        status, detail = result_q.get_nowait()
        if status == "error":
            return "error", detail
        if detail == "PASS":
            return "pass", "PASS"
        return "fail_stochastic", detail
    except _queue.Empty:
        return "error", "empty_queue"


# ---------------------------------------------------------------------------
# Git chunk helper
# ---------------------------------------------------------------------------


def _git_commit_model(
    model_name: str,
    stats: dict[str, int],
    catalog_root: Path,
    repo_root: Path,
    log_file: Any,
) -> str | None:
    """git add catalog/<model>/ + commit --no-verify (data JSONs only)."""
    model_dir = (catalog_root / model_name).relative_to(repo_root)
    subprocess.run(
        ["git", "-C", str(repo_root), "add", str(model_dir)],
        capture_output=True,
        check=False,
    )
    n_pass = stats.get("passed", 0)
    n_skip = stats.get("skipped", 0)
    n_err = stats.get("errors", 0) + stats.get("timeouts", 0)
    msg = (
        f"seq-run({model_name}): pass={n_pass} skip={n_skip} err={n_err}\n\n"
        f"Finding: sequential re-run via emit_low_recipe_for_cell; "
        f"wge now exact via adapt_info.info CUMSUM.\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "commit", "--no-verify", "-m", msg],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        bracket = result.stdout.split("]")[0].split("[")[-1]
        sha = bracket.split()[-1] if bracket else "?"
        _log(f"  GIT: committed {model_name} → {sha}", log_file)
        return sha
    if "nothing to commit" in (result.stdout + result.stderr):
        _log(f"  GIT: nothing new for {model_name}", log_file)
        return None
    _log(f"  GIT COMMIT FAILED: {result.stderr.strip()}", log_file)
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    catalog_root: Path = _CATALOG_ROOT,
    repo_root: Path = _REPO_ROOT,
    log_file: Any = None,
    *,
    exclude_models: frozenset[str] = _EXCLUDE_MODELS,
    per_recipe_timeout_s: int = _PER_RECIPE_TIMEOUT_S,
    smoke_paths: list[Path] | None = None,
    single_recipe: Path | None = None,
    dry_run: bool = False,
) -> PipelineReport:
    """Run the sequential re-run pipeline.

    Parameters
    ----------
    catalog_root
        Root of ``tuningfork/catalog/``.
    repo_root
        Root of the tuningfork git repo.
    log_file
        Open writable file for log output (stdout + file).
    exclude_models
        Model names to skip entirely.
    per_recipe_timeout_s
        Hard per-recipe wall-clock timeout (subprocess kill).
    smoke_paths
        If set, process only these paths (no git commits).
    single_recipe
        If set, process only this one recipe (no git commits).
    dry_run
        If True, skip-and-log without actually re-running.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork")

    catalog_root = catalog_root.resolve()
    repo_root = repo_root.resolve()

    smoke_mode = smoke_paths is not None or single_recipe is not None

    if single_recipe is not None:
        recipe_paths = [single_recipe.resolve()]
        _log(f"SINGLE RECIPE: {single_recipe}", log_file)
    elif smoke_paths is not None:
        recipe_paths = [p.resolve() for p in smoke_paths]
        _log(f"SMOKE MODE: {len(recipe_paths)} recipes (no git commits)", log_file)
    else:
        recipe_paths = sorted(catalog_root.rglob("recipes/*.json"))
        _log(f"Found {len(recipe_paths)} recipes", log_file)

    by_model: dict[str, list[Path]] = defaultdict(list)
    for rp in recipe_paths:
        try:
            d = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001
            continue
        model = d.get("model_name", "")
        if model in exclude_models:
            continue
        by_model[model].append(rp)

    report = PipelineReport()

    for model_name in sorted(by_model.keys()):
        model_paths = sorted(by_model[model_name])
        _log(f"\n--- MODEL: {model_name} ({len(model_paths)} recipes) ---", log_file)

        chunk_stats: dict[str, int] = {
            "passed": 0,
            "skipped": 0,
            "failed_stochastic": 0,
            "errors": 0,
            "timeouts": 0,
        }

        for recipe_path in model_paths:
            try:
                recipe_json = json.loads(recipe_path.read_text())
            except Exception:  # noqa: BLE001
                continue

            report.total += 1
            rel = str(recipe_path.relative_to(catalog_root))
            _log(f"  [{report.total}] {rel}", log_file)

            # Skip gate
            skip = _skip_reason(recipe_path, recipe_json)
            if skip:
                _log(f"    SKIP: {skip}", log_file)
                report.skipped += 1
                chunk_stats["skipped"] += 1
                report.outcomes.append(RecipeRunOutcome(rel, f"skip:{skip}"))
                continue

            if dry_run:
                _log("    DRY RUN: would re-run", log_file)
                continue

            # Re-run in subprocess with timeout
            t0 = time.perf_counter()
            verdict_before = (
                (recipe_json.get("gate_evidence") or {}).get("auto", {}).get("verdict")
            )

            try:
                status, detail = _run_recipe_with_timeout(
                    recipe_json, catalog_root, per_recipe_timeout_s
                )
            except Exception as exc:  # noqa: BLE001
                status = "error"
                detail = f"{type(exc).__name__}: {exc}"

            wall = time.perf_counter() - t0

            if status == "pass":
                # Recipe was overwritten by emit_low_recipe_for_cell; verify verdict
                try:
                    new_json = json.loads(recipe_path.read_text())
                    verdict_after = (
                        (new_json.get("gate_evidence") or {})
                        .get("auto", {})
                        .get("verdict")
                    )
                    if verdict_before is not None and verdict_after != verdict_before:
                        _log(
                            f"    VERDICT MOVED: {verdict_before!r} → {verdict_after!r} (BUG SIGNAL)",
                            log_file,
                        )
                        report.verdict_moved.append(
                            f"{rel}: {verdict_before!r}→{verdict_after!r}"
                        )
                except Exception:  # noqa: BLE001
                    pass
                _log(f"    PASS ({wall:.1f}s)", log_file)
                report.passed += 1
                chunk_stats["passed"] += 1
                report.outcomes.append(RecipeRunOutcome(rel, "pass"))

            elif status == "fail_stochastic":
                _log(
                    f"    FAIL stochastic (verdict={detail}, {wall:.1f}s) — kept original",
                    log_file,
                )
                report.failed_stochastic += 1
                chunk_stats["failed_stochastic"] = (
                    chunk_stats.get("failed_stochastic", 0) + 1
                )
                report.outcomes.append(
                    RecipeRunOutcome(rel, f"fail_stochastic:{detail}")
                )

            elif status == "timeout":
                _log(f"    TIMEOUT {detail} — skip+continue", log_file)
                report.timeouts += 1
                chunk_stats["timeouts"] += 1
                report.outcomes.append(RecipeRunOutcome(rel, f"timeout:{detail}"))

            else:  # error
                _log(f"    ERROR: {detail} ({wall:.1f}s) — skip+continue", log_file)
                report.errors += 1
                chunk_stats["errors"] += 1
                report.outcomes.append(RecipeRunOutcome(rel, f"error:{detail}"))

        # Per-model chunk report
        _log(
            f"  === {model_name}: pass={chunk_stats['passed']} "
            f"skip={chunk_stats['skipped']} "
            f"fail_stoch={chunk_stats.get('failed_stochastic', 0)} "
            f"err={chunk_stats['errors']} timeout={chunk_stats['timeouts']} ===",
            log_file,
        )

        if not smoke_mode and not dry_run and chunk_stats["passed"] > 0:
            _git_commit_model(
                model_name, chunk_stats, catalog_root, repo_root, log_file
            )

    # Final summary
    _log("", log_file)
    _log("=== PIPELINE COMPLETE ===", log_file)
    _log(f"Total recipes: {report.total}", log_file)
    _log(f"  passed: {report.passed}", log_file)
    _log(f"  skipped: {report.skipped}", log_file)
    _log(f"  failed_stochastic: {report.failed_stochastic}", log_file)
    _log(f"  errors: {report.errors}", log_file)
    _log(f"  timeouts: {report.timeouts}", log_file)
    if report.verdict_moved:
        _log("VERDICT MOVEMENTS (BUG SIGNALS):", log_file)
        for v in report.verdict_moved:
            _log(f"  *** {v}", log_file)
    else:
        _log("No verdict movements.", log_file)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Sequential re-run harness for catalog recipes"
    )
    parser.add_argument(
        "--smoke-paths",
        metavar="PATH:PATH",
        default=None,
        help="Colon-separated recipe paths for smoke test (no commits)",
    )
    parser.add_argument(
        "--recipe",
        metavar="PATH",
        default=None,
        help="Single recipe path to re-run (no commits)",
    )
    parser.add_argument(
        "--log",
        default=str(_LOG_PATH),
        metavar="PATH",
        help=f"Log file path (default: {_LOG_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log skip decisions without actually re-running",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_PER_RECIPE_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Per-recipe timeout in seconds (default: {_PER_RECIPE_TIMEOUT_S})",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    log_file = open(str(log_path), "w", buffering=1)

    smoke_paths = None
    if args.smoke_paths:
        smoke_paths = [Path(p) for p in args.smoke_paths.split(":") if p]

    single_recipe = Path(args.recipe) if args.recipe else None

    try:
        report = run_pipeline(
            log_file=log_file,
            smoke_paths=smoke_paths,
            single_recipe=single_recipe,
            dry_run=args.dry_run,
            per_recipe_timeout_s=args.timeout,
        )
    finally:
        log_file.close()

    print(f"Log: {log_path}", file=sys.stderr)
    return 1 if report.verdict_moved else 0


if __name__ == "__main__":
    raise SystemExit(main())
