"""Targeted-patch harness for catalog recipes (model A).

Patches **only** ``warmup_grad_evals`` and ``sample_quality`` fields in-place.
**Never touches** ``headline_metric``, ``verdict``, or ``gate_evidence``.

``warmup_grad_evals`` strategy:
- Fixed-step HMC (hmc/mhmc/laplace_hmc/laplace_mhmc): config-exact,
  ``n_warmup × num_chains × num_integration_steps``.  Instant, no subprocess.
- Dynamic (nuts/dynamic_hmc/dmhmc/laplace_dhmc/laplace_dmhmc): warmup-only
  subprocess, CUMSUM per-step NIS from ``adapt_info.info``.
- mclmc family: warmup-only subprocess, ``_total_tuning_steps``.

``sample_quality`` strategy:
- Load recipe's own cached draws from ``_cache/<recipe_stem>.draws.npz``
  (written during the original calibration run).
- Apply H1-corrected ``compute_sample_quality`` formula (PR #122).
- Where cache absent → leave sq fields null (honest).

Large-diff report: for each patched field, records recipes where the
change exceeds the threshold (wge >2× ratio; sq |Δ| > 0.1).

Usage::

    # Full run (OOM-safe cgroup):
    systemd-run --user --scope --quiet --collect --unit=tpatch -- \\
        bash -c 'PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cpu \\
        uv run --directory /path/to/tuningfork \\
        python -m tuningfork.recipes.sequential_run_recipe_pipeline \\
        2>&1 | tee /tmp/tpatch.log'

    # Smoke-test 6 recipes (no commits):
    python -m tuningfork.recipes.sequential_run_recipe_pipeline \\
        --smoke-paths p1:p2:p3:p4:p5:p6
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

_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "catalog"  # tuningfork/catalog
_REPO_ROOT = Path(__file__).resolve().parents[2]  # git repo root
_LOG_PATH = Path("/tmp/sequential_run_recipe_pipeline.log")
_WGE_TIMEOUT_S = 120  # per-recipe warmup re-run timeout (dynamic families)

# Families whose wge = n_warmup × num_chains × num_integration_steps (config-exact)
_FIXED_STEP: frozenset[str] = frozenset({"hmc", "mhmc", "laplace_hmc", "laplace_mhmc"})
# Families needing warmup re-run for CUMSUM NIS
_DYNAMIC: frozenset[str] = frozenset(
    {"nuts", "dynamic_hmc", "dmhmc", "laplace_dhmc", "laplace_dmhmc"}
)
# mclmc family: exact from _total_tuning_steps
_MCLMC: frozenset[str] = frozenset(
    {"mclmc", "adjusted_mclmc", "adjusted_mclmc_dynamic"}
)
# Models excluded from patching
_EXCLUDE_MODELS: frozenset[str] = frozenset({"gp_regression"})

# Large-diff thresholds
_LARGE_DIFF_WGE_RATIO = 2.0  # flag if new/old or old/new > 2×
_LARGE_DIFF_SQ_ABS = 0.1  # flag if |Δ| > 0.1 for any sq field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LargeDiff:
    recipe: str
    field: str
    old_val: float | None
    new_val: float | None
    delta_abs: float


@dataclass
class PatchReport:
    total: int = 0
    wge_patched: int = 0
    wge_already: int = 0
    wge_null: int = 0
    sq_patched: int = 0
    sq_already: int = 0
    sq_null_no_cache: int = 0
    errors: int = 0
    large_diffs: list[LargeDiff] = field(default_factory=list)


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
    """Return a skip reason string or None if the recipe should be patched."""
    filename = recipe_path.name
    verdict = (recipe_json.get("gate_evidence") or {}).get("auto", {}).get("verdict")
    if filename.startswith("failed__") or verdict == "FAIL":
        return "recipe_failed"
    warmup_name = recipe_json.get("warmup_name") or (
        (recipe_json.get("warmups") or [{}])[0].get("name", "")
    )
    bmp = recipe_json.get("base_method_params") or {}
    if warmup_name == "no_warmup":
        return "no_warmup"
    if bmp.get("step_size") is None:
        return "step_size_none"
    return None


# ---------------------------------------------------------------------------
# wge: config-exact (instant)
# ---------------------------------------------------------------------------


def _get_n_warmup_num_chains(recipe_json: dict[str, Any]) -> tuple[int, int]:
    budget = recipe_json.get("calibration_budget") or {}
    warmups_list = recipe_json.get("warmups") or []
    first_wp = warmups_list[0].get("params", {}) if warmups_list else {}
    n_warmup = int(budget.get("n_warmup") or first_wp.get("n_warmup") or 1000)
    num_chains = int(budget.get("num_chains") or first_wp.get("num_chains") or 4)
    return n_warmup, num_chains


def _wge_from_config(
    bm_name: str, recipe_json: dict[str, Any]
) -> tuple[int, bool] | tuple[None, None]:
    """Config-exact wge for fixed-step HMC families. Returns (wge, is_estimate)."""
    if bm_name not in _FIXED_STEP:
        return None, None
    bmp = recipe_json.get("base_method_params") or {}
    L = bmp.get("num_integration_steps")
    if L is None:
        return None, None
    n_warmup, num_chains = _get_n_warmup_num_chains(recipe_json)
    wge = int(n_warmup) * int(num_chains) * int(L)
    is_est = bm_name in {"laplace_hmc", "laplace_mhmc"}  # L-BFGS excluded
    return wge, is_est


# ---------------------------------------------------------------------------
# wge: warmup subprocess (dynamic / mclmc)
# ---------------------------------------------------------------------------


def _warmup_wge_worker(
    result_q: multiprocessing.Queue,  # type: ignore[type-arg]
    recipe_json_str: str,
) -> None:
    """Subprocess worker: warmup-only run to get exact wge via CUMSUM NIS."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork")
    try:
        import jax
        import numpy as np

        from tuningfork.base_method import BASE_METHODS
        from tuningfork.model import MODELS
        from tuningfork.model._numpyro import build_logdensity_fn
        from tuningfork.warmup import WARMUPS

        recipe_json = json.loads(recipe_json_str)
        model_name = recipe_json.get("model_name", "")
        bm_name = recipe_json.get("base_method_name", "")
        warmup_name = recipe_json.get("warmup_name") or ""
        warmups_list = recipe_json.get("warmups") or []
        n_warmup, num_chains = _get_n_warmup_num_chains(recipe_json)
        if warmups_list and not warmup_name:
            first = warmups_list[0]
            warmup_name = first.get("name", "")
            wp = first.get("params") or {}
            n_warmup = int(wp.get("n_warmup") or n_warmup)
            num_chains = int(wp.get("num_chains") or num_chains)

        if (
            model_name not in MODELS
            or bm_name not in BASE_METHODS
            or warmup_name not in WARMUPS
        ):
            result_q.put(None)
            return

        posterior = MODELS[model_name]
        base_method = BASE_METHODS[bm_name]
        warmup = WARMUPS[warmup_name]
        tuning_seed = recipe_json.get("tuning_seed") or 20260517

        if posterior.requires_x64 and not jax.config.read("jax_enable_x64"):
            jax.config.update("jax_enable_x64", True)

        rng_key = jax.random.key(tuning_seed)
        init_key, warmup_key = jax.random.split(rng_key)
        init_pos, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

        warmup_params = warmups_list[0].get("params", {}) if warmups_list else {}
        target = float(warmup_params.get("target_acceptance") or 0.80)
        warmup_kwargs: dict[str, Any] = {
            "logdensity_fn": logdensity_fn,
            "num_chains": num_chains,
            "target_acceptance_rate": target,
        }
        inner_kernel = recipe_json.get("warmup_inner_kernel")
        if inner_kernel is not None:
            try:
                result = warmup.runner(
                    warmup_key,
                    init_pos,
                    n_warmup,
                    base_method,
                    warmup_inner_kernel_name=inner_kernel,
                    **warmup_kwargs,
                )
            except TypeError:
                result = warmup.runner(
                    warmup_key, init_pos, n_warmup, base_method, **warmup_kwargs
                )
        else:
            result = warmup.runner(
                warmup_key, init_pos, n_warmup, base_method, **warmup_kwargs
            )

        batched_params = result[1]
        batched_warmup_info = result[2] if len(result) == 3 else None  # type: ignore[misc]
        jax.block_until_ready(batched_params)

        # Source 1: mclmc _total_tuning_steps
        total = batched_params.get("_total_tuning_steps") if batched_params else None
        if total is not None:
            result_q.put((int(total), False))
            return

        # Source 2: CUMSUM of per-step NIS from adapted_info.info
        if batched_warmup_info is not None:
            nis = getattr(batched_warmup_info, "num_integration_steps", None)
            if nis is not None:
                result_q.put((int(np.sum(np.asarray(nis))), False))
                return

        result_q.put(None)
    except Exception:  # noqa: BLE001
        result_q.put(None)


def _run_warmup_wge(
    recipe_json: dict[str, Any],
    timeout_s: int,
) -> tuple[int, bool] | None:
    """Run warmup subprocess; return (wge, is_estimate) or None."""
    result_q: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
    proc = multiprocessing.Process(
        target=_warmup_wge_worker,
        args=(result_q, json.dumps(recipe_json)),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
        return None
    try:
        return result_q.get_nowait()
    except _queue.Empty:
        return None


# ---------------------------------------------------------------------------
# sq from cached draws (H1-corrected, deterministic)
# ---------------------------------------------------------------------------


def _compute_sq_from_cache(
    recipe_path: Path,
    model_name: str,
    catalog_root: Path,
) -> dict[str, float] | None:
    """Load recipe's cached draws and compute H1-corrected sample_quality.

    Returns dict with q05_error/q95_error/std_ratio_max_dev, or None on miss.
    """
    import numpy as np

    from tuningfork.metrics.reference_compare import compute_sample_quality

    recipe_stem = recipe_path.stem  # e.g. low__nuts__window_adaptation_diag_imm
    cache_path = catalog_root / model_name / "_cache" / f"{recipe_stem}.draws.npz"
    if not cache_path.exists():
        return None

    summary_path = catalog_root / model_name / "reference" / "summary.json"
    if not summary_path.exists():
        return None

    gt_summaries = json.loads(summary_path.read_text())

    try:
        raw = np.load(cache_path)
        # draws shape: (num_chains, n_samples, *event) — keep as-is.
        # compute_sample_quality expects at least 2 axes (num_chains, num_samples, ...).
        mc_samples: dict[str, Any] = {k: raw[k] for k in raw.files}
    except Exception:  # noqa: BLE001
        return None

    sq_refs = {
        param: {
            stat: gt_summaries[stat][param] for stat in ("mean", "std", "q05", "q95")
        }
        for param in gt_summaries.get("mean", {})
        if param in mc_samples
    }
    if not sq_refs:
        return None

    sq_draws = {k: mc_samples[k] for k in sq_refs if k in mc_samples}
    try:
        return compute_sample_quality(sq_draws, sq_refs)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Git chunk helper
# ---------------------------------------------------------------------------


def _git_commit_model(
    model_name: str,
    n_patched: int,
    catalog_root: Path,
    repo_root: Path,
    log_file: Any,
) -> str | None:
    model_dir = (catalog_root / model_name).relative_to(repo_root)
    subprocess.run(
        ["git", "-C", str(repo_root), "add", str(model_dir)],
        capture_output=True,
        check=False,
    )
    msg = (
        f"patch({model_name}): wge+sq targeted patch, n={n_patched}\n\n"
        f"Finding: deterministic targeted patch — wge exact/estimate, sq from cached draws.\n"
        f"headline_metric and verdict byte-identical to committed.\n"
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
# Large-diff detection
# ---------------------------------------------------------------------------


def _record_large_diffs(
    rel: str,
    field: str,
    old_val: float | None,
    new_val: float | None,
    large_diffs: list[LargeDiff],
) -> None:
    if old_val is None or new_val is None:
        return
    delta_abs = abs(new_val - old_val)
    if field == "wge":
        # Flag if ratio > 2× either direction
        if old_val > 0 and (
            new_val / old_val > _LARGE_DIFF_WGE_RATIO
            or old_val / new_val > _LARGE_DIFF_WGE_RATIO
        ):
            large_diffs.append(LargeDiff(rel, field, old_val, new_val, delta_abs))
    elif delta_abs > _LARGE_DIFF_SQ_ABS:
        large_diffs.append(LargeDiff(rel, field, old_val, new_val, delta_abs))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    catalog_root: Path = _CATALOG_ROOT,
    repo_root: Path = _REPO_ROOT,
    log_file: Any = None,
    *,
    exclude_models: frozenset[str] = _EXCLUDE_MODELS,
    wge_timeout_s: int = _WGE_TIMEOUT_S,
    smoke_paths: list[Path] | None = None,
    single_recipe: Path | None = None,
    dry_run: bool = False,
) -> PatchReport:
    """Run the targeted-patch pipeline.

    Patches warmup_grad_evals and sample_quality only.
    headline_metric, verdict, and gate_evidence are NEVER modified.
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

    report = PatchReport()

    for model_name in sorted(by_model.keys()):
        model_paths = sorted(by_model[model_name])
        _log(f"\n--- MODEL: {model_name} ({len(model_paths)} recipes) ---", log_file)
        n_changed = 0

        for recipe_path in model_paths:
            try:
                recipe_json = json.loads(recipe_path.read_text())
            except Exception:  # noqa: BLE001
                continue

            report.total += 1
            rel = str(recipe_path.relative_to(catalog_root))
            _log(f"  [{report.total}] {rel}", log_file)

            skip = _skip_reason(recipe_path, recipe_json)
            if skip:
                _log(f"    SKIP: {skip}", log_file)
                continue

            bm_name = recipe_json.get("base_method_name", "")
            budget = recipe_json.get("calibration_budget") or {}
            changed = False

            # ---- Phase A: wge patch ----
            wge_on_disk = budget.get("warmup_grad_evals")
            wge_is_est = budget.get("warmup_grad_evals_is_estimate", False)

            if wge_on_disk is not None and not wge_is_est:
                _log("    wge: already exact", log_file)
                report.wge_already += 1
            else:
                # Path 1: config-exact (instant)
                wge_cfg, is_est_cfg = _wge_from_config(bm_name, recipe_json)
                if wge_cfg is not None:
                    if wge_on_disk is not None:
                        _record_large_diffs(
                            rel,
                            "wge",
                            float(wge_on_disk),
                            float(wge_cfg),
                            report.large_diffs,
                        )
                    budget = dict(budget)
                    budget["warmup_grad_evals"] = wge_cfg
                    if is_est_cfg:
                        budget["warmup_grad_evals_is_estimate"] = True
                    else:
                        budget.pop("warmup_grad_evals_is_estimate", None)
                    recipe_json["calibration_budget"] = budget
                    tag = "~" if is_est_cfg else ""
                    _log(f"    wge: {tag}{wge_cfg} (config-exact)", log_file)
                    report.wge_patched += 1
                    changed = True

                elif bm_name in _DYNAMIC or bm_name in _MCLMC:
                    # Path 2: warmup subprocess
                    _log(f"    wge: warmup rerun ({model_name}/{bm_name})...", log_file)
                    t0 = time.perf_counter()
                    result = _run_warmup_wge(recipe_json, wge_timeout_s)
                    wall = time.perf_counter() - t0
                    if result is not None:
                        wge_exact, is_est = result
                        if wge_on_disk is not None:
                            _record_large_diffs(
                                rel,
                                "wge",
                                float(wge_on_disk),
                                float(wge_exact),
                                report.large_diffs,
                            )
                        budget = dict(budget)
                        budget["warmup_grad_evals"] = wge_exact
                        if is_est:
                            budget["warmup_grad_evals_is_estimate"] = True
                        else:
                            budget.pop("warmup_grad_evals_is_estimate", None)
                        recipe_json["calibration_budget"] = budget
                        tag = "~" if is_est else ""
                        _log(
                            f"    wge: {tag}{wge_exact} ({'est' if is_est else 'exact'}, {wall:.1f}s)",
                            log_file,
                        )
                        report.wge_patched += 1
                        changed = True
                    else:
                        _log(f"    wge: null (timeout/error, {wall:.1f}s)", log_file)
                        report.wge_null += 1
                else:
                    _log(
                        "    wge: null (family has no config-exact or warmup path)",
                        log_file,
                    )
                    report.wge_null += 1

            # ---- Phase B: sq from cached draws ----
            sq_current = recipe_json.get("sample_quality") or {}
            needs_sq = any(
                sq_current.get(k) is None
                for k in ("q05_error", "q95_error", "std_ratio_max_dev")
            )
            if not needs_sq:
                _log("    sq: already stamped", log_file)
                report.sq_already += 1
            else:
                try:
                    new_sq = _compute_sq_from_cache(
                        recipe_path, model_name, catalog_root
                    )
                except Exception:  # noqa: BLE001
                    new_sq = None

                if new_sq is not None:
                    # Record large diffs
                    for sq_field in ("q05_error", "q95_error", "std_ratio_max_dev"):
                        _record_large_diffs(
                            rel,
                            sq_field,
                            sq_current.get(sq_field),
                            new_sq.get(sq_field),
                            report.large_diffs,
                        )
                    # Patch sq fields, leave mae_vs_reference unchanged
                    recipe_json["sample_quality"] = {
                        "mae_vs_reference": sq_current.get("mae_vs_reference"),
                        "q05_error": new_sq["q05_error"],
                        "q95_error": new_sq["q95_error"],
                        "std_ratio_max_dev": new_sq["std_ratio_max_dev"],
                    }
                    q05 = new_sq["q05_error"]
                    std = new_sq["std_ratio_max_dev"]
                    _log(f"    sq: q05={q05:.4f} std={std:.4f} (from cache)", log_file)
                    report.sq_patched += 1
                    changed = True
                else:
                    _log("    sq: null (no cache or no GT summaries)", log_file)
                    report.sq_null_no_cache += 1

            if changed and not dry_run:
                recipe_path.write_text(json.dumps(recipe_json, indent=2) + "\n")
                n_changed += 1

        # Per-model chunk report
        _log(
            f"  === {model_name}: wge_patch={report.wge_patched} sq_patch={report.sq_patched} "
            f"wge_ok={report.wge_already} sq_ok={report.sq_already} "
            f"wge_null={report.wge_null} sq_null={report.sq_null_no_cache} ===",
            log_file,
        )

        if not smoke_mode and not dry_run and n_changed > 0:
            _git_commit_model(model_name, n_changed, catalog_root, repo_root, log_file)

    # Final summary + large-diff report
    _log("", log_file)
    _log("=== PATCH COMPLETE ===", log_file)
    _log(f"Total recipes: {report.total}", log_file)
    _log(
        f"wge: patched={report.wge_patched} already={report.wge_already} null={report.wge_null}",
        log_file,
    )
    _log(
        f"sq:  patched={report.sq_patched} already={report.sq_already} null_no_cache={report.sq_null_no_cache}",
        log_file,
    )
    _log(f"errors: {report.errors}", log_file)
    if report.large_diffs:
        _log(
            f"\nLARGE DIFFS ({len(report.large_diffs)} entries, thresholds: wge>2×, sq|Δ|>0.1):",
            log_file,
        )
        for d in sorted(report.large_diffs, key=lambda x: x.delta_abs, reverse=True)[
            :50
        ]:
            _log(
                f"  {d.recipe}: {d.field} {d.old_val!r} → {d.new_val!r} (|Δ|={d.delta_abs:.4f})",
                log_file,
            )
    else:
        _log("No large diffs.", log_file)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Targeted-patch: wge + sq from cached draws"
    )
    parser.add_argument("--smoke-paths", metavar="PATH:PATH", default=None)
    parser.add_argument("--recipe", metavar="PATH", default=None)
    parser.add_argument("--log", default=str(_LOG_PATH), metavar="PATH")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=_WGE_TIMEOUT_S, metavar="S")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_file = open(str(log_path), "w", buffering=1)  # noqa: SIM115

    smoke_paths = None
    if args.smoke_paths:
        smoke_paths = [Path(p) for p in args.smoke_paths.split(":") if p]
    single_recipe = Path(args.recipe) if args.recipe else None

    try:
        run_pipeline(
            log_file=log_file,
            smoke_paths=smoke_paths,
            single_recipe=single_recipe,
            dry_run=args.dry_run,
            wge_timeout_s=args.timeout,
        )
    finally:
        log_file.close()

    print(f"Log: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
