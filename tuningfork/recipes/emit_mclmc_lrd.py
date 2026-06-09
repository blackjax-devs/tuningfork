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
"""Private helpers for MCLMC-LRD calibration sweeps.

``_emit_lrd_cert_sweep`` is the internal cert-sweep implementation delegated
to by ``tuningfork.recipes._generate_starter.emit_mclmc_lrd_recipes`` when
``calibrate=True``.  Do not call directly — use
``_generate_starter.emit_mclmc_lrd_recipes(calibrate=True)`` instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from tuningfork.base_method import BASE_METHODS
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from tuningfork.recipes._base import Effort, Recipe
from tuningfork.recipes._sweep_runner import (
    GATE_DIV_RATE_PASS,
    GATE_ESS_PASS,
    GATE_RHAT_PASS,
)
from tuningfork.warmup import WARMUPS

# Default catalog root: tuningfork/catalog/ relative to this file.
_DEFAULT_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "catalog"


def _emit_lrd_cert_sweep(
    model_names: list[str],
    *,
    cert_seeds: tuple[int, ...] = (11111, 22222, 33333),
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
    k_rank: int = 40,
    catalog_root: Path | None = None,
    variant_label: str = "mclmc_lrd",
    effort: Effort = Effort.LOW,
    tuningfork_version: str = "0.0.0.dev0",
) -> list[Path]:
    """Run the LRD cert-sweep for a list of models and emit calibrated recipes.

    This is an internal helper — call
    ``tuningfork.recipes._generate_starter.emit_mclmc_lrd_recipes(calibrate=True)``
    instead of invoking this directly.

    For each model: run warmup + 4-chain sampling for each seed in
    ``cert_seeds``; gate on R̂/ESS/div (GATE_RHAT_PASS/GATE_ESS_PASS/
    GATE_DIV_RATE_PASS from ``_sweep_runner``); if ≥ 2/3 PASS, bake the
    best PASS seed into a LOW recipe with ``bake_warmup=True`` and save the
    LRD IMM sidecar.

    Parameters
    ----------
    model_names
        List of model registry names to sweep (e.g., ``["ill_cond_50"]``).
    cert_seeds
        Random seeds for the certification sweep.
    n_warmup
        Number of steps for both the pilot NUTS warmup and MCLMC tuning.
    n_samples
        Post-warmup samples per chain for the gate check.
    num_chains
        Number of sampling chains for the gate check.  The warmup always
        uses ``num_chains=1`` (via ``from_warmup_only``).
    k_rank
        LRD approximation rank.
    catalog_root
        Root directory for the catalog.  Defaults to ``tuningfork/catalog/``.
    variant_label
        Filename-stem label for the emitted recipes (default ``"mclmc_lrd"``).
    effort
        Effort tier for recipes that PASS (default ``Effort.LOW``).
    tuningfork_version
        Version string to embed in recipe provenance.

    Returns
    -------
    list[Path]
        Paths of the written recipe JSON files (one per model).

    Raises
    ------
    KeyError
        If a model name is not in ``MODELS`` or if ``"mclmc"`` / ``"mclmc_lrd_tuning"``
        are missing from ``BASE_METHODS`` / ``WARMUPS``.
    """
    root = Path(catalog_root) if catalog_root is not None else _DEFAULT_CATALOG_ROOT
    warmup = WARMUPS["mclmc_lrd_tuning"]
    base_method = BASE_METHODS["mclmc"]

    written_paths: list[Path] = []

    for model_name in model_names:
        posterior = MODELS[model_name]

        # ── Cert sweep ───────────────────────────────────────────────────────
        seed_results: list[dict[str, Any]] = []

        for seed in cert_seeds:
            _result = _run_cert_seed(
                seed=seed,
                posterior=posterior,
                base_method=base_method,
                warmup=warmup,
                n_warmup=n_warmup,
                n_samples=n_samples,
                num_chains=num_chains,
                k_rank=k_rank,
                tuningfork_version=tuningfork_version,
                variant_label=variant_label,
            )
            seed_results.append(_result)

        # Count PASSing seeds.
        passing = [r for r in seed_results if r["verdict"] == "PASS"]
        pass_count = len(passing)
        total = len(seed_results)

        if pass_count >= (total + 1) // 2:  # ≥ ceil(total/2) = 2 out of 3
            # Bake the best PASS seed (highest ESS/grad).
            best = max(passing, key=lambda r: r["ess_per_grad"])
            recipe = Recipe.from_warmup_only(
                posterior,
                base_method,
                warmup,
                n_warmup=n_warmup,
                rng_key=jax.random.key(best["seed"]),
                tuningfork_version=tuningfork_version,
                effort=effort,
                headline_metric=best["ess_per_grad"],
                bake_warmup=True,
                attempted_configurations=[_result_to_dict(r) for r in seed_results],
                notes=(
                    f"LRD-MCLMC calibrated: {pass_count}/{total} seeds PASS. "
                    f"Best seed {best['seed']}: ESS/grad={best['ess_per_grad']:.4f}, "
                    f"R-hat={best['rhat_max']:.4f}, minESS={best['min_bulk_ess']:.0f}."
                ),
                variant_label=variant_label,
            )
            # Patch gate_evidence with the best-seed diagnostics.
            import dataclasses

            gate_evidence = {
                "auto": {
                    "rhat_max": best["rhat_max"],
                    "min_bulk_ess": best["min_bulk_ess"],
                    "n_divergences": best["n_divergences"],
                    "max_abs_mean_z": None,
                    "verdict": "PASS",
                    "ess_per_grad": best["ess_per_grad"],
                    "total_grad_evals": best["total_grad_evals"],
                    "seed": best["seed"],
                    "margins": {},
                },
                "override": {
                    "reason": "",
                    "statistician_id": "",
                    "decision": "",
                },
            }
            recipe = dataclasses.replace(recipe, gate_evidence=gate_evidence)
            path = recipe.save(root, imm_sidecar="auto")
        else:
            # < 2/3 PASS: emit a FAILED recipe recording the attempted configs.
            import dataclasses

            from tuningfork.recipes._instructions import render_instructions

            recipe_kwargs: dict[str, Any] = dict(
                model_name=posterior.name,
                base_method_name=base_method.name,
                warmup_name="mclmc_lrd_tuning",
                effort=Effort.FAILED,
                base_method_params={},
                warmup_params={"n_warmup": n_warmup},
                warmups=[
                    {"name": "mclmc_lrd_tuning", "params": {"n_warmup": n_warmup}}
                ],
                headline_metric=None,
                sample_quality=None,
                calibration_budget={
                    "trials": len(seed_results),
                    "wall_seconds_estimate": sum(
                        r.get("wall_seconds", 0.0) for r in seed_results
                    ),
                    "n_warmup": n_warmup,
                    "seed_evidence": [_result_to_dict(r) for r in seed_results],
                },
                difficulty=None,
                instructions="",
                notes=(
                    f"LRD-MCLMC cert FAILED: {pass_count}/{total} seeds PASS "
                    f"(need ≥{(total + 1) // 2}).  See calibration_budget.seed_evidence."
                ),
                variant_label=variant_label,
                tuning_seed=cert_seeds[0],
                tuningfork_version=tuningfork_version,
                blackjax_version=_get_blackjax_version(),
                jax_version=_get_jax_version(),
                timestamp_utc=_now_utc_iso(),
            )
            provisional = Recipe(**recipe_kwargs)
            recipe_kwargs["instructions"] = render_instructions(provisional)
            recipe = Recipe(**recipe_kwargs)
            path = recipe.save(root)

        written_paths.append(path)

    return written_paths


# ── private helpers ──────────────────────────────────────────────────────────


def _run_cert_seed(
    *,
    seed: int,
    posterior: Any,
    base_method: Any,
    warmup: Any,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    k_rank: int,
    tuningfork_version: str,
    variant_label: str,
) -> dict[str, Any]:
    """Run one cert-seed: warmup + 4-chain sampling → diagnostics dict.

    Returns a dict with keys:
        seed, verdict, rhat_max, min_bulk_ess, n_divergences, ess_per_grad,
        total_grad_evals, wall_seconds, adapted_params
    """
    from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

    from tuningfork.warmup._base import squeeze_single_chain

    master_key = jax.random.key(seed)
    init_key, warmup_key, sampling_key = jax.random.split(master_key, 3)
    init_position, logdensity_fn, _ = build_logdensity_fn(init_key, posterior)

    t0 = time.perf_counter()

    # Stage 1+2+3: LRD warmup (num_chains=1 via from_warmup_only pattern).
    warmup_keys_1 = jax.random.split(warmup_key, 2)
    # Run the full mclmc_lrd_tuning runner with num_chains=1 to get adapted params.
    warmup_result = warmup.runner(
        warmup_keys_1[0],
        init_position,
        n_warmup,
        base_method,
        logdensity_fn=logdensity_fn,
        num_chains=1,
        k_rank=k_rank,
    )
    batched_state, batched_params = warmup_result[0], warmup_result[1]
    jax.block_until_ready((batched_state, batched_params))
    _state, adapted_params = squeeze_single_chain(batched_state, batched_params)

    # Extract adapted LRD + step_size + L.
    lrd_imm = adapted_params["inverse_mass_matrix"]
    step_size = float(jnp.mean(jnp.asarray(adapted_params["step_size"])))
    L = float(jnp.mean(jnp.asarray(adapted_params["L"])))

    # Stage 4: 4-chain sampling with fixed adapted params.
    from tuningfork.base_method.mclmc import make_lrd_kernel

    sampling_keys = jax.random.split(sampling_key, num_chains)

    @jax.vmap
    def sample_one_chain(k: jax.Array) -> tuple[Any, Any]:
        import blackjax

        init_k, run_k = jax.random.split(k)
        state = blackjax.mcmc.mclmc.init(init_position, logdensity_fn, init_k)
        kernel = make_lrd_kernel(lrd_imm)

        def body_fn(state, rng_key):
            state, info = kernel(
                rng_key,
                state,
                logdensity_fn,
                inverse_mass_matrix=lrd_imm,
                L=L,
                step_size=step_size,
            )
            return state, (state.position, info)

        _, (positions, infos) = jax.lax.scan(
            body_fn, state, jax.random.split(run_k, n_samples)
        )
        return positions, infos

    positions_batched, infos_batched = sample_one_chain(sampling_keys)
    jax.block_until_ready((positions_batched, infos_batched))
    wall_seconds = time.perf_counter() - t0

    # Compute diagnostics.
    # positions_batched: pytree with leading dims (num_chains, n_samples, ...)
    rhat_tree = jax.tree.map(
        lambda x: potential_scale_reduction(x, chain_axis=0, sample_axis=1),
        positions_batched,
    )
    ess_tree = jax.tree.map(
        lambda x: effective_sample_size(x, chain_axis=0, sample_axis=1),
        positions_batched,
    )
    rhat_max = float(jnp.max(jnp.array(jax.tree.leaves(rhat_tree))))
    min_bulk_ess = float(jnp.min(jnp.array(jax.tree.leaves(ess_tree))))

    # Count divergences from MCLMCInfo.nonans (inverted: nonans=True means no NaN).
    # MCLMC doesn't have divergences in the HMC sense; use NaN-step indicator.
    # nonans=True (no NaN) → not divergent. Count False entries as "divergences".
    nan_flags = infos_batched.nonans  # shape (num_chains, n_samples), bool
    n_divergences = int(jnp.sum(~nan_flags))
    n_draws_total = num_chains * n_samples
    div_rate = n_divergences / max(n_draws_total, 1)

    # ESS/grad: MCLMC costs 2 grads/step.
    total_grad_evals = 2 * n_samples * num_chains
    ess_per_grad = min_bulk_ess / total_grad_evals

    # Gate.
    passes = (
        rhat_max < GATE_RHAT_PASS
        and min_bulk_ess >= GATE_ESS_PASS
        and div_rate <= GATE_DIV_RATE_PASS
    )
    verdict = "PASS" if passes else "FAIL"

    return {
        "seed": seed,
        "verdict": verdict,
        "rhat_max": rhat_max,
        "min_bulk_ess": min_bulk_ess,
        "n_divergences": n_divergences,
        "div_rate": div_rate,
        "ess_per_grad": ess_per_grad,
        "total_grad_evals": total_grad_evals,
        "wall_seconds": wall_seconds,
        "adapted_params": adapted_params,  # carried for bake step
    }


def _result_to_dict(r: dict[str, Any]) -> dict[str, Any]:
    """Serialise a cert-seed result to a JSON-safe dict (no adapted_params)."""
    return {
        "seed": r["seed"],
        "verdict": r["verdict"],
        "rhat_max": r["rhat_max"],
        "min_bulk_ess": r["min_bulk_ess"],
        "n_divergences": r["n_divergences"],
        "div_rate": r["div_rate"],
        "ess_per_grad": r["ess_per_grad"],
        "total_grad_evals": r["total_grad_evals"],
        "wall_seconds": r["wall_seconds"],
    }


def _get_blackjax_version() -> str:
    try:
        import blackjax

        return getattr(blackjax, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unavailable"


def _get_jax_version() -> str:
    try:
        import jax

        return jax.__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


def _now_utc_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
