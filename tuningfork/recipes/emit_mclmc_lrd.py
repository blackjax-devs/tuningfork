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
"""Single implementation module for MCLMC-LRD recipe emission.

Both the stub (``calibrate=False``) and cert-sweep (``calibrate=True``)
paths live here.  The public entry point is
``tuningfork.recipes._generate_starter.emit_mclmc_lrd_recipes`` — call that
instead of the private helpers below.

``_emit_mclmc_lrd_recipes_impl``
    Dispatcher: stub path or cert-sweep path depending on ``calibrate``.

``_emit_lrd_cert_sweep``
    Cert-sweep implementation (delegated to by the above when
    ``calibrate=True``).

``_run_cert_seed``
    Run one cert seed: warmup + 4-chain sampling → diagnostics dict.
    Used for phase (c) runs — call signature is
    stable; do not rename without coordinating.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

from tuningfork._machine_info import get_machine_info as _get_machine_info
from tuningfork._version import __version__ as _tuningfork_version
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

# Default catalog root: tuningfork/tuningfork/catalog/ relative to this file.
# parents[0] = recipes/, parents[1] = tuningfork/ (package), then /catalog.
_DEFAULT_CATALOG_ROOT = Path(__file__).resolve().parents[1] / "catalog"

# Methods eligible for MCLMC-LRD recipes — mirrors _generate_starter.MCLMC_LRD_METHOD_NAMES.
_MCLMC_LRD_METHOD_NAMES: list[str] = ["mclmc"]


def _emit_mclmc_lrd_recipes_impl(
    model_names: list[str],
    *,
    calibrate: bool = False,
    seed: int = 0,
    cert_seeds: tuple[int, ...] = (11111, 22222, 33333),
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
    k_rank: int = 40,
    pilot_n_warmup: int = 1000,
    pilot_n_samples: int = 1000,
    sampler: str | None = None,
    catalog_root: Path | None = None,
    variant_label: str = "mclmc_lrd",
    effort: Effort = Effort.LOW,
    tuningfork_version: str = _tuningfork_version,
) -> list[Path]:
    """Unified MCLMC-LRD recipe emitter — both stub and cert-sweep paths.

    Called by ``tuningfork.recipes._generate_starter.emit_mclmc_lrd_recipes``.
    Do not call directly.

    Parameters
    ----------
    model_names
        Models to emit recipes for.
    calibrate
        ``False`` (default): emit a MEDIUM stub recipe per model via a single
        warmup run (deterministic key derived from ``seed``).
        ``True``: run the full cert sweep (delegates to ``_emit_lrd_cert_sweep``).
    seed
        Base random seed for the ``calibrate=False`` stub path.
        ``jax.random.fold_in`` derives per-recipe keys deterministically from
        ``(model_name, method_name, "mclmc_lrd")``.  Ignored when
        ``calibrate=True`` (use ``cert_seeds`` instead).
    cert_seeds
        Random seeds for the certification sweep.  Used only when
        ``calibrate=True``.
    n_warmup
        LRD adaptation steps.
    n_samples
        Post-warmup samples per chain for the gate check (``calibrate=True`` only).
    num_chains
        Chains for the gate check (``calibrate=True`` only).
    k_rank
        LRD approximation rank.
    sampler
        If set, restrict to this base-method name.
    catalog_root
        Root directory for the catalog.  Defaults to ``_DEFAULT_CATALOG_ROOT``.
    variant_label
        Filename-stem label for emitted recipes.
    effort
        Effort tier for passing recipes (``calibrate=True`` only).
    tuningfork_version
        Version string for recipe provenance.

    Returns
    -------
    list[Path]
        Paths of written recipe JSON files.
    """
    root = Path(catalog_root) if catalog_root is not None else _DEFAULT_CATALOG_ROOT

    if calibrate:
        # Sampler guard: cert sweep is always mclmc.
        if sampler is not None and sampler not in _MCLMC_LRD_METHOD_NAMES:
            return []
        return _emit_lrd_cert_sweep(
            model_names,
            cert_seeds=cert_seeds,
            n_warmup=n_warmup,
            n_samples=n_samples,
            num_chains=num_chains,
            k_rank=k_rank,
            pilot_n_warmup=pilot_n_warmup,
            pilot_n_samples=pilot_n_samples,
            catalog_root=root,
            variant_label=variant_label,
            effort=effort,
            tuningfork_version=tuningfork_version,
        )

    # calibrate=False stub: one warmup run per (model, method) with a
    # deterministic per-recipe key derived from (model_name, method_name, "mclmc_lrd").
    mclmc_lrd_tuning = WARMUPS["mclmc_lrd_tuning"]
    generated: list[Path] = []

    for model_name in model_names:
        posterior = MODELS[model_name]
        for method_name in _MCLMC_LRD_METHOD_NAMES:
            if sampler is not None and method_name != sampler:
                continue
            base_method = BASE_METHODS[method_name]

            if not mclmc_lrd_tuning.is_compatible(method_name):
                continue

            hash_val = hash((model_name, method_name, "mclmc_lrd")) & 0xFFFFFFFF
            key = jax.random.fold_in(jax.random.key(seed), hash_val)

            recipe = Recipe.from_warmup_only(
                posterior,
                base_method,
                mclmc_lrd_tuning,
                n_warmup=n_warmup,
                rng_key=key,
                tuningfork_version=tuningfork_version,
            )
            path = recipe.save(root)
            generated.append(path)

    return generated


def _emit_lrd_cert_sweep(
    model_names: list[str],
    *,
    cert_seeds: tuple[int, ...] = (11111, 22222, 33333),
    n_warmup: int = 1000,
    n_samples: int = 1000,
    num_chains: int = 4,
    k_rank: int = 40,
    pilot_n_warmup: int = 1000,
    pilot_n_samples: int = 1000,
    catalog_root: Path | None = None,
    variant_label: str = "mclmc_lrd",
    effort: Effort = Effort.LOW,
    tuningfork_version: str = _tuningfork_version,
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
        LRD adaptation steps (``lrd_num_steps`` in the upstream Scheme A
        warmup).  Default 1000.
    n_samples
        Post-warmup samples per chain for the gate check.
    num_chains
        Number of sampling chains for the gate check.  The warmup always
        uses ``num_chains=1`` (via ``from_warmup_only``).
    k_rank
        LRD approximation rank.
    pilot_n_warmup
        Diagonal MCLMC pilot warmup steps (``pilot_num_warmup`` in upstream).
        Default 1000.  Certified configs: german_credit 5000, ill_cond_50 1000.
    pilot_n_samples
        Pilot samples for SVD geometry estimation (``pilot_num_samples`` in
        upstream).  Default 1000.  Certified configs: german_credit 5000,
        ill_cond_50 10000.
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
                pilot_n_warmup=pilot_n_warmup,
                pilot_n_samples=pilot_n_samples,
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

            # R1 fix: construct Recipe directly from best["adapted_params"] so that
            # the committed golden's step_size/L/IMM == the measured/certified values.
            # The previous path called from_warmup_only(rng_key=jax.random.key(seed))
            # which derives its warmup key as split(rng, 2)[1], while _run_cert_seed
            # uses split(split(master, 3)[1], 2)[0] — a different realisation.
            # Using adapted_params directly avoids the redundant warmup re-run and
            # guarantees param identity between gate_evidence and the recipe artifact.
            from tuningfork.calibration.tune import default_params_for
            from tuningfork.recipes._base import _to_jsonable
            from tuningfork.recipes._instructions import render_instructions

            adapted_params = best["adapted_params"]
            # Strip underscore-prefixed metadata (tuning-internal keys, e.g. _total_tuning_steps).
            clean_adapted = {
                k: v for k, v in adapted_params.items() if not k.startswith("_")
            }
            # De-broadcast LRD IMM if needed: the certified runner broadcasts the
            # single shared LRD to a (num_chains, d[, k]) namedtuple so jax.vmap
            # can slice per chain.  Goldens/sidecars use the unbatched (d,)/(d,k)/(k,)
            # format.  Guard on sigma.ndim > 1 so that callers who already squeezed
            # (e.g. _run_cert_seed uses num_chains=1 + squeeze_single_chain) are not
            # double-squeezed — sigma[0] on a (d,) array gives a scalar, not (d,).
            if "inverse_mass_matrix" in clean_adapted:
                imm = clean_adapted["inverse_mass_matrix"]
                if isinstance(imm, LowRankInverseMassMatrix) and imm.sigma.ndim > 1:
                    clean_adapted = dict(clean_adapted)
                    clean_adapted["inverse_mass_matrix"] = LowRankInverseMassMatrix(
                        sigma=imm.sigma[0],
                        U=imm.U[0],
                        lam=imm.lam[0],
                    )
            # Merge with base-method defaults (adapted values win).
            base_params = {**default_params_for(base_method), **clean_adapted}
            # Coerce JAX arrays → Python scalars/lists; LRD namedtuple passes through.
            base_params = _to_jsonable(base_params)
            # Old-golden contract: base_method_params carries k_rank so consumers
            # know which rank was used without parsing calibration_budget.
            base_params["k_rank"] = k_rank

            calibration_budget_pass: dict[str, Any] = {
                "trials": 0,
                "wall_seconds_estimate": best.get("wall_seconds", 0.0),
                "n_warmup": n_warmup,
                "seed_evidence": [_result_to_dict(r) for r in seed_results],
                "baked_from": {
                    "warmup_name": warmup.name,
                    "n_warmup": n_warmup,
                    "k_rank": k_rank,  # provenance: LRD rank used during cert run
                    "pilot_n_warmup": pilot_n_warmup,
                    "pilot_n_samples": pilot_n_samples,
                    "tuning_seed": best["seed"],
                },
                "machine_info": _get_machine_info(),
            }

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

            recipe_kwargs_pass: dict[str, Any] = dict(
                model_name=posterior.name,
                base_method_name=base_method.name,
                warmup_name="",  # baked: warmup fields blanked at runtime
                effort=effort,
                base_method_params=base_params,
                warmup_params={"n_warmup": n_warmup},
                # Populate warmups so Recipe.load derives warmup_name="mclmc_lrd_tuning"
                # from warmups[0]["name"] (line 796-799 of _base.py).  Without this,
                # warmups=[] → load takes the legacy path → warmup_name="" →
                # emit_script raises "No emit function for warmup ''".
                warmups=[
                    {
                        "name": warmup.name,
                        "params": {
                            "n_warmup": n_warmup,
                            "num_chains": num_chains,
                            "k_rank": k_rank,
                            "pilot_n_warmup": pilot_n_warmup,
                            "pilot_n_samples": pilot_n_samples,
                        },
                    }
                ],
                headline_metric=best["ess_per_grad"],
                headline_basis={
                    "total_grad_evals": best["total_grad_evals"],
                    # Use the headline ESS (effective_sample_size, non-rank-normalised)
                    # so that headline_metric == min_bulk_ess / total_grad_evals is
                    # self-consistent.  The gate uses min_bulk_ess (ess_bulk,
                    # rank-normalised) which lives in gate_evidence.auto.min_bulk_ess.
                    "min_bulk_ess": best["min_bulk_ess_headline"],
                    "grad_count_convention": "2",
                    "is_lower_bound": False,
                },
                sample_quality=None,
                calibration_budget=calibration_budget_pass,
                difficulty=None,
                instructions="",  # rendered below after provisional construction
                notes=(
                    f"LRD-MCLMC calibrated: {pass_count}/{total} seeds PASS. "
                    f"Best seed {best['seed']}: ESS/grad={best['ess_per_grad']:.4f}, "
                    f"R-hat={best['rhat_max']:.4f}, minESS={best['min_bulk_ess']:.0f}."
                ),
                variant_label=variant_label,
                gate_evidence=gate_evidence,
                tuning_seed=best["seed"],
                tuningfork_version=tuningfork_version,
                blackjax_version=_get_blackjax_version(),
                jax_version=_get_jax_version(),
                timestamp_utc=_now_utc_iso(),
            )
            provisional = Recipe(**recipe_kwargs_pass)
            recipe_kwargs_pass["instructions"] = render_instructions(provisional)
            recipe = Recipe(**recipe_kwargs_pass)
            path = recipe.save(root, imm_sidecar="auto")
        else:
            # < 2/3 PASS: emit a FAILED recipe recording the attempted configs.
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
    pilot_n_warmup: int,
    pilot_n_samples: int,
    tuningfork_version: str,
    variant_label: str,
) -> dict[str, Any]:
    """Run one cert-seed: warmup + 4-chain sampling → diagnostics dict.

    Returns a dict with keys:
        seed, verdict, rhat_max, min_bulk_ess, n_divergences, ess_per_grad,
        total_grad_evals, wall_seconds, adapted_params
    """
    import numpy as np
    from blackjax.diagnostics import effective_sample_size

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
        pilot_n_warmup=pilot_n_warmup,
        pilot_n_samples=pilot_n_samples,
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
    #
    # GATE diagnostics: blackjax.diagnostics.ess_bulk (rank-normalised split-chain
    # ESS, Vehtari 2021) is bit-identical to az.ess(bulk) at rel diff ≤ 1e-6
    # since blackjax 1.6.1.  The historical mismatch (3–10× lower values for slow-
    # mixing models) was caused by the old effective_sample_size using a different
    # autocorrelation formula — ess_bulk (1.6.1+) fixes this.
    #
    # Headline ESS/grad: computed via blackjax.diagnostics.effective_sample_size
    # (per headline.py contract — headline basis is blackjax by design; the two
    # estimators differ by rank-normalisation, intentionally).
    from blackjax.diagnostics import ess_bulk as _bj_ess_bulk_gate
    from blackjax.diagnostics import rhat as _bj_rhat_gate

    rhat_values: list[float] = []
    ess_values_gate: list[float] = []
    for leaf in jax.tree.leaves(positions_batched):
        arr_np = np.asarray(leaf)  # shape (num_chains, n_samples, *event_shape)
        rhat_arr = _bj_rhat_gate(arr_np, chain_axis=0, sample_axis=1)
        ess_arr = _bj_ess_bulk_gate(arr_np, chain_axis=0, sample_axis=1)
        rhat_values.append(float(np.max(np.asarray(rhat_arr))))
        ess_values_gate.append(float(np.min(np.asarray(ess_arr))))

    rhat_max = float(max(rhat_values))
    min_bulk_ess = float(min(ess_values_gate))

    # Headline ESS via blackjax estimator (blackjax basis — per headline.py contract).
    ess_tree = jax.tree.map(
        lambda x: effective_sample_size(x, chain_axis=0, sample_axis=1),
        positions_batched,
    )
    min_bulk_ess_headline = float(
        jnp.min(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(ess_tree)]))
    )

    # Count divergences from MCLMCInfo.nonans (inverted: nonans=True means no NaN).
    # MCLMC doesn't have divergences in the HMC sense; use NaN-step indicator.
    # nonans=True (no NaN) → not divergent. Count False entries as "divergences".
    nan_flags = infos_batched.nonans  # shape (num_chains, n_samples), bool
    n_divergences = int(jnp.sum(~nan_flags))
    n_draws_total = num_chains * n_samples
    div_rate = n_divergences / max(n_draws_total, 1)

    # ESS/grad: MCLMC costs 2 grads/step.  Use headline (blackjax) ESS per headline.py.
    total_grad_evals = 2 * n_samples * num_chains
    ess_per_grad = min_bulk_ess_headline / total_grad_evals

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
        # min_bulk_ess_headline is the headline ESS (from effective_sample_size,
        # non-rank-normalised, per headline.py contract).  Distinct from min_bulk_ess
        # which uses ess_bulk (rank-normalised, gate basis).  headline_basis must store
        # the *headline* ESS so that headline_metric == min_bulk_ess_headline / total_grad_evals
        # is self-consistent with the basis.
        "min_bulk_ess_headline": min_bulk_ess_headline,
        "n_divergences": n_divergences,
        "div_rate": div_rate,
        "ess_per_grad": ess_per_grad,
        "total_grad_evals": total_grad_evals,
        "wall_seconds": wall_seconds,
        "step_size": step_size,
        "L": L,
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
        "step_size": float(r["step_size"]) if "step_size" in r else None,
        "L": float(r["L"]) if "L" in r else None,
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
