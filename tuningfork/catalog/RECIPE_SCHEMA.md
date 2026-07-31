# Tuningfork recipe schema — design reference

This document specifies the **Recipe JSON schema** for the tuningfork catalog. A recipe is a pinned `(model, warmup, sampler)` configuration with provenance + gate evidence, serialised as JSON at `tuningfork/catalog/<model>/recipes/<effort>__<sampler>__<warmup>[__<modifier>...].json` (or `groundtruth.json` for `Effort.GROUNDTRUTH`).

This is the **authoritative design** for the schema. The Python `dataclass` at `tuningfork/recipes/_base.py:Recipe` is the runtime implementation; this doc is the spec the implementation must match.

Three audiences:

- **Consumer** (loading recipes, inspecting diagnostics, reproducing inference): reads sections §1–§5 for the field list.
- **Implementer** (extending the runner, emit_script, registry): reads §3–§7 for the load-bearing semantics + backward-compat rules.
- **Future schema evolutions** (slow/fast warmup separation, new step_policy kinds, ...): reads §2 + §8 for the open questions and migration patterns.

## §1 — Recipe dataclass fields (overview)

```python
@dataclass(frozen=True)
class Recipe:
    # Identity
    model_name: str                    # MODELS registry key, e.g. "eight_schools_ncp"
    base_method_name: str              # BASE_METHODS registry key, e.g. "nuts", "dynamic_hmc"
    warmup_name: str                   # WARMUPS registry key — see §2 for repeated-warmup evolution
    effort: Effort                     # GROUNDTRUTH | LOW | MEDIUM | HIGH | FAILED

    # Pinned config (what the sampler actually uses at run time)
    base_method_params: dict           # step_size, inverse_mass_matrix, ± num_integration_steps
    warmup_params: dict                # n_warmup, num_chains, target_acceptance, ...

    # Step policy (§4): callable-injection points (e.g. integration_steps_fn for dynamic_hmc)
    step_policy: dict | None = None

    # Warmup inner kernel (§3): what drives the adaptation (None → default per substitute logic)
    warmup_inner_kernel: str | None = None

    # Metadata
    headline_metric: float | None      # min_bulk_ess / total_grad_evals (filled at PASS) — see §4.5
    headline_basis: dict | None        # accounting behind headline_metric — see §4.5
    sample_quality: dict | None        # vs-reference comparisons (optional)
    calibration_budget: dict           # trials, wall_seconds_estimate, n_warmup, n_samples, num_chains
    difficulty: dict | None            # HIGH-only; BO study output
    instructions: str                  # auto-templated user-facing prose
    notes: str = ""                    # statistician-authored note (MEDIUM-workaround rationale)

    # Provenance + reproducibility
    tuning_seed: int                   # JAX random key for reproducibility
    tuningfork_version: str
    blackjax_version: str
    jax_version: str
    timestamp_utc: str

    # Gate
    gate_evidence: dict                # {"auto": {rhat_max, min_bulk_ess, n_divergences, verdict, margins}, "override": {...}}

    # Workflow / failure trail
    workflow: list[dict]               # statistician workflow steps (MEDIUM/HIGH only)
    failure_diagnosis: FailureDiagnosis | None = None    # FAILED only
    attempted_configurations: list[dict] | None = None   # FAILED only (forking-path log)

    # Sidecar (large IMM)
    inverse_mass_matrix_path: str | None = None   # relative path to .imm.npz if IMM is sidecared
```

## §2 — `warmup_name` field — repeated-warmup evolution (PROPOSED)

**Current state**: `warmup_name` is a single string (one warmup runs before sampling).

**Limitation**: Real workflows often chain warmups. E.g., `multipathfinder_window_adaptation` (PR #25) was a single fused warmup but conceptually a chain: multipathfinder for init + window_adaptation for (step_size, IMM). Likewise the user-proposed slow/fast adaptation split would isolate `window_adaptation`'s slow phase (Welford + step_size tracking) from its fast phase (final step_size tuning, IMM frozen) — two distinct stages currently fused inside `blackjax.window_adaptation`. These would be handled via extended schema work (future warmup-list / warmup_inner_kernel expansion).

### §2.1 — Single-warmup design (current)

```json
{
  "warmup_name": "window_adaptation_diag_imm",
  "warmup_params": {"n_warmup": 1000, "num_chains": 4, "target_acceptance": 0.8}
}
```

Runner: `warmup = WARMUPS[warmup_name]; state, params = warmup.runner(key, init, **warmup_params)`.

### §2.2 — Repeated-warmup proposal

Schema:

```json
{
  "warmups": [
    {"name": "pathfinder",                   "params": {"n_paths": 4, "num_chains": 4}},
    {"name": "window_adaptation_diag_imm",   "params": {"n_warmup": 1000, "num_chains": 4, "target_acceptance": 0.8}}
  ]
}
```

`warmups` is an ordered list. Runner semantics:

1. The **first** warmup is called with `init_position` from the model's prior sample.
2. Each **subsequent** warmup is called with the previous warmup's output state's `position` as its `init_position`. The previous warmup's `adapted_params` may be passed as additional kwargs (e.g., pathfinder produces a dense IMM that window_adaptation can use as `initial_inverse_mass_matrix`).
3. The **final** warmup's `(state, adapted_params, warmup_info)` is what flows into the sampler stage (via `transform_warmup_state` per §3.3).

Backward-compat: at the `warmups` schema-add point (Phase X) the legacy `warmup_name`/`warmup_params` are dropped from `Recipe.save` output (per §2.4 — Q1 resolved 2026-05-21 in favour of immediate deprecation since there's no downstream consumer of recipe JSON beyond this project). `Recipe.load` continues to accept legacy recipes by constructing `warmups = [{"name": warmup_name, "params": warmup_params}]` for one transition cycle so on-disk recipes (the LOW/MEDIUM/FAILED inventory) keep loading.

Filename for multi-step chains: rather than concatenating warmup names with separators (which scales poorly past 2–3 steps), each canonical chain composition gets a **single named entry in the mix-warmup glossary** (§2.5) of the form `mix_warmup_v{N}`. The filename stays short:

```
low__nuts__mix_warmup_v1.json         ← chain defined in glossary §2.5
low__nuts__mix_warmup_v2.json         ← different chain, also in glossary
```

This trades a tiny indirection (filename → glossary) for stable, short, sortable filenames. The recipe JSON itself records the FULL `warmups` list inline, so the recipe is self-documenting even without consulting the glossary.

### §2.3 — Extreme case: slow/fast adaptation isolated

The most ambitious application: factor `blackjax.window_adaptation` into separately-pinnable slow and fast phases. Currently:

```python
# blackjax.window_adaptation internally runs:
# 1. Slow phase: Welford windows + step_size adaptation (interleaved)
# 2. Fast phase: step_size only (IMM frozen from slow phase)
```

With repeated warmups, the recipe could express:

```json
{
  "warmups": [
    {"name": "window_adaptation_slow",
     "params": {"n_slow_windows": 5, "window_lengths": [75, 75, 150, 300, 600], "target_acceptance": 0.8}},
    {"name": "window_adaptation_fast",
     "params": {"n_fast": 50, "target_acceptance": 0.8}}
  ]
}
```

This requires blackjax to expose `window_adaptation_slow` and `window_adaptation_fast` as separate primitives — currently a single function. Out of scope for the schema doc; in scope when the upstream factor lands.

Until then, `warmups: [{"name": "window_adaptation_diag_imm", ...}]` is the canonical single-element list for the fused version.

### §2.4 — Migration path

Per Q1 ratification (§8, 2026-05-21): there's no downstream consumer of recipe JSON outside this project, so the deprecation is immediate rather than staged.

**Schema extension for warmups list** (future): Add `warmups: list[dict]` to `Recipe`. `Recipe.save` writes `warmups` only (no legacy fields emitted). `Recipe.load` accepts EITHER `warmups` OR legacy `warmup_name`/`warmup_params` — old recipes on disk continue to load via construction `warmups = [{"name": warmup_name, "params": warmup_params}]`. The legacy load path stays indefinitely so existing recipe JSONs in the catalog don't need regen.

**Optional recipe catalog refresh** (future): Re-emit existing recipes (44 LOW + 7 MEDIUM + 20 FAILED) under the new schema for filename / on-disk-JSON cleanliness. No runtime difference; pure prettification. Defer until a different schema change forces re-emission anyway.

### §2.5 — Mix-warmup glossary

Canonical named compositions for multi-step warmup chains. Each entry has a stable version number; entries are append-only (no v1 redefinition once landed).

| Slug | Composition | Use case |
|---|---|---|
| `mix_warmup_v1` | `[pathfinder, window_adaptation_diag_imm]` (placeholder spec — actual params TBD when first recipe uses it) | Heavy-tailed posteriors where pathfinder init mitigates poor random-init mode-capture, then standard window adaptation tunes (step_size, IMM). |

(Initial glossary is empty pending first real use. When schema extension for warmups list / warmup_inner_kernel (PR #54) or a future statistician workflow needs a composite warmup, the entry lands here in the same PR.)

Glossary maintenance:

- Slugs are append-only — `mix_warmup_v1` never gets redefined. New compositions get new version numbers.
- Each entry's full composition spec (warmup names + per-step params) is documented in the recipe JSON itself; the glossary entry is human-facing prose explaining *what the chain is for*, not a parsing contract.
- For one-off compositions that don't deserve a glossary entry, the recipe filename can use the explicit fallback `mix_warmup_adhoc` slug and the JSON body fully defines the chain — but this is discouraged in favour of named versions.

## §3 — `warmup_inner_kernel` field

Originally introduced in the d-hmc step_policy plan thread §12; promoted here as a first-class schema field.

### §3.1 — Why explicit

The recipe-runner has an implicit substitute-family logic in `tuningfork/warmup/_laplace_adapter.py:WARMUP_SUBSTITUTE_METHOD_NAMES`: methods whose `.init` signature requires extra kwargs that `blackjax.window_adaptation` doesn't supply (laplace_\*, dynamic_hmc, dmhmc) get NUTS substituted as the warmup kernel.

This is **implicit**: the recipe records `base_method_name = "dynamic_hmc"` and `warmup_name = "window_adaptation_diag_imm"`, but the actual warmup kernel (NUTS) is computed at run time from the substitute set. Future schema evolution requires making this explicit.

### §3.2 — Semantics + default rules

```python
warmup_inner_kernel: str | None = None
```

- `None`: defer to `resolve_warmup_algorithm(base_method)` — current behaviour. For substitute-family methods this resolves to `"nuts"`; for standard methods it resolves to `base_method_name`.
- `"nuts"`: warmup uses `blackjax.nuts` regardless of `base_method_name`. Forced for substitute family; opt-in for standard family.
- `"hmc"` / `"mhmc"` / `"mala"` / ...: warmup uses that kernel. Default for the matching `base_method_name` on the standard path.

Backward-compat: recipes without `warmup_inner_kernel` (before schema extension for warmups list / warmup_inner_kernel) load with `None` → resolved at run time. No regen needed.

### §3.3 — Transform-callable abstraction

The warmup output (`adapted_params + warmup_info`) must be transformed into the sampler's required init kwargs. The transform is a function of `(warmup_inner_kernel, base_method_name)`.

```python
def transform_warmup_state(
    warmup_inner_kernel: str,
    base_method_name: str,
    adapted_params: dict,
    warmup_info: Any,
) -> dict:
    """Returns sampler init kwargs: {step_size, IMM, [num_integration_steps], [step_policy], ...}."""
```

Lives at `tuningfork/base_method/_warmup_to_sampler_transform.py` (new module for schema extension for warmups list / warmup_inner_kernel, PR #54).

### §3.4 — Resolution table

| `warmup_inner_kernel` | `base_method` | Transform |
|---|---|---|
| `nuts` (resolved or explicit) | `nuts` | `{step_size, IMM}` (identity) |
| `nuts` | `hmc` / `mhmc` | `{step_size, IMM, num_integration_steps=median(warmup_info["num_integration_steps"])}` |
| `nuts` | `dynamic_hmc` / `dmhmc` | `{step_size, IMM, step_policy=empirical(warmup_info["num_integration_steps"])}` |
| `nuts` | `laplace_*` | `{step_size, IMM, num_integration_steps=median(NIS), log_joint_fn, theta_init}` |
| `hmc` (matches base) | `hmc` / `mhmc` | `{step_size, IMM}` (identity) |
| `mala` (matches base) | `mala` | `{step_size, IMM}` (identity) |
| `barker` (matches base) | `barker` | `{step_size, IMM}` (identity) |

`harvest_step_policy_from_nis(nis_array, max_values=24)` is the helper that powers the `step_policy=empirical(...)` cell (§4.3 below).

### §3.5 — Filename convention

When `warmup_inner_kernel` is explicit AND differs from the implicit-default, append `__inner_<kernel>`:

```
low__hmc__window_adaptation_diag_imm__inner_nuts.json     ← opt-in NUTS-warmup for HMC
low__nuts__window_adaptation_diag_imm.json                 ← default (no suffix; matches base)
```

For substitute-family methods, `inner_nuts` is the implicit default. No suffix needed unless the recipe explicitly opts to NOT use the substitute (rare; future-extend).

## §4 — `step_policy` field

Sampler-specific callable-injection points: `dynamic_hmc / dmhmc / laplace_dhmc / laplace_dmhmc` accept an `integration_steps_fn(key)` callable that picks per-step trajectory length; other samplers don't use this field.

### §4.1 — Schema spec (Path A + Path B)

```python
step_policy: dict | None = None
```

- `None`: use the library default for the sampler. For dynamic_hmc, this is `lambda key: jax.random.randint(key, (), 1, 10)`.
- `dict`: a spec recognised by `build_step_policy(spec)`. Two paths:

**Path A — Parametric kinds**: flat dict with `kind` + per-kind fields + standard `low`/`high` bounds.

```json
{"kind": "uniform_int",      "low": 1, "high": 10}
{"kind": "uniform_int",      "low": 50, "high": 200}
{"kind": "log_uniform_int",  "low": 1, "high": 1024}
{"kind": "poisson",          "lam": 20, "low": 1, "high": null}
{"kind": "pow2_choice",      "options": [2, 4, 8, 16, 32, 64]}
```

Bounds semantics per kind:

| Kind | `low`/`high` interpretation |
|---|---|
| `uniform_int` | `low` inclusive, `high` exclusive — direct `jax.random.randint(key, (), low, high)` |
| `log_uniform_int` | Sample `u ~ Uniform(log(low), log(high))`, return `round(exp(u))` clipped to `[low, high]` |
| `poisson` | `lam` = mean; `low` floor via `jnp.maximum`; `high` = `null` means no ceiling; numeric `high` triggers `jax.lax.while_loop` rejection |
| `pow2_choice` | `options` = explicit list (replaces `low`/`high`) |

**Path B — Empirical histogram** (compressed):

```json
{"kind": "empirical",
 "values": [3, 7, 15, 31, 63, 127],
 "weights": [0.05, 0.10, 0.20, 0.30, 0.25, 0.10]}
```

`values` are sorted distinct integer L values (or bin centres after histogram-binning); `weights` are normalised probabilities. Inverse-CDF sampling via `jnp.searchsorted`.

**Compression rule (Q5 resolved 2026-05-21)**: the empirical spec is ALWAYS a compressed representation of the true NIS histogram, **capped at 24 distinct entries** ("24-bit cap" — see note below on interpretation). For low-variance NIS distributions (radon NIS_med=15, irt_2pl NIS_med=31), distinct integer L values are typically < 24 and stored directly. For high-variance distributions (horseshoe NIS up to 1023, gp_regression up to ~500), the harvest function histogram-bins into ≤24 bin-centres covering the observed range. Compression is lossy but bounded; the recipe stays JSON-readable; storage ≈ 24 × (small-int + small-float) ≈ 600 bytes per spec.

Interpretation note on "24 bits": the user's directive (2026-05-21) was *"cap it at 24 bits"*. The natural reading for a histogram-compression context is 24 distinct bins (= 24-entry-cap), which is what the implementation uses. If a stricter binary-encoding ("each entry packed in 24 bits = 12 bits L + 12 bits weight") was intended, the change is mechanical — the schema doc would specify binary packing of `values`/`weights` and the storage drops to ~72 bytes per spec at the cost of JSON readability. Flag this for clarification when implementation lands.

### §4.2 — Path A registry

`build_step_policy(spec)` in `tuningfork/base_method/_step_policy_registry.py` reconstructs the runtime callable from the JSON spec. Schema wiring (PR #39) completes the end-to-end path.

### §4.3 — NUTS-harvested step_policy (two sources)

The `kind: "empirical"` policy can be harvested from two sources:

**Path A — Post-warmup chain_stats** (`harvest_step_policy_from_chain_stats(chain_stats_path)`):

Loads `<model>/_cache/<recipe_stem>.chain_stats.npz`, reads `num_integration_steps`, builds the empirical histogram. Requires the matching nuts recipe to have been run first (populates the cache via `tuningfork.catalog._rerun_inference.cached_idata_for_recipe`).

Statistically cleaner: post-warmup is steady-state.

**Path B — Live warmup_info** (`harvest_step_policy_from_nis(nis_array)`):

Takes a raw integer array (from the **current run's** NUTS warmup_info), builds the empirical histogram. No dependency on a separate cache. Cheaper.

The NUTS-harvested step_policy work used Path B for `ill_cond_50 × W1 × dynamic_hmc` (and the dmhmc sibling): the same run's NUTS warmup produces the L distribution, which becomes the step_policy for the sampling stage. The recipe pins the harvested spec.

### §4.4 — Filename convention

When `step_policy` is non-None AND differs from V0 default, append `__policy_<slug>`:

```
low__dynamic_hmc__window_adaptation_diag_imm.json                              ← V0 default
medium__dynamic_hmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json   ← NUTS-harvested step_policy
medium__dynamic_hmc__window_adaptation_diag_imm__policy_v2-long.json              ← V2 parametric
```

The `<slug>` matches a variant in the policy catalog. Slug-to-anchor mapping is 1:1 with the catalog.

## §4.5 — `headline_metric` and `headline_basis`

`headline_metric` is the number an algorithm developer chases: worst-mixing bulk-ESS divided by the run's gradient budget. `headline_basis` records the accounting that produced it, so a headline is interpretable and auditable without re-running the cell.

```json
{
  "headline_metric": 0.21033146977424622,
  "headline_basis": {
    "total_grad_evals": 8000,
    "min_bulk_ess": 1682.65175819397,
    "ess_estimator": "ess_bulk",
    "min_bulk_ess_classic_legacy": 1553.2,
    "estimator_ratio": 1.0833,
    "grad_count_convention": "2",
    "is_lower_bound": false
  }
}
```

### §4.5.1 — Which ESS estimator

`ess_estimator` names the `blackjax.diagnostics` function whose value is in `min_bulk_ess`. The catalog convention is **`ess_bulk`** — the rank-normalised split-chain estimator of Vehtari et al. (2021), which is what Stan, ArviZ `ess(method="bulk")` and NumPyro report. A tuningfork headline is therefore directly comparable to a published number.

There is a sharper reason than external comparability, and it is what actually forced the switch. The auto-gate's PASS/FAIL verdict (`gate_evidence.auto.min_bulk_ess`, computed by `calibration/_gate/mixing.py` from `blackjax.diagnostics.ess_bulk`) has always used the rank-normalised estimator — every certification decision in the catalog was already made on `ess_bulk`. Before this migration, `headline_metric` was computed from `effective_sample_size` instead (see `metrics/headline.py`), so a PASS recipe reported two different ESS numbers for the same draws: the rank-normalised one that decided whether the cell passed, and the classic one printed as its headline. The headline was the one field out of step with the gate that produced it, not a free-standing choice of convention — `ess_bulk` was already load-bearing everywhere else. `stamp_headline_from_chain_stats` makes the fix explicit: it reads the headline straight off `gate_evidence.auto.min_bulk_ess` rather than recomputing anything, because after the switch the two are meant to be the same number.

The field exists because a `headline_basis` that merely reproduces `headline_metric` is *self-consistent under any estimator*. Consistency checks cannot tell you which one ran; only a recorded provenance stamp can. Every code path that fills a headline writes this field:

| Path | Source of `min_bulk_ess` |
|---|---|
| generated certification (gradient) | `metrics.headline.build_headline_basis` over the generated run's validated draws |
| generated certification (gradient-free) | same evaluator and draws; denominator is the total draw count |
| `recipes/emit_mclmc_lrd.py` | `metrics.headline.min_bulk_ess` over the cert-seed draws |
| `_recipe_runner.stamp_headline_from_chain_stats` | the gate's `ess_bulk` value, recovered without re-sampling |

`min_bulk_ess_classic_legacy` carries `blackjax.diagnostics.effective_sample_size` — no chain splitting, no rank normalisation — computed on the **same draws**, and `estimator_ratio` is `min_bulk_ess / min_bulk_ess_classic_legacy`. They exist so a change in a committed headline can be attributed: a re-emit produces fresh draws, so diffing new against committed confounds the estimator with run-to-run noise, while the ratio isolates the estimator on one fixed sample. Neither field feeds any gate or ranking. Both are `null` where no draws were available (the `stamp_headline_from_chain_stats` path).

### §4.5.2 — Models excluded from the estimator convention

One model is **deliberately** left on the older estimator, so the catalog is knowingly mixed:

| Model | Why |
|---|---|
| `gp_regression` | Compute cost. Dense 200×200 RBF kernel → ~50× the per-step cost of a peer model (~63 h reference certification wall), and its only headline-carrying recipe is HIGH effort, so a faithful re-measurement means re-running the hyperparameter search, not one sampler run. |

**A `gp_regression` headline is not comparable to any other model's.** That is the exact failure mode the headline metric exists to avoid, so the exclusion is recorded in three places rather than one: the machine-readable list at `catalog/_estimator_provenance.py:HEADLINE_ESTIMATOR_EXCLUDED_MODELS`, a `## Headline numbers are not comparable` section in `catalog/gp_regression/lessons.md`, and a `headline_ESS_estimator` row that `summarize_recipe` prints with an inline `NOT COMPARABLE` caveat.

It is not recorded inside the recipe JSON, because recipe artifacts are written only by the emit harness — hand-editing one to add a provenance marker would defeat the provenance the marker is meant to carry.

`tests/recipes/test_emit.py::test_estimator_exclusions_are_live_and_still_excluded` keeps the list honest in both directions: a named model that the catalog no longer has fails, and so does a listed model that turns out to have been re-measured (its recipes would then carry an `ess_estimator` stamp, contradicting the list).

### §4.5.3 — The exact-reproduction invariant

`min_bulk_ess` is back-derived as `headline_metric × denominator`, so

```
headline_metric == headline_basis.min_bulk_ess / headline_basis.total_grad_evals
```

holds to floating-point exactness (tolerance 1e-9, not a percentage band) for every gradient-path recipe. `tests/recipes/test_emit.py::test_catalog_headline_basis_reproduces_headline_metric` enforces it catalog-wide.

For a gradient-free sampler `total_grad_evals` is `0` and the denominator is the total draw count instead — the metric is per-draw efficiency, and `grad_count_convention` records that. The invariant test skips those rows.

`is_lower_bound` is true for the Laplace family, where the gradient count under-counts the inner optimisation.

## §5 — Filename composition

Recipe filename structure (after all modifiers compose):

```
<effort>__<sampler>__<warmup>[__inner_<kernel>][__policy_<slug>][__<other-modifier>...].json
```

Examples:

- `low__nuts__window_adaptation_diag_imm.json` (baseline)
- `medium__hmc__window_adaptation_dense_imm.json` (MEDIUM — no modifiers)
- `medium__dynamic_hmc__window_adaptation_diag_imm__policy_v7-empirical-oracle.json` (NUTS-harvested empirical-oracle step_policy)
- `low__hmc__window_adaptation_diag_imm__inner_nuts.json` (hypothetical hmc-via-NUTS-warmup variant)
- `low__nuts__pathfinder+window_adaptation_diag_imm.json` (hypothetical chained warmup — §2.2)

**Ordering of modifier slots** (left to right): `inner_*`, `policy_*`, then any future modifiers (e.g., `init_*` for over-dispersed init strategy). Stable order so filenames sort meaningfully.

## §6 — Backward-compat strategy

Schema additions (`warmup_inner_kernel`, `step_policy`, `warmups`) are additive. Existing recipes without these fields load cleanly:

- `Recipe.load` defaults missing fields to `None`.
- The runner resolves `warmup_inner_kernel=None` via the current substitute-family logic.
- The runner treats `step_policy=None` as V0 (library default).
- The runner treats missing `warmups` by constructing `warmups = [{"name": warmup_name, "params": warmup_params}]`.

No regen needed on schema-add. Regen IS needed when the **runtime behaviour** changes (e.g., PR #42's NUTS-substitute change → 23 recipes regened in PR #43 for refreshed `gate_evidence`).

## §7 — Implementation locations

| Concern | File |
|---|---|
| Recipe dataclass | `tuningfork/recipes/_base.py` |
| Runner (emit recipes, apply transforms) | `tuningfork/recipes/_recipe_runner.py` |
| emit_script (reproduction-script codegen) | `tuningfork/recipes/_emit_script.py` |
| Templates (warmup / sampler / inference loop) | `tuningfork/recipes/_templates/{warmups,samplers,...}/*.py.tmpl` |
| step_policy registry | `tuningfork/base_method/_step_policy_registry.py` |
| Transform callable (§3.3) | `tuningfork/base_method/_warmup_to_sampler_transform.py` (new for schema extension for warmups list / warmup_inner_kernel) |
| Substitute-family resolution | `tuningfork/warmup/_laplace_adapter.py:WARMUP_SUBSTITUTE_METHOD_NAMES + resolve_warmup_algorithm` |
| Catalog inspection (consumer) | `tuningfork/catalog/inspect.py:load_recipe + summarize_recipe` |

## §8 — Open questions (ALL RESOLVED 2026-05-21)

| # | Question | Resolution |
|---|---|---|
| 1 | When `warmups` (list) lands, deprecate `warmup_name`/`warmup_params` immediately, or keep both? | **RESOLVED — (a) deprecate immediately**. No downstream consumer of recipe JSON outside this project; clean break. `Recipe.save` emits only `warmups`. `Recipe.load` accepts legacy fields for backward-load of on-disk recipes (no mass regen needed). See §2.4. |
| 2 | Filename length for multi-step warmup chains: separator-concatenation vs. some other notation? | **RESOLVED — glossary + `mix_warmup_v{N}` slug**. Each canonical chain composition gets a named glossary entry (§2.5); filename uses the short slug. Avoids separator-concatenation entirely; scales gracefully past 2–3 steps. |
| 3 | Does `warmup_inner_kernel` need a corresponding `warmup_inner_kwargs` field? | **RESOLVED — (a) defer**. Implicit kwargs via `default_value_for_space` works; revisit when a concrete cell needs explicit pinning. |
| 4 | Multi-chain `warmup_info` shape for `transform_warmup_state`: ravel across chains, or per-chain transforms? | **RESOLVED — (a) ravel**. Pool across chains for one canonical L distribution / median; ensures all chains run the same sampling protocol (necessary for cross-chain rhat). |
| 5 | When `kind="empirical"` storage exceeds 8 KB, sidecar to `.npz`? | **RESOLVED — compress always, cap at 24-bit equivalent**. Empirical spec is always a compressed histogram with ≤24 distinct entries (§4.1 Path B). Storage stays ≈ 600 bytes per spec; no sidecar needed. Interpretation note on "24 bits" filed in §4.1 (current implementation: 24 bin-centres; tighter binary packing possible as a follow-up if user prefers). |

## §9 — Versioning

This schema is unversioned at the file level (no `schema_version` field on Recipe). Versioning is by code: the Python `Recipe` class defines what's required; older Python with newer recipes may misinterpret newly-added fields silently (default to None). For breaking changes, bump `tuningfork_version` in the recipe; consumers check.

If the schema diverges in incompatible ways (renames, removed fields), introduce a `schema_version: int` field at that point and have `Recipe.load` dispatch on it.

## §10 — Related documents

- **step_policy variant catalog**: step_policy variant catalog, per-cell prediction matrix, NUTS-harvested step_policy work execution log
- **Effort taxonomy** decision: effort-taxonomy-canonical-c (2026-05-10)
- **Catalog README**: [`catalog/README.md`](README.md) — user-facing consumption guide
- **Inspection API**: [`catalog/notebooks/inspect_README.md`](notebooks/inspect_README.md)
- **Recipe runner source**: [`recipes/_recipe_runner.py`](../recipes/_recipe_runner.py) — runtime implementation

## §11 — Change log

| Date | Change |
|---|---|
| 2026-05-21 | Initial schema doc. Extracted from the step_policy design thread §5, §10, §12. Added §2 repeated-warmup proposal (per user direction). |
| 2026-07-29 | Headline adopts the rank-normalised split-chain bulk-ESS estimator (`blackjax.diagnostics.ess_bulk`), keeping the field name `min_bulk_ess`. Added §4.5 documenting `headline_basis`, including the new `ess_estimator` provenance stamp and the `min_bulk_ess_classic_legacy` / `estimator_ratio` attribution pair. `gp_regression` excluded on compute cost (§4.5.2) — the catalog is knowingly mixed. |
| 2026-07-30 | §4.5.1: added the gate/headline estimator-mismatch paragraph — `gate_evidence.auto.min_bulk_ess` was always `ess_bulk`, but `headline_metric` was `effective_sample_size` before the 2026-07-29 switch, so a PASS recipe reported two different ESS numbers for the same draws. Post-merge #254 follow-up: comparability-to-literature was documented, but the internal mismatch that actually forced the switch was not. |
| 2026-05-21 (same day, later) | Locked all 5 §8 open questions per user direction: Q1 immediate-deprecate; Q2 `mix_warmup_v{N}` glossary in lieu of separator-concatenation; Q3 defer `warmup_inner_kwargs`; Q4 ravel across chains; Q5 always-compress empirical spec with ≤24-entry cap. §2.5 mix-warmup glossary section added (initially empty pending first real use). `max_values` default in `harvest_step_policy_from_*` updated 512 → 24. |
