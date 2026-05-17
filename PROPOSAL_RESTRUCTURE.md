# Restructure proposal: tuningfork as a recipe library

**Status**: PROPOSAL, pending review
**Date**: 2026-05-17
**Author**: TL (after user retrospective)
**Supersedes**: implicit structure documented in `CLAUDE.md` § Architecture

---

## TL;DR

The library has three concerns mixed together inside `tuningfork/recipes/`:
**(a)** recipe schema + generators (Python code), **(b)** recipe artifacts
(JSON pins + IMM sidecars), and **(c)** orchestration helpers (`_STARTER_ROOT`).
At today's ~9 groundtruth recipes this is mildly awkward; at Recipe-Phase-1+'s
target of ~700 artifacts across 14 model subdirs, it becomes structurally
incoherent.

Concurrently, three first-class concepts are missing or implicit:
**(1) failed recipes** — combinations that don't work have no recording
mechanism, so we either lose the diagnostic exit memo or it lands in
worklog case-studies (which are agent-team artifacts, not library
artifacts); **(2) per-model lessons** — sampling quirks and forking-path
history have no home next to the model code/artifacts; **(3) benchmark
suite** — the original library goal, mostly superseded by recipes, but
the wall-time-filtered "smoke test the recipes that finish in <2 min"
pattern is unimplemented.

This proposal recommends a **two-layer split** of the package:

- **Generator layer** (`tuningfork.{model, base_method, warmup, smc,
  recipes, calibration, metrics, runner}`) — used to *produce*
  recipes. Contributors of new sampler wrappers or models work here.
- **Library layer** (`tuningfork.catalog`) — used to *consume*
  recipes. Contains: per-model artifact directories
  (`catalog/<model>/{lessons.md, groundtruth.json, reference/,
  recipes/, _cache/}`), user-facing API helpers (`load_recipe`,
  `load_idata`, `summarize_recipe`, `emit_script`), ArviZ diagnostics,
  and template notebooks. The library subpackage is the only surface
  a "regular user" needs.

The proposal also adds three first-class features that don't exist
today:

1. **`Effort.FAILED` recipes** with `FailureDiagnosis` enum — recording
   what doesn't work and why is as valuable as recording what does.
2. **Per-model `lessons.md`** — user-facing distilled sampling
   knowledge co-located with model artifacts (worklog substrate stays
   as the agent process record; different audience).
3. **`emit_script(recipe)` code-gen** — given any recipe, emit a
   standalone Python script with zero `import tuningfork`. Recipes
   become portable units: take the script + data sidecar to a fresh
   project, install `jax/blackjax/numpyro`, and reproduce. This pins
   wiring code to ~30 lines per sampler (a template), making the
   wiring auditable and surfacing any heavy-wrapper design smells as
   BlackJAX upstream issues.

---

## 1. Retrospective: where are we

After PR #6 (cleanup-and-simplify, 2026-05-17) the package layout is:

```
tuningfork/tuningfork/
├── model/<model>.py              # 14 model defs + MODELS, MODELS_BY_FAMILY
├── base_method/                  # 24 sampler wrappers + BASE_METHODS
├── warmup/                       # 10 warmup wrappers + WARMUPS
├── smc/                          # 6 SMC wrappers + SMC_METHODS
├── recipes/                      # ← THE PAIN POINT
│   ├── _base.py                  # Recipe dataclass (CODE)
│   ├── _instructions.py          # instruction templates (CODE)
│   ├── _generate_groundtruth.py  # groundtruth orchestrator (CODE)
│   ├── _generate_starter.py      # starter orchestrator (CODE)
│   ├── __init__.py
│   └── starter/<model>/*.json    # PINS (ARTIFACTS, ~700 eventual)
├── calibration/                  # cert + tune + auto-gate
├── metrics/                      # headline + grad_counter + reference_compare
├── runner/smc.py                 # SMC init + run helpers
├── reference/                    # ← ALSO MIXED
│   ├── _io.py                    # cache reader/writer (CODE)
│   ├── _posteriordb_xcheck.py    # xcheck logic (CODE)
│   └── <model>/                  # cert outputs (ARTIFACTS)
│       ├── metadata.json
│       ├── summary.json
│       ├── adaptation.json
│       ├── xcheck.json
│       ├── draws.npz             # gitignored
│       ├── chain_stats.npz       # gitignored
│       └── warmup_checkpoint/    # gitignored
├── data/                         # raw input datasets (CSV/NPZ; ARTIFACTS but stable)
├── inspect.py, render.py         # user-facing helpers
├── diagnostics.py                # ArviZ rendering
└── cli.py
```

Working well: model/, base_method/, warmup/, smc/, calibration/, metrics/,
inspect+render+diagnostics, cli. These are pure code, organized by concern,
with thin wiring (per Principle A below).

Mixed concerns: `recipes/` and `reference/`. Both subpackages contain code
AND artifacts. The current code-vs-artifact split inside each is implicit
(underscore prefix for code, leaf dirs for artifacts) but visually noisy
and conceptually muddled.

Missing entirely: failure-recipe category, per-model lessons, benchmark
suite.

---

## 2. Principles (from user retrospective 2026-05-17)

**A. Wiring is minimal.** BlackJAX is designed to be modular. If a sampler
needs heavy wrapper code to connect to a warmup, that's a BlackJAX design
smell upstream — not something to paper over in tuningfork. Wiring code
should be readable in one sitting per cell: a recipe should let a reader
trace "load Posterior → call warmup → call sampler with these params"
top-to-bottom.

**B. Models delegate to a PPL.** A `Posterior` provides the standard
Bayesian-model interface — prior sample, prior predictive, logdensity_fn
conditioned on data, posterior predictive. NumPyro already supports
these; tuningfork's role is to ensure each model's `Posterior` interacts
cleanly with the wiring layer.

**B-bis. Groundtruth is a model property.** Every certified model has a
groundtruth: either analytic samples (closed-form) or a long single-chain
NUTS reference (curated for sample quality via the auto-gate). This is
intrinsic to the model, not to any particular recipe; it's the "true
distribution" against which any recipe's output is judged.

**C. Recipes record what worked AND what didn't.** Each (model × warmup ×
sampler) cell that produces a gate-passing setup gets a recipe artifact
recording (a) the parameters that made it work and (b) the engineering
+ wall effort to get there. Cells that *don't* work — even after
Statistician + HIGH-effort BO — also need a recipe artifact, recording
the failure diagnosis so future agents don't redo the work. This is the
missing FAILED category.

**D. Per-model lessons co-locate with the model.** The forking-path
narrative for sampling a particular model ("gp_regression has these
geometry quirks; horseshoe needs these tricks; stoch_vol requires
MCLMC") is a first-class library artifact, not an agent-team artifact.
Currently it lives in `worklog/lessons/case-studies/<model>/<date>-<topic>.md`
which works for the dev team but is invisible to library users.

**E. Benchmark = filter over recipes by wall time.** Don't introduce a
new abstraction. Each recipe carries `calibration_budget.wall_seconds`;
pytest can collect recipes where that's under a threshold and run them
as `@pytest.mark.benchmark` cases.

**F. Recipes are portable; generated scripts have zero tuningfork
dependency.** A recipe is the unit of reproducible inference. Given a
recipe JSON, a user — *with no knowledge of tuningfork* — should be
able to invoke a code-gen helper and obtain a runnable Python script
that imports only `jax`, `blackjax`, `numpyro`, `numpy`, `arviz`, and
(optionally) loads a sidecar data file. **No `import tuningfork`** in
the emitted script. The generated script is self-contained: model body
inlined, warmup + sampler calls expanded to direct BlackJAX function
calls, all hyperparameters hard-coded from the recipe pin.

The TUNINGFORK-side wiring (`base_method/`, `warmup/`, `smc/`) exists to
*generate* recipes. The recipe + generated script is the *output*. The
two flows are different audiences: contributor-of-new-cell vs
consumer-of-existing-recipe.

This also means **user-facing code (helpers, notebooks) sit with the
recipes, not with the wiring**. A user who wants to "load a recipe,
inspect its diagnostics, look at the lessons.md" pulls from one
location (`tuningfork/catalog/`); a user who wants to "extend tuningfork
with a new sampler wrapper" works in a completely different layer
(`tuningfork/base_method/`).

---

## 3. Specific pain points

### 3.1 Code/artifact muddle in `recipes/` and `reference/`

The two subpackages each blend Python module files with data file trees.
Result:

- The `_STARTER_ROOT` constant in `_generate_starter.py` points to a
  sibling-of-self subdir (`tuningfork/recipes/starter/`). This is
  awkward — the code resolves its own filesystem location to find an
  adjacent artifact tree. Any future packaging concern (wheels,
  package_data manifests) has to account for this.
- The `recipes/__init__.py` re-exports schema names. A user doing
  `from tuningfork.recipes import Recipe` correctly imports the schema
  class. But `tuningfork.recipes` as a Python package is more than just
  the schema — it also lexically *contains* every committed JSON artifact.
  Importing `tuningfork.recipes` doesn't expose the artifacts, but a
  directory listing makes them visible. Conceptually confusing.
- Same shape for `reference/`: it has `_io.py` (code) +
  `_posteriordb_xcheck.py` (code) + `<model>/` (artifact dirs).

### 3.2 Discoverability — "everything about gp_regression in one place"

A new contributor asking "what do I need to know about sampling
gp_regression?" today has to inspect FIVE locations:

1. `tuningfork/model/gp_regression.py` — model definition
2. `tuningfork/reference/gp_regression/{metadata,summary,adaptation,xcheck}.json`
   — cert artifacts
3. `tuningfork/recipes/starter/gp_regression/*.json` — recipe pins
4. `worklog/lessons/case-studies/gp_regression/*.md` — sampling quirks
   (agent-team artifact, not library-user-visible)
5. `worklog/threads/_archive/phase0-statistician-3holdouts.md` — closeout
   narrative

The library should consolidate (2)–(4) into one per-model directory.
(1) stays separate because it's code; (5) stays in worklog because it's
process narrative, not library artifact.

### 3.3 No failure recipe category

Today's `Effort` enum: `LOW`, `MEDIUM`, `HIGH`, `GROUNDTRUTH`. Missing:
**`FAILED`**.

Scenarios that currently have NO library-level recording:

- `mclmc` + `stoch_vol` at LOW (default params) — fails the auto-gate;
  Statistician identifies "needs MCLMC HP injection that isn't
  implemented yet"; HIGH-effort BO doesn't recover it either. Today:
  the failure exit memo lands in `worklog/lessons/case-studies/`. A
  user reading the recipe matrix sees a 🔴 cell with no detail.
- `rmhmc` + `horseshoe` — the model lacks a callable metric for the
  Riemannian manifold sampler. Hard-excluded category in
  `RECIPE_GENERATION.md`, but no artifact says "this won't work and
  here's why".
- `fullrank_vi` + 50-D ill-conditioned Gaussian — VI choice doesn't
  scale to high-d; statistician declares out-of-scope. Same problem.

A FAILED recipe is a real artifact with informational value: future
agents avoid re-running the experiment; users learn the diagnostic
rationale.

### 3.4 Benchmark suite undefined

`pyproject.toml` reserves the `benchmark` pytest marker but no test
uses it. The original library goal ("calibrated comparison of MCMC/VI/SMC
on a curated suite") needs a periodic-run mechanism that fires the
recipes and records performance over time, but the mechanism doesn't
exist.

### 3.5 Recipe artifacts are at risk of overwhelming the package

At ~700 JSON files across 14 model subdirs, the recipe artifact tree
becomes the largest single component of the repository by file count.
Burying it inside `tuningfork/recipes/starter/` (with its current
namespace + leading-underscore code siblings) is a smell — a Python
package shouldn't have hundreds of data files as siblings of its
module code.

---

## 4. Recommended layout

### 4.1 Two-layer split: wiring (generator) vs library (consumer)

The recipe-portability principle (Principle F) reframes the layout. The
package has TWO audiences with TWO entry surfaces:

- **Generator side** (`tuningfork.base_method`, `.warmup`, `.smc`,
  `.calibration`, `.metrics`, `.runner`): used by tuningfork ITSELF to
  produce recipes. A contributor adding a new sampler wrapper or
  warmup adapter works here. End users of recipes never touch these.

- **Library side** (`tuningfork.catalog`): used by anyone consuming
  existing recipes. Contains the recipe artifacts (per-model dirs),
  the user-facing API helpers (load recipe, summarize, render
  diagnostics), the code-gen function that emits standalone scripts,
  template notebooks, and per-model lessons.md. The library subpackage
  is the only thing a "regular user" needs to know about.

```
tuningfork/                              # repo root
├── tuningfork/                          # PYTHON PACKAGE
│   │
│   │   # ─────────── GENERATOR LAYER (produces recipes) ───────────
│   ├── model/                           # 14 numpyro models
│   │   ├── _base.py                     # Posterior schema
│   │   ├── _numpyro.py                  # logdensity_fn builder
│   │   ├── _registry.py                 # MODELS + MODELS_BY_FAMILY
│   │   ├── _data/                       # raw input datasets (CSV/NPZ)
│   │   ├── banana.py, eight_schools_ncp.py, ...   # 14 model files
│   │   └── __init__.py
│   ├── base_method/                     # 24 sampler wrappers + ENTRIES
│   ├── warmup/                          # 10 warmup wrappers + ENTRIES
│   ├── smc/                             # 6 SMC wrappers + ENTRIES
│   ├── recipes/                         # SCHEMA + GENERATORS
│   │   ├── _base.py                     # Recipe dataclass + Effort + FailureDiagnosis
│   │   ├── _instructions.py             # instruction template renderer
│   │   ├── _generate.py                 # groundtruth + starter orchestrators
│   │   ├── _emit_script.py              # NEW: code-gen function (recipe → standalone .py)
│   │   ├── _templates/                  # NEW: per-sampler / per-warmup script templates
│   │   │   ├── nuts.py.tmpl, hmc.py.tmpl, mclmc.py.tmpl, ...
│   │   │   ├── window_adaptation_diag_imm.py.tmpl, pathfinder.py.tmpl, ...
│   │   │   └── inference_loop.py.tmpl
│   │   └── __init__.py
│   ├── calibration/                     # cert + tune + auto-gate
│   ├── metrics/                         # headline + grad_counter + reference_compare
│   ├── runner/                          # SMC init + run helpers
│   ├── _cache_io.py                     # internal cache reader (was reference/_io.py)
│   ├── cli.py                           # CLI entry points (orchestrates both layers)
│   │
│   │   # ─────────── LIBRARY LAYER (consumes recipes) ───────────
│   └── catalog/                         # USER-FACING SUBPACKAGE
│       ├── __init__.py                  # exports load_recipe, load_idata, ...
│       ├── _api.py                      # CONSOLIDATED: was inspect.py + render.py
│       ├── diagnostics.py               # ArviZ family-aware rendering
│       ├── emit.py                      # public wrapper for recipe → script
│       ├── notebooks/                   # template + example notebooks
│       │   ├── recipe_diagnostics.md    # parametrized inspector
│       │   ├── inspect_example.md       # worked example
│       │   └── inspect_README.md        # user-API docs
│       └── <model>/                     # PER-MODEL ARTIFACTS
│           ├── lessons.md               # NEW: forking-path narrative
│           ├── groundtruth.json
│           ├── groundtruth.imm.npz      # sidecar (high-dim only)
│           ├── reference/               # committed cert artifacts
│           │   ├── metadata.json, summary.json, adaptation.json, xcheck.json
│           ├── recipes/                 # per-cell recipes
│           │   ├── low__nuts__window_adaptation_diag_imm.json
│           │   ├── medium__mala__window_adaptation_diag_imm.json
│           │   ├── high__hmc__window_adaptation_diag_imm.json
│           │   └── failed__rmhmc__window_adaptation_diag_imm.json    # NEW: failure recipe
│           └── _cache/                  # gitignored
│               ├── draws.npz, chain_stats.npz, warmup_checkpoint/
│
├── benchmarks/                          # NEW: test_benchmark suite (consumer side)
│   └── test_fast_recipes.py             # wall-time-filtered recipe runs
├── tests/                               # generator-side unit tests
├── tools/
├── RECIPE_GENERATION.md
├── README.md, CLAUDE.md, CONTRIBUTING.md
└── pyproject.toml
```

**Key changes from current**:

- **Drop `tuningfork/reference/` as a package subdir.** Code
  (`_io.py`, `_posteriordb_xcheck.py`) collapses into
  `tuningfork/_cache_io.py`. Per-model artifact dirs move to
  `tuningfork/catalog/<model>/reference/`.
- **Drop `tuningfork/recipes/starter/` namespace.** Artifacts move to
  `tuningfork/catalog/<model>/recipes/`. The `tuningfork/recipes/`
  package now holds ONLY code (schema + generators + templates).
- **Move `tuningfork/data/` → `tuningfork/model/_data/`.** Raw
  datasets are model inputs; they belong with model code.
- **Hoist user-facing helpers into `tuningfork/catalog/`**: was
  `tuningfork/inspect.py` + `tuningfork/render.py` (currently at
  package root), now consolidated in `tuningfork/catalog/_api.py`.
  Public API: `from tuningfork.catalog import load_recipe,
  summarize_recipe, load_idata, render_diagnostics, emit_script`.
- **Move `tuningfork/diagnostics.py` → `tuningfork/catalog/diagnostics.py`.**
  It's pure consumer-side rendering; belongs with the library API.
- **Move repo-root `notebooks/` → `tuningfork/catalog/notebooks/`.**
  The template notebooks operate on recipes and live next to them.
  (The notebooks remain Jupytext `.md` per project policy.)
- **Merge `_generate_groundtruth.py` + `_generate_starter.py` →
  `recipes/_generate.py`.** The two orchestrators share ~80%
  scaffolding.
- **Drop the `tuningfork/recipes/__init__.py` re-export shim** for
  the schema; new `recipes/__init__.py` exports only the schema
  symbols (`Recipe`, `Effort`, `FailureDiagnosis`).
- **New `tuningfork/catalog/<model>/lessons.md`** per certified model.
- **New `failed__*.json` recipe category** (see § 4.2).
- **New `tuningfork/recipes/_emit_script.py` + `_templates/`** —
  the recipe → standalone-script code-gen (see § 4.5).
- **New `benchmarks/`** at repo root (see § 4.4).

### 4.2 Failure recipe schema

**Framing (user direction 2026-05-17)**: a FAILED recipe is **"a hard
direction to land"**, not "this won't work". We cannot exhaustively
search the (warmup × sampler × HP) space, so a failure recipe signifies
*"conceptually this pairing makes sense, but the Statistician's
investigation through these specific forking paths did not produce a
gate-passing config"*. Future agents are free to try directions the
Statistician didn't — and a FAILED recipe is precisely the data that
tells them which directions ARE already explored.

This narrative framing has a schema implication: a FAILED recipe must
capture the **forking-path log** (all HP combinations attempted, with
per-attempt diagnostics), not just a single exit memo + closest-attempt
config.

#### Schema

Add `Effort.FAILED = "failed"` to the existing enum. A FAILED recipe
extends the existing Recipe with two new fields:

```python
@dataclass(frozen=True)
class Recipe:
    # ... existing fields (unchanged) ...
    effort: Effort                                        # adds FAILED as 5th value
    failure_diagnosis: FailureDiagnosis | None = None     # NEW (None for non-failed)
    attempted_configurations: list[AttemptedConfig] = field(default_factory=list)  # NEW
    workflow: str                                         # already exists; HEAVILY used for FAILED
                                                          # = high-level narrative; per-attempt
                                                          # detail in attempted_configurations


@dataclass(frozen=True)
class AttemptedConfig:
    """One forking-path branch the Statistician walked down."""
    base_method_params: dict          # the HP combination tried
    warmup_params: dict
    seed: int                         # tuning_seed of this attempt
    gate_verdict: dict                # rhat_max, min_bulk_ess, n_divergences, verdict
    wall_seconds: float               # how long this attempt cost
    note: str                         # one-line "why I tried this and what I saw"
                                      # e.g. "tighter step_size to recover from divergences;
                                      #       divergence rate dropped 5%→1% but ESS halved"


class FailureDiagnosis(StrEnum):
    OUT_OF_SCOPE          = "out_of_scope"           # sampler conceptually wrong for model class
                                                     # (e.g., gradient-free RWM on 503-D state-space)
    REQUIRES_ALT_SAMPLER  = "requires_alt_sampler"   # needs kernel not in v1 (e.g., SGMCMC)
    REQUIRES_MODEL_CHANGE = "requires_model_change"  # model parameterization needs work
                                                     # (e.g., latent GP marginalization)
    TRIVIAL_FIX_DEFERRED  = "trivial_fix_deferred"   # known fix, not landed yet
                                                     # (e.g., MCLMC HP injection gap)
    HARD_DIRECTION        = "hard_direction"         # tried-but-could-not-land; no specific
                                                     # diagnosis. Reader is invited to retry
                                                     # with a different forking path.
```

A FAILED recipe is emitted by the Statistician when their HIGH-effort
investigation closes without a gate-passing config. Semantically the
recipe records:

- **`failure_diagnosis`**: the diagnosis classification (often
  `HARD_DIRECTION` — see narrative framing above; `OUT_OF_SCOPE` is
  reserved for cases where the sampler family is conceptually wrong
  for the model class).
- **`attempted_configurations`**: the complete forking-path log
  (typically 5–30 attempts for a HIGH-effort investigation). Each
  attempt is a self-contained record a future agent can read and say
  "this branch was tried, here's what was seen — I'll try a different
  branch."
- **`workflow`**: the narrative connecting the attempts. Why this
  ordering, what hypothesis each attempt was testing, what the
  Statistician concluded.
- **`base_method_params` / `warmup_params`**: at the top level, the
  **closest-to-passing** attempt's config (a useful starting point for
  a future agent picking up the thread). Not "the answer" — there is no
  answer.
- **`calibration_budget`**: total wall + engineering time spent across
  all attempts.

#### Filename convention

`failed__<sampler>__<warmup>.json`. Example:

```json
{
  "model_name": "stoch_vol",
  "base_method_name": "mclmc",
  "warmup_name": "mclmc_tuning",
  "effort": "failed",
  "failure_diagnosis": "hard_direction",
  "workflow": "MCLMC on 503-D NCP stoch_vol: explored 8 forking paths \
across step-size scaling, L-tuning regimes, and IMM-rank choices. None \
recovered a gate-pass. Closest attempt (#7) reached min_bulk_ESS=287 \
which is below the 400 threshold; the path was abandoned when extending \
the tuning budget by 2× produced no further improvement. We DID NOT try: \
hand-tuned step-size schedules, alternative L distributions, hybrid \
NUTS-warmstart + MCLMC-production. A future agent picking this up should \
start there.",
  "attempted_configurations": [
    {
      "base_method_params": {"step_size": 0.01, "L": 5.0},
      "warmup_params": {"n_warmup": 1000},
      "seed": 42,
      "gate_verdict": {
        "rhat_max": 1.18, "min_bulk_ess": 42.1, "n_divergences": 0,
        "verdict": "FAIL"
      },
      "wall_seconds": 89.0,
      "note": "Default MCLMC tuning — ESS too low. Hypothesis: step_size too aggressive for narrow latent valleys."
    },
    {
      "base_method_params": {"step_size": 0.001, "L": 5.0},
      "warmup_params": {"n_warmup": 1000},
      "seed": 42,
      "gate_verdict": {
        "rhat_max": 1.04, "min_bulk_ess": 287.0, "n_divergences": 0,
        "verdict": "FAIL"
      },
      "wall_seconds": 95.0,
      "note": "10× smaller step_size — ESS climbs but still under threshold. CLOSEST attempt."
    }
    // ... 6 more attempts ...
  ],
  "base_method_params": {"step_size": 0.001, "L": 5.0},   // closest attempt (#2)
  "warmup_params": {"n_warmup": 1000},                    // closest attempt (#2)
  "tuning_seed": 42,
  "calibration_budget": {
    "trials": 8,
    "wall_seconds_estimate": 720.0,
    "statistician_wall_hours": 4.0
  },
  "gate_evidence": {
    "auto": {
      "verdict": "FAIL",
      "rhat_max": 1.04,        // closest attempt
      "min_bulk_ess": 287.0,   // closest attempt — note < 400 gate
      "n_divergences": 0
    }
  }
}
```

#### Consumer-side semantics

`Recipe.load_cached_samples()` on a FAILED recipe raises
`RecipeFailedError` with `failure_diagnosis` + a pointer to
`attempted_configurations` and `workflow`. Notebooks, CLI, and the test
harness all check `recipe.is_failed()` before attempting to run.

`emit_script(failed_recipe)` (the code-gen function) emits a script
that runs the **closest-to-passing** attempt and reports the gate
verdict — but the script's docstring prominently states "This recipe
FAILED to clear the auto-gate; the emitted script reproduces the
closest attempt for diagnostic purposes only. See workflow for the
forking-path narrative."

### 4.3 Per-model lessons.md

Living next to the recipes for each model: a markdown file with
free-form sections covering sampling quirks. Suggested template:

```markdown
# Sampling lessons: <model_name>

## TL;DR
One-line summary of "what's tricky about sampling this model".

## Canonical recipe
What the lowest-effort working setup is (link to the LOW recipe).

## Sampling quirks
Specific geometry / parameterization issues a sampler must handle.
E.g., "GP regression has strong correlation between log_lengthscale
and log_sigma_f; diagonal IMM is insufficient at high N."

## Known-bad combinations
List of FAILED recipes with one-line rationale each. Links to the
failed JSONs.

## History
References to relevant worklog case-studies + decision docs.

## Citations
Papers, Stan reference parameterizations, etc.
```

Initial population comes from migrating existing `worklog/lessons/case-studies/<model>/*.md` content into the corresponding `catalog/<model>/lessons.md`. The worklog substrate stays as the agent-team
process layer; the catalog lessons.md is the user-facing distilled
version.

### 4.4 Benchmark suite

Pattern:

```python
# benchmarks/test_fast_recipes.py
import pytest
from tuningfork.recipes import iter_recipes, Effort

# Pre-collect at module load: all recipes with predicted wall < 120 s
_FAST_RECIPES = [
    r for r in iter_recipes()
    if r.effort != Effort.FAILED
    and r.calibration_budget["wall_seconds_estimate"] < 120
]


@pytest.mark.benchmark
@pytest.mark.parametrize("recipe", _FAST_RECIPES, ids=lambda r: r.cell_id)
def test_recipe_runs_under_budget(recipe, benchmark):
    """Run the recipe and check it finishes within 2× pinned wall."""
    result = benchmark(recipe.run)
    assert result.gate_verdict == "PASS"
    assert benchmark.stats["mean"] < 2 * recipe.calibration_budget["wall_seconds_estimate"]
```

CI invocation: `make benchmark` runs `pytest -m benchmark --benchmark-json=...`.
Output ingested into a tracking dashboard (or just stored as artifacts;
trend analysis is a separate concern).

This is intentionally minimal — the library design says "recipes are
self-describing units of work", so the benchmark is just a wall-time
filter + a thin assertion harness.

---

### 4.5 Recipe portability — `emit_script(recipe)` code-gen

A recipe is the unit of reproducible inference. The library exposes a
single public function:

```python
from tuningfork.catalog import emit_script, load_recipe

recipe = load_recipe("eight_schools_ncp/recipes/low__nuts__window_adaptation_diag_imm.json")
script_text = emit_script(recipe)

# Write to disk and run standalone (no tuningfork dependency):
Path("run_eight_schools.py").write_text(script_text)
# $ uv run --with jax --with blackjax --with numpyro --with arviz \
#         run_eight_schools.py
```

The emitted script is **standalone**: zero `import tuningfork` lines.
It depends only on `jax`, `jax.numpy`, `blackjax`, `numpyro`,
`numpy`, and (optionally) `arviz` for diagnostics. Sample contents:

```python
"""Auto-generated from tuningfork recipe.

Source: eight_schools_ncp/recipes/low__nuts__window_adaptation_diag_imm.json
Recipe hash: e7d4f9a2... (matches tuningfork == 0.x.y, blackjax == ...)
Effort: low. Verdict: PASS (R̂=1.005, min_bulk_ESS=2451, n_div=0).
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import blackjax

# === MODEL: eight_schools_ncp (inlined from tuningfork/model/eight_schools_ncp.py) ===
J = 8
y_obs = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma_obs = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def eight_schools_ncp(y, sigma):
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    theta_raw = numpyro.sample("theta_raw", dist.Normal(0.0, 1.0).expand([J]))
    theta = mu + tau * theta_raw
    numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)

from numpyro.infer.util import initialize_model
init_params, potential_fn, *_ = initialize_model(
    jax.random.key(0), eight_schools_ncp, model_args=(y_obs, sigma_obs)
)
def logdensity_fn(position):
    return -potential_fn(position)

# === WARMUP: window_adaptation_diag_imm (n_warmup=1000, target_acceptance_rate=0.8) ===
key = jax.random.key(42)  # seed pinned by recipe
init_position = init_params.z

warmup = blackjax.window_adaptation(
    blackjax.nuts, logdensity_fn,
    target_acceptance_rate=0.8, progress_bar=False,
)
(state, adapted_params), _ = warmup.run(key, init_position, num_steps=1000)

# === SAMPLER: nuts (step_size=0.234, max_num_doublings=10) ===
# Note: inverse_mass_matrix is adapted at warmup time; no sidecar needed
# for this LOW recipe (d=10 inline). High-dim recipes would np.load() here.
kernel = blackjax.nuts(
    logdensity_fn,
    step_size=adapted_params["step_size"],
    inverse_mass_matrix=adapted_params["inverse_mass_matrix"],
    max_num_doublings=10,
).step

# === INFERENCE LOOP ===
@jax.jit
def one_step(state, rng_key):
    state, info = kernel(rng_key, state)
    return state, (state, info)

NUM_SAMPLES = 4000
keys = jax.random.split(jax.random.key(43), NUM_SAMPLES)
_, (samples, infos) = jax.lax.scan(one_step, state, keys)

# samples.position is a dict[str, jax.Array]; ready for ArviZ:
print(f"R̂ max: ...; min bulk-ESS: ...; divergences: {int(infos.is_divergent.sum())}")
```

#### Architecture

The code-gen is template-based. Each `base_method/<name>.py`,
`warmup/<name>.py`, and `model/<name>.py` defines BOTH:
1. The Python wrapper / `ENTRY` (used by tuningfork at generation
   time).
2. An `EMIT_TEMPLATE: str` constant (a parameterized snippet used at
   script-emission time).

The templates live in `tuningfork/recipes/_templates/`:

```
_templates/
├── preamble.py.tmpl              # imports + recipe-hash docstring header
├── models/<model>.py.tmpl        # one per model — inlined NumPyro body + data loader
├── warmups/<warmup>.py.tmpl      # one per warmup — direct blackjax call
├── samplers/<sampler>.py.tmpl    # one per sampler — direct blackjax call + inference loop
└── postamble.py.tmpl             # headline metric + verdict reporting
```

`emit_script(recipe)` assembles:

```python
def emit_script(recipe: Recipe) -> str:
    return "\n\n".join([
        render(PREAMBLE_TEMPLATE, recipe_hash=recipe.hash, ...),
        render(MODEL_TEMPLATES[recipe.model_name]),
        render(WARMUP_TEMPLATES[recipe.warmup_name], **recipe.warmup_params),
        render(SAMPLER_TEMPLATES[recipe.base_method_name], **recipe.base_method_params),
        render(INFERENCE_LOOP_TEMPLATE),
        render(POSTAMBLE_TEMPLATE, expected_verdict=recipe.gate_evidence["auto"]),
    ])
```

#### Data inlining vs. sidecar

For models with raw data (radon, irt_2pl, stoch_vol, german_credit,
horseshoe, gp_regression, lotka_volterra):
- **Small data (n ≤ 100 floats)**: inlined as `jnp.array([...])` in the
  model template.
- **Larger data**: the emit_script function writes a sidecar `.npy` or
  `.csv` file next to the script and the emitted script does
  `np.load("data_<model>.npy")`. The user gets a tarball with
  `run_<recipe>.py` + `data_<model>.npy`.

#### IMM sidecar handling

High-dim recipes (stoch_vol, gp_regression, ...) already carry an
`.imm.npz` sidecar. `emit_script` copies that file alongside the
generated script and emits `np.load("groundtruth.imm.npz")` in the
sampler template.

#### What this buys

1. **Recipes are portable units of inference.** A user takes a recipe +
   the generated script + the data sidecar (if needed); runs in any
   fresh environment with just `jax/blackjax/numpyro` installed.
2. **The wiring layer is provably minimal.** If a sampler's template
   exceeds, say, 30 lines, it's a signal the sampler has heavy wrapping
   in tuningfork. Per Principle A, that's a BlackJAX design issue
   upstream — fix there, not here.
3. **Templates make the wiring auditable.** Reading
   `_templates/samplers/nuts.py.tmpl` is faster than tracing through
   `base_method/nuts.py` import chains. Newcomers can verify "is the
   recipe doing what I think it's doing" by reading a 20-line template.
4. **Versioning via recipe hash.** The script docstring includes the
   recipe hash + the tuningfork version that generated it. Users can
   diff the emitted script against a re-emission later to detect
   silent regressions in the template wiring.

#### Cost / risk

- Each sampler / warmup / model needs a template, in addition to its
  Python wrapper. ~50 templates total (24 samplers + 10 warmups +
  14 models + 2-3 cross-cutting). Each ~10-30 lines. **Total: ~1000
  lines of templates** added to the codebase.
- Templates and wrappers can drift. **Mitigation**: a CI gate that
  emits a script for each committed recipe + executes it (under
  fast suite limits — small sample counts) and verifies the gate
  verdict matches the recipe's pinned `gate_evidence.auto`. This
  doubles as the benchmark suite (§ 4.4).

---

## 5. Alternatives considered

### 5.1 Option A: keep current; just drop `starter/` namespace

```
tuningfork/recipes/<model>/<file>.json
tuningfork/reference/<model>/<file>.json
```

**Pros**: minimal churn (just rename one dir + update `_STARTER_ROOT`).
**Cons**: doesn't solve the code/artifact muddle. Doesn't address
lessons.md, FAILED recipes, or benchmark. Doesn't consolidate per-model
artifacts. Doesn't free `tuningfork/recipes/` and `tuningfork/reference/`
from being half-code-half-data subpackages.

**Verdict**: tactical fix, doesn't address the deeper structural
question. Reject in favor of full restructure.

### 5.2 Option B: artifacts at repo root, parallel to `tuningfork/`

```
tuningfork/
├── tuningfork/          # PACKAGE (code only, with everything moved up)
└── catalog/             # ALL ARTIFACTS at repo root
    └── <model>/...
```

**Pros**: cleanest possible code/artifact split — artifacts aren't even
*inside* the Python package.
**Cons**: packaging complications — `pip install tuningfork` wouldn't
include the artifact tree without `MANIFEST.in` or `package_data`
declarations. `Recipe.load()` path resolution becomes harder (where's
the library root for an installed-via-pip user?).

**Verdict**: cleanest in theory; worse in practice for an
open-sourced-as-pip-package library. The next-best after § 5.3.

### 5.3 Option C: recommended (§ 4) — `tuningfork/catalog/<model>/`

**Pros**: inside the package (packaging works trivially via
`include_package_data = true`); per-model directory is the unit;
clear code-vs-artifact split (everything-except-library is code,
library is artifacts).
**Cons**: 50+ files move; need to update every path constant + every
test fixture. Migration is real work.

**Verdict**: recommended.

### 5.4 Option D: leave `reference/` per-model, move recipes only

Keep `reference/<model>/` where it is; only move recipes to
`tuningfork/catalog/<model>/recipes/`.

**Pros**: smaller churn than (C).
**Cons**: asymmetric — two parallel per-model trees. The whole point
is consolidating per-model artifacts. Doesn't add lessons.md or
benchmarks.

**Verdict**: half-measure. Reject.

---

## 6. Migration plan

Phased to minimize churn risk and validate each step:

### Phase R1: Schema work (~3 commits; Junior-SWE or TL)

- Add `Effort.FAILED` enum value
- Add `FailureDiagnosis` enum + `failure_diagnosis` Recipe field
- Add `Recipe.is_failed()` method + custom `RecipeFailedError`
- Update `_instructions.py` to render a FAILED-tier prose template
- Add tests covering FAILED recipe load / save round-trip
- **Gate**: schema is additive; existing recipes continue to load.

### Phase R2: Move artifacts to `tuningfork/catalog/<model>/` (~5 commits; SWE)

- `git mv tuningfork/reference/<model>/* tuningfork/catalog/<model>/reference/` (14 model subdirs)
- `git mv tuningfork/recipes/starter/<model>/* tuningfork/catalog/<model>/recipes/` (rename groundtruth files to `groundtruth.json`)
- Rewrite path constants:
  - `_STARTER_ROOT` (in `_generate_starter.py`) → `_LIBRARY_ROOT / model_name / "recipes"`
  - `_io.py` cache helpers → look in `catalog/<model>/` instead of `reference/<model>/`
- Update `.gitignore` patterns
- Rewrite ~30 string-literal path refs (notebooks, CLI, docstrings, tests)
- **Gate**: full fast suite + slow suite pass; `git check-ignore -v` audit clean.

### Phase R3: Code reorg inside the package (~4 commits; SWE)

- `git mv tuningfork/reference/_io.py tuningfork/_cache_io.py`
- `git mv tuningfork/reference/_posteriordb_xcheck.py` content into
  `_cache_io.py` (single file is fine — it's all cache I/O)
- `rmdir tuningfork/reference/`
- `git mv tuningfork/data/ tuningfork/model/_data/` (raw datasets =
  model inputs)
- Merge `_generate_groundtruth.py` + `_generate_starter.py` → `recipes/_generate.py`
- `git mv tuningfork/model/__init__.py` → `tuningfork/model/_registry.py`; new `__init__.py` re-exports
- **Library subpackage**: `git mv tuningfork/inspect.py tuningfork/catalog/_api.py` (merging with `render.py` → same file); `git mv tuningfork/diagnostics.py tuningfork/catalog/diagnostics.py`; `git mv notebooks/ tuningfork/catalog/notebooks/`
- `tuningfork/catalog/__init__.py` re-exports the user-facing API
- Update all imports across source + tests + notebooks
- **Gate**: fast + slow suite pass; pre-commit clean.

### Phase R3.5: Recipe portability — emit_script + templates (~5 commits; SWE)

- Add `tuningfork/recipes/_templates/{preamble,postamble,inference_loop}.py.tmpl`
- Add `tuningfork/recipes/_templates/models/<name>.py.tmpl` for each
  of the 14 models — each template inlines the NumPyro model body + data
  loader. Auto-extractable from `tuningfork/model/<name>.py` source via
  AST inspection, or hand-written (template + source verified against
  each other in tests)
- Add `tuningfork/recipes/_templates/warmups/<name>.py.tmpl` for each
  of the 10 warmups (~10-20 lines each)
- Add `tuningfork/recipes/_templates/samplers/<name>.py.tmpl` for each
  of the 24 samplers (~10-30 lines each; flag any that exceed 30 lines
  as upstream BlackJAX design smells per Principle A)
- Implement `tuningfork/recipes/_emit_script.py` (the assembler)
- Public wrapper at `tuningfork/catalog/emit.py` re-exports
- Round-trip test: for each committed recipe, `emit_script(recipe)` →
  exec the emitted script with `n_samples=200` → verify cell runs
  without error AND `R̂ < 1.05` (loose threshold for the smoke gate)
- **Gate**: round-trip test passes for all committed recipes; template-
  drift CI gate active (emit + exec on every PR).

### Phase R4: Add `lessons.md` per model (~1 commit; TL or tech-writer)

- For each of the 14 models, create
  `tuningfork/catalog/<model>/lessons.md` initial template
- For models with prior worklog case-studies (gp_regression, stoch_vol,
  horseshoe, eight_schools_ncp, ...): port distilled content from the
  worklog `lessons/case-studies/<model>/*.md` files
- The worklog substrate stays as-is; lessons.md is the user-facing
  distilled version
- **Gate**: lessons.md exists for all 14 models; pre-commit clean.

### Phase R5: Backfill FAILED recipes (~1-3 commits; statistician + TL)

- Identify "hard direction" cells from the Phase 5 cell matrix and
  `RECIPE_GENERATION.md` red zones
- Statistician writes FAILED recipe JSON + exit memo for each (e.g.,
  rwm × stoch_vol, fullrank_vi × ill_cond_50, mclmc × gmm_25 multimodal,
  ...)
- Estimate: ~30-50 FAILED recipes for v1 (covering hard-excluded
  categories from the matrix's R-cell column)
- **Gate**: each FAILED recipe loads + reports diagnosis cleanly via
  the new `Recipe.is_failed()` check.

### Phase R6: Benchmark suite (~2 commits; SWE)

- Add `benchmarks/test_fast_recipes.py` with the recipe-iter + filter pattern
- Add `make benchmark` target → `pytest -m benchmark --benchmark-json=...`
- Wire CI workflow file
- **Gate**: `make benchmark` runs locally; ≥ 5 recipes collected;
  all pass within 2× pinned wall.

### Phase R7: Documentation refresh (~1 commit; tech-writer)

- README "Layout" section updated to new tree
- CLAUDE.md architecture diagram updated
- CONTRIBUTING.md updated (test markers + benchmark guidance)
- New section in README: "The library" — links to a sample `catalog/<model>/lessons.md` + recipe + reference
- **Gate**: docs review.

**Total effort estimate**: 7 phases, ~15 commits, ~2-3 working days
of agent time. Largest risk is R2 (broad rename + 30+ path string
updates) which we've now done twice in this session (PR #6 + #7) so
the playbook is well-rehearsed: bulk-sed `find + grep -rl` patterns,
then verify with the pattern audit.

---

## 7. Locked decisions (user signoff 2026-05-17)

All 10 open questions answered. Recorded here for reference; the
companion decision doc lives at
`worklog/decisions/2026-05-17-tuningfork-restructure.md`.

| # | Question | Decision |
|---|----------|----------|
| 1 | Overall direction | **YES** — proceed with two-layer split + per-model catalog. |
| 2 | Folder name | **`catalog/`**. (Sweep: removed `inventory/` as a mentioned alternative in this proposal; README/CLAUDE.md uses of "inventory" describe the cell matrix and remain unchanged — different concept.) |
| 3 | `_data/` move to `tuningfork/model/_data/` | **YES**. Raw datasets are model inputs; co-locate with model code. |
| 4 | Failure-recipe filename | **`failed__<sampler>__<warmup>.json`**. Reframing in § 4.2: "this is a hard direction to land," not "this won't work." |
| 5 | Benchmark wall-time policy | **Selection filter at 180 s; test execution cap at 240 s** (≈ 1.33× headroom over the pinned estimate). Recipes with `calibration_budget.wall_seconds_estimate < 180` are collected; each test fails if it exceeds 240 s wall. |
| 6 | R2/R3 sequencing | **Split**. R2 (artifact move) lands as its own PR; R3 (code reorg) follows. Each gets independent review + CI. |
| 7 | Worklog lessons compatibility | **Keep** — `worklog/lessons/case-studies/<model>/` substrate stays alongside `catalog/<model>/lessons.md`. Different audiences (agent process record vs user-facing distillation). |
| 8 | Code-gen strictness | **Strict** for the inference part — emitted script has zero `import tuningfork` AND no reference to tuningfork file paths in its inference code path. **Cross-check test added**: a separate integration test verifies that `from tuningfork.model import <name>` paired with the emitted function (warmup + sampler bodies) composes correctly — i.e., the strict-emitted-script and the tuningfork-imported-model produce identical kernel output given the same seed. This double-checks the model inlining round-trips. **Initial-position confirmation**: the recipe carries `tuning_seed: int` (already present); init_position is derived deterministically from `(model, seed)` via NumPyro's init mechanism. No separate `initial_position` field needed in v1. |
| 9 | `emit_script` output target | **Whatever is easier given (8)**. Recommended `return-string` (pure function; user writes wherever) unless the strict cross-check test gates more naturally on a CLI-managed output dir. SWE picks during R3.5 build. |
| 10 | Template authoring | **Hand-written + round-trip CI gate** (the recommendation in the original proposal). Templates live in `tuningfork/recipes/_templates/`; CI executes each emitted script at low n_samples to catch drift. |

### Notes on Q1 (FAILED recipe semantics, per user comment)

A FAILED recipe records "a hard direction to land," not a categorical
"won't work." The schema (§ 4.2) captures the **full forking-path log**:
every HP combination the Statistician tried, with per-attempt gate
diagnostics and a one-line "why I tried this / what I saw" note. The
top-level `base_method_params` reports the **closest-to-passing**
attempt as a starting point for future agents. Future agents picking
up a FAILED thread read the `workflow` narrative + `attempted_configurations`
list, identify directions NOT yet tried, and either land a passing
recipe (upgrading to LOW/MEDIUM/HIGH) or extend the failure log with
their own attempts.

The new `FailureDiagnosis` value `HARD_DIRECTION` is the default — it
reads as "tried these paths, none worked, you're welcome to try
others." The other values (`OUT_OF_SCOPE`, `REQUIRES_ALT_SAMPLER`,
`REQUIRES_MODEL_CHANGE`, `TRIVIAL_FIX_DEFERRED`) are reserved for
narrower diagnoses.

### Notes on Q5 (benchmark thresholds)

Filter at 180 s / cap at 240 s is the v1 policy. Implementation:

```python
# benchmarks/test_fast_recipes.py
_FAST_RECIPES = [
    r for r in iter_recipes()
    if r.effort not in (Effort.FAILED, Effort.GROUNDTRUTH)
    and r.calibration_budget["wall_seconds_estimate"] < 180
]

@pytest.mark.benchmark
@pytest.mark.timeout(240)        # hard cap on test execution wall
@pytest.mark.parametrize("recipe", _FAST_RECIPES, ids=lambda r: r.cell_id)
def test_recipe_runs_under_budget(recipe, benchmark):
    result = benchmark(recipe.run)
    assert result.gate_verdict == "PASS"
```

The 60 s margin between selection (180) and execution cap (240) absorbs
CI runner variance and benchmark warmup overhead.

---

## 8. What this proposal does NOT change

- BlackJAX API integration: no change to how tuningfork wires to
  BlackJAX. If wiring is heavy, fix BlackJAX upstream per Principle A.
- Auto-gate thresholds (R̂ < 1.01, min bulk-ESS ≥ 400, etc.): unchanged.
- Reference protocol (1 chain × 40k samples × 4 chunks for split-R̂):
  unchanged.
- Cache invalidation policy: unchanged (audit-trail-only `code_sha`;
  Statistician owns "needs redo").
- Effort taxonomy semantics (LOW/MEDIUM/HIGH gate-driven escalation):
  unchanged. Only adds the FAILED tier.

---

## 9. Decision checkpoint — ✅ LOCKED 2026-05-17

All 10 questions answered (see § 7 above). Companion decision doc:
[`worklog/decisions/2026-05-17-tuningfork-restructure.md`](../worklog/decisions/2026-05-17-tuningfork-restructure.md).

- [x] **Overall direction**: per-model `catalog/<model>/` + two-layer
      split (generator vs catalog)
- [x] **Folder name**: `catalog/`
- [x] **`_data/` move** to `tuningfork/model/_data/`: YES
- [x] **FAILED recipe naming**: `failed__<sampler>__<warmup>.json`;
      narrative reframed as "hard direction to land"; full
      forking-path log in `attempted_configurations`
- [x] **Benchmark thresholds**: select at < 180 s, execute cap at
      ≤ 240 s
- [x] **R2/R3 sequencing**: split into two PRs
- [x] **Worklog lessons substrate**: kept alongside `catalog/lessons.md`
- [x] **Code-gen strictness**: STRICT for inference part; cross-check
      test gates `from tuningfork.model import <name>` + emitted
      function composition; `tuning_seed` suffices for init_position
      reproducibility
- [x] **`emit_script` output**: whatever R3.5 SWE finds easier given
      the strict cross-check requirement
- [x] **Template authoring**: hand-written + round-trip CI gate

Next steps: TL writes the worklog decision doc, then spawns R1
(schema work — adds `Effort.FAILED`, `FailureDiagnosis`,
`AttemptedConfig` to `tuningfork.recipes._base`).
