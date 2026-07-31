# Codegen-First Recipe Lifecycle

**Status:** Accepted

**Date:** 2026-07-30

## Decision

Code generation is the only supported sampling path for tuningfork experiments
and catalog recipe evidence. The narrow direct-sampling exceptions are canonical
ground-truth/reference generation in ``calibration/certify_reference.py`` and
``groundtruth/_nuts_multichain.py``; analytic or deterministic reference
builders are also outside the sampling route.

A recipe is not only a final sampler configuration. It is the versioned,
progressively enriched record of the work:

```text
recipe intent
    → resolve and generate warmup + sampling code
    → execute the generated program
    → persist samples, chain statistics, and an execution receipt
    → evaluate against diagnostics and ground-truth samples
    → append the attempt and gate evidence to the recipe
    → stamp the recipe PASS or FAIL
    → assign an easy / medium / hard title from the recorded effort
```

The generated program must faithfully implement the recipe. A sampling
experiment that needs a hand-written script exposes a missing recipe or codegen
capability. That capability must be added to the schema and generator, covered
by a regression test, and the redundant script then removed.

Tuningfork is a clone-first development repository. Git LFS-backed ground-truth
samples are central development inputs, not optional package data. Producing a
small standalone wheel, supporting a data-free installation, or otherwise
optimizing for conventional package distribution is not a goal of this design.

## Why this is the boundary

Multiple executable sampling paths make it possible for a successful experiment
to exercise different warmup, random-key, parameter, or diagnostic behavior
from the recipe that is eventually committed. They also accumulate procedural
scripts that encode one-off versions of the same routing logic.

Making codegen the execution boundary gives tuningfork one auditable answer to
the question:

> What sampling routine did this recipe request, and what routine actually ran?

The recipe and its execution receipt answer that question together. Small,
explicit generation stages follow BlackJAX's design preference for composable
functions and explicit state while keeping the emitted program standalone.

## Scope

This decision applies to:

- MCMC, VI, and SMC recipe production;
- calibration and parameter-search candidates;
- sampled ground-truth production and certification, except the canonical
  direct NUTS reference-generation paths named above;
- catalog re-runs, revalidation, and performance experiments whose results may
  become recipe evidence; and
- benchmarks that claim to execute a catalog recipe.

Analytic or deterministic reference builders do not need to pretend to be a
sampling route. Their outputs, and the canonical direct-reference exceptions,
must still enter the same recipe evidence envelope with exact provenance.

Pure algorithm prototypes and unit tests may call lower-level BlackJAX APIs
directly. Their samples are not admissible as catalog or certification evidence.
Once a prototype is intended to inform a recipe, the minimal corresponding
capability must move into codegen.

## Recipe model

One recipe document remains the practical unit that an engineer or agent reads
and updates. Within that document, the following concepts are distinct.

### Stable identity

`recipe_id` identifies the research cell or question. It must not depend on:

- the current filename or title;
- PASS or FAIL;
- easy, medium, or hard;
- machine or dependency versions; or
- which attempt is currently selected.

Changing executable intent produces a new configuration revision and hash. It
does not erase earlier revisions or attempts from the recipe.

### Intent

Intent is the declarative input to codegen:

- model and model/data revision;
- sampler or SMC method;
- ordered warmup stages and warmup inner kernel;
- initialization, precision, and compatibility policies;
- requested parameter overrides and callable policies;
- seed policy, chain count, warmup count, sample count, and other budgets;
- diagnostic and gate policy versions; and
- the ground-truth reference and protocol to use for evaluation.

Requested values must be distinguishable from adapted or otherwise resolved
values. In particular, a parameter learned during warmup is evidence produced by
an attempt, even if it later becomes the pinned input for a reproduction run.

### Resolved route

Codegen resolves intent into an explicit execution plan before rendering source.
The plan identifies every warmup stage, the sampling kernel, random-key
choreography, resolved parameters, diagnostics to collect, and artifact outputs.

The normalized plan and its hash are recorded in the recipe. The emitted source
is a regenerable derivative of that plan, not a second source of truth.

No material behavior may be supplied by an undocumented default. A recipe field
must either:

1. affect the normalized plan or emitted manifest;
2. be explicitly declared evaluation-only or presentation-only; or
3. be rejected as unsupported.

### Attempts and evidence

Every execution appends an attempt, including executions that fail, error,
diverge, time out, or require review. An attempt records at least:

- an attempt identifier and rationale;
- a snapshot and hash of executable intent;
- the normalized plan and emitted-program hash;
- code, generator, schema, dependency, model, and data revisions;
- seed and sampling budgets;
- machine and precision information;
- sample and chain-stat artifact references and content hashes;
- the exact ground-truth Git LFS object identifier or content hash;
- quality metrics and their measurement conditions;
- gate policy version, thresholds, margins, and automatic verdict;
- any review or override without replacing the automatic verdict;
- wall time and other effort evidence; and
- failure diagnosis, intervention, and what was learned.

The current successful configuration may be materialized as a convenient view,
but it must point to a retained attempt. It must never replace the attempt
history.

### Gate and certification state

Lifecycle stage, gate verdict, certification stamp, and effort title are
orthogonal:

- lifecycle stage: `DRAFT`, `GENERATED`, `SAMPLED`, `EVALUATED`, or `CURATED`;
- attempt verdict: `NOT_RUN`, `PASS`, `REVIEW`, `FAIL`, or `ERROR`;
- final certification stamp: `PASS` or `FAIL`; and
- effort title: `easy`, `medium`, or `hard`.

`REVIEW` must be resolved before a final certification stamp is written. An
override records its actor, reason, timestamp, and the exact attempt and gate
evidence it addresses. It never deletes or rewrites the automatic verdict.

`FAILED` is an outcome, not an effort level. `GROUNDTRUTH` is a recipe purpose,
not an effort level. The target schema must stop encoding either concept as an
easy/medium/hard tier.

### Effort title

The curation step assigns `easy`, `medium`, or `hard` from recorded production
effort. The label describes how difficult the configuration was to discover; it
does not describe posterior quality, robustness, or expected user runtime.

The title must cite a versioned rubric and effort evidence such as:

- number and kind of generated attempts;
- machine wall time and declared search budget;
- initialization, seed, or routing interventions;
- parameter search or model-specific work; and
- whether the attempted direction was exhausted without a passing result.

A curation agent may propose or write the title, but schema validation constrains
the label and requires its provenance. Titles and effort labels are presentation
metadata; they never participate in recipe identity.

The existing low/medium/high records map naturally to easy/medium/hard during
migration, but their original values and wording must be retained as provenance.

## Codegen architecture

The implementation should expose small transformations with one-way
dependencies:

```text
validate intent
    → resolve execution plan
    → emit standalone program
    → launch generated program
    → read run artifacts
    → evaluate and gate
    → append attempt to recipe
    → certify and curate
```

Responsibilities are:

- **Recipe/schema:** lossless loading, validation, canonicalization, identity,
  and pure recipe updates.
- **Planner:** recipe intent to a typed, normalized execution plan.
- **Emitter:** execution plan to standalone Python plus an embedded manifest.
- **Launcher:** execute only a generated program and capture its receipt.
  The two canonical direct-reference paths are isolated exceptions and still
  record the same reference provenance and certification artifacts.
- **Evaluator:** run artifacts plus exact ground truth to metrics and gate
  evidence. It does not construct a sampler.
- **Recipe writer:** atomically append evidence and update the materialized
  current view. It does not run inference.
- **Workspace:** resolve the clone, catalog, local artifacts, and Git LFS
  references, and fail clearly when required data is not hydrated.
- **Curator:** assign effort metadata and title after evidence is present.

MCMC and SMC retain typed, family-specific plans and metric payloads. They share
the lifecycle envelope; they do not need to share one untyped execution schema.

The existing descriptor-driven emitters are the intended direction. Shared
capabilities belong in typed descriptors or small plan/emit functions. They
must not be replaced with a single branch-heavy generator.

An in-process runner may temporarily remain as a compatibility shim or test
oracle. It is not a second production implementation: supported callers must
make it generate and launch the emitted program. It cannot produce admissible
recipe evidence by constructing and running a sampler directly, outside the
canonical direct-reference exceptions named above.

## Generated-program contract

Before sampling, each emitted program writes or embeds a manifest containing:

- `recipe_id` and executable configuration hash;
- normalized execution-plan hash;
- emitted-source hash;
- recipe schema and generator version;
- source-code revision;
- model/data and ground-truth revisions;
- resolved sampler and warmup identities and parameters;
- seed, chain count, warmup count, and sample count; and
- dependency and precision information.

The launcher and recipe writer reject results if the receipt does not match the
current executable configuration. Changing any material intent field invalidates
the previous result and requires a new attempt. Existing evidence remains
attached to the configuration revision that produced it.

Generated programs remain auditable and standalone. Sharing plan semantics must
not introduce a runtime dependency on a second sampling implementation.

## Custom sampling scripts

A tracked script may orchestrate a matrix of recipes, launch generated programs,
or analyze run artifacts. It may not contain its own warmup, kernel construction,
inference loop, or accepted-evidence sampling path.

When a custom sampling script exists:

1. identify the missing declarative capability;
2. add a typed recipe field or capability descriptor;
3. implement it in plan resolution and codegen;
4. convert the script's useful case into a recipe fixture and regression test;
5. reproduce the relevant behavior through generated code; and
6. delete the redundant sampling script.

Temporary exceptions must identify the missing capability and have an owner and
expiry. Results from an exception are diagnostic only and cannot update a
certified recipe.

Static architecture checks should reject new direct sampling routines outside
the codegen implementation, lower-level algorithm tests, and explicitly scoped
prototypes. The allowlist must shrink as migration proceeds.

## Ground truth and clone-first operation

Ground-truth draws, summaries, and chain statistics in Git LFS are intentional
repository assets. They are excluded from slimming targets.

Workspace validation must distinguish:

- hydrated, canonical Git LFS ground-truth artifacts;
- an unhydrated LFS pointer;
- a missing artifact; and
- a local, derived cache.

Evaluation fails closed with the exact missing path and hydration command when a
required canonical artifact is unavailable. It must not silently substitute a
cache or a different reference revision.

Every comparison records the exact reference protocol, shape, split, sample
count, seed information, and LFS object identifier or content hash. Existing LFS
objects are not regenerated merely to migrate the recipe schema.

## Failure evidence is a product feature

Tuningfork exists in part to document why plausible sampling routes did not
work. Negative paths are therefore first-class catalog content, not migration
debris.

The following information must never be discarded:

- failed, reviewed, errored, and superseded attempts;
- configurations and seeds attempted;
- failure diagnoses and interventions;
- metrics, gate margins, and automatic verdicts;
- review and override history;
- timing, environment, software, and data provenance;
- long-form workflow notes and procedural lessons;
- sidecar and sample references; and
- unknown legacy annotations.

Loaders must not silently filter unknown fields. Until a field has a typed home,
it must survive in a lossless extension/legacy section with its original key,
value, and source location.

No refactor may land on the strength of PASS-path fixtures alone. Removing code
is allowed only after its procedural knowledge is represented by recipe intent,
retained attempt evidence, and a regression test.

## Conformance gates

### Recipe and migration gates

- Every committed recipe loads and writes without losing a JSON leaf.
- Unknown-field sentinel data survives load, update, and save.
- Attempt count, order, intent snapshot, verdict, diagnosis, notes, provenance,
  and artifact references survive migration.
- Legacy MCMC, SMC, and ground-truth recipes retain their original semantics and
  original labels as provenance.
- Serialize/load/serialize is idempotent in the new schema.
- Migration emits a machine-readable mapping from every legacy JSON pointer to
  its new location; the dropped-field set must be empty.
- Git LFS object identifiers and content hashes are unchanged.

### Recipe-to-code gates

For every supported sampler family, warmup family, and special capability:

- emission succeeds and the source compiles;
- every material recipe field appears in the normalized plan or manifest;
- a tiny generated run produces expected shapes and diagnostic fields;
- the receipt matches the recipe and generated source;
- mutating a material field changes the plan or program hash; and
- unsupported combinations fail explicitly at generation time.

During migration, a small differential test may compare the old runner with the
generated program on stable observables: resolved identities and parameters,
shapes, diagnostic fields, and gate inputs. Bit-for-bit draws are required only
when random-key choreography is intentionally identical. The old runner is
removed as an evidence path once parity is established.

These checks establish semantic fidelity. Statistical quality remains the job of
the diagnostic and ground-truth gates; a tiny codegen smoke test is not a
convergence claim.

### Architecture gates

- Catalog evidence always has a matching generated-program receipt.
- Evaluators do not construct samplers.
- Orchestration and analysis scripts do not implement sampling loops.
- The codegen capability matrix has no silent fallback or unclassified entry.
- Any temporary direct-sampling exception is diagnostic-only, owned, and
  time-bounded.

## Adoption sequence

1. Inventory the current recipe corpus, including unknown fields, sidecars,
   failed attempts, and custom sampling paths. Freeze a losslessness baseline.
2. Introduce versioned recipe sections and compatibility readers that preserve
   all legacy information before changing writers.
3. Add the typed execution plan, embedded manifest, and execution receipt.
4. Add recipe-to-code coverage, mutation, receipt, and small-run conformance
   tests across the registered capability matrix.
5. Route calibration, certification, catalog re-runs, revalidation, and
   benchmarks through generated programs.
6. Migrate each custom sampling script capability-by-capability, retain a
   declarative regression case, and remove the script.
7. Split large runner and emitter modules along the lifecycle boundaries above,
   consolidating only behavior proven to be shared.
8. Migrate the committed corpus and remove compatibility code only after the
   zero-information-loss and statistical gates pass.

Packaging and wheel-content work are deliberately outside this sequence.

## Definition of done

The refactor is complete when:

- a recipe fully explains its intent, every attempted route, the generated
  routine that ran, the resulting evidence, its PASS/FAIL stamp, and its effort
  title;
- all admissible sampling evidence comes from codegen;
- recipe intent and generated behavior are checked across the full registered
  capability matrix;
- no negative-path or legacy recipe information was dropped;
- custom sampling scripts have become declarative recipe capabilities and
  regression tests;
- ground-truth Git LFS artifacts remain intact and explicitly bound to
  evaluations; and
- the resulting implementation consists of small lifecycle components rather
  than parallel runners or a new codegen monolith.
