# ExitPlanMode: Phase-Naming Audit for Open-Source Readiness

**Scope**: Strip internal project phase-naming from shipped user-facing surfaces (docs, docstrings, recipe JSON metadata) and replace with external/plain-language names per the `CLAUDE.md` internal→external map.

**Deliverable**: Full inventory of phase refs by file/context, replacement mapping, and execution plan (split between @tech-writer and @swe).

---

## Internal → External Name Map (from tuningfork/CLAUDE.md)

| Internal | External | Context |
|---|---|---|
| Phase 5 | Inventory close-out | Completed 2026-05-10 |
| Recipe Generation Phase | recipe generation | Active phase name |
| Recipe Phase 1 | the window-adaptation × HMC-family sweep | Warmup sweep |
| Recipe Phase 2 | the MCLMC recipe sweep | MCLMC sampler sweep |
| Recipe Phase 3 | the SMC recipe sweep | SMC sampler sweep |
| Recipe Phase 4 | the pathfinder warmup sweep | Warmup sweep |
| Phase 0 | groundtruth certification / reference certification | Cert phase |
| Phase 3 (recipe-emit) | the LOW recipe sweep | LOW effort tier |
| Phase 4 (recipe-emit) | the MEDIUM recipe sweep | MEDIUM effort tier |
| Phase A (step-policy) | Schema wiring (PR #39) | Dev-only, EXCLUDE |
| Phase B (step-policy) | step_policy harvest work | Dev-only, EXCLUDE |
| Phase B-2 (step-policy) | Schema extension for warmups list / warmup_inner_kernel (PR #54) | Dev-only, EXCLUDE |
| Phase C+ (step-policy) | Deferred follow-up work | Dev-only, EXCLUDE |
| "empirical oracle" / "V7 oracle" | "NUTS-harvested step_policy" | When mentioned in user-facing context |
| `harvest_oracle_spec` / `harvest_oracle_spec_from_array` | `harvest_step_policy_from_chain_stats` / `harvest_step_policy_from_nis` | Function names (dev-only code) |
| R3.5 / R3.5b / R3.5-MVP | Version codenames; remove or replace with commit SHA | User-facing context |
| P5.x (e.g., P5.12, P5.15.5) | Specific checkpoint refs; remove for user-facing, keep internal only | Dev-only context |

---

## Inventory: User-Facing Phase References

### Tier 1 — High Priority (Actively Used, Visible to External Users)

| File | Content | Current Phrasing | Replacement | Priority |
|---|---|---|---|---|
| `catalog/README.md` | General intro + model status tables | None found | N/A | — |
| `catalog/notebooks/inspect_README.md` | API guide, design rationale | "R3.5-MVP follow-up clarification, 2026-05-17", "R3.5b expands to the full 10 warmups × 24 samplers" | Remove version codenames; replace with "As of tuningfork v1" or specific feature names | HIGH |
| `catalog/notebooks/recipe_diagnostics.md` | NUTS/HMC diagnostics notebook | Line 23: "the deferred design in `worklog/threads/notebook-arviz-redesign.md`" | Update to point to new MCLMC notebook | HIGH |
| `RECIPE_GENERATION.md` | Effort matrix + model groups (USER-FACING) | Lines 4, 26, 141, 156: "P5", "P5.12 VI kickoff", "P5.12", "Recipe Phases 2+" | Replace P5.x with "during groundtruth certification" or remove entirely; replace "Recipe Phases 2+" with "the MCLMC / SMC recipe sweeps" | HIGH |
| `CONTRIBUTING.md` | Dev guide + roadmap | Line 154: "**Benchmark suite** (Phase 8 v1)" | Replace with "**Benchmark suite** (the initial benchmark release)" or simply "**Benchmark suite**" | HIGH |
| `catalog/RECIPE_SCHEMA.md` | Recipe schema documentation + examples | Lines 294, 312: "V7 (NUTS-harvested step_policy work)", "policy_v7-empirical-oracle" | Replace "V7" with "NUTS-harvested step_policy" (already done in map); keep policy_v7 as literal field name in examples (it's part of old JSON structure) | MEDIUM |

### Tier 2 — Medium Priority (Generator Code that May Emit User-Facing Metadata)

| File | Content | Current Phrasing | Risk | Plan |
|---|---|---|---|---|
| `tuningfork/recipes/_instructions.py` | Recipe `notes`/`instructions` field generation | Unknown (must audit) | Code emits phase-named metadata into recipe JSONs | @swe: audit, clean generator so future emits are phase-name-free |
| `tuningfork/recipes/_templates/samplers/*.py.tmpl` | Sampler templates that emit scripts | Unknown (must audit) | Code comments/docstrings may reference internal phases | @swe: clean comments (don't change logic) |
| `tuningfork/recipes/_templates/warmups/*.py.tmpl` | Warmup templates | Unknown (must audit) | Same risk | @swe: clean comments |
| `tuningfork/cli.py` | CLI docstrings + help text | Unknown (must audit) | CLI is user-facing | @swe: audit if shipped publicly |

### Tier 3 — Low Priority (Dev-Only, Exclude from Cleaning)

| File | Keep As? |
|---|---|
| `CLAUDE.md` | **YES** — internal dev ref, holds the authoritative map; keep unchanged |
| `worklog/` tree | **YES** — internal coordination; keep unchanged |
| `tests/` | **YES** — internal testing; keep unchanged |
| `experiments/` | **FLAG** — Check if shipped in open-source distribution. If yes → clean or exclude. If no → keep as-is (internal scratch). |

---

## Execution Plan (Coordination between @tech-writer and @swe)

### @tech-writer (Docs & Copy)

**Deliverable**: Edited user-facing docs with phase names replaced per the map above.

**Files to edit**:
1. `catalog/notebooks/inspect_README.md` — Remove R3.5/R3.5-MVP version codenames (2 occurrences, lines 195+205)
2. `RECIPE_GENERATION.md` — Replace P5.x refs with plain language (≈5 occurrences)
3. `CONTRIBUTING.md` — Replace "Phase 8" with "initial benchmark release"
4. `catalog/RECIPE_SCHEMA.md` — Replace "V7" with "NUTS-harvested step_policy" (keep policy_v7 field names as literal examples)
5. Audit `catalog/README.md` — spot-check for any missed refs

**Flagged for decision**:
- `recipe_diagnostics.md` line 23 update: should the reference be to "the MCLMC diagnostics notebook" (once written) or to the broader `notebook-arviz-redesign` worklog thread? → **Decision**: point to new MCLMC notebook once it exists.

### @swe (Code & Generator)

**Deliverable**: Generator code cleaned so future recipe emits are phase-name-free; `benchmark.yml` updated per the round-3 CI fix PR.

**Files to audit/clean**:
1. `benchmark.yml` (CI workflow) — Strip "Phase 8" from workflow name/comments. **Fold into round-3 CI-fix PR.**
2. `tuningfork/recipes/_instructions.py` — Audit recipe `notes`/`instructions` field generation for phase-named boilerplate. Clean any emitted strings.
3. `tuningfork/recipes/_templates/samplers/*.py.tmpl` — Check code comments; clean if present (but don't change function signatures/logic).
4. `tuningfork/recipes/_templates/warmups/*.py.tmpl` — Same as above.
5. `tuningfork/cli.py` — Audit docstrings/help text if this is public-facing.

**Note**: The goal is **future-proofing** — existing recipe JSONs may have phase-named `instructions` fields (that's OK, they're historical artifacts). The generator code should not create NEW phase refs when emitting future recipes.

### @statistician (On-Call)

**Role**: Review any `instructions`/`notes` fields that @swe flags as needing a correctness-preserving reword. Ensure technical accuracy is preserved when replacing internal phase terminology with external plain language.

---

## Decision Point: `experiments/` Directory

**Question**: Is `experiments/` shipped in the open-source distribution, or is it internal scratch (ignored on release)?

**If shipped**: Must clean phase-named scripts (`expJ_phase1_*, expJ_phase2_*, etc.`) or exclude from the distribution package.

**If internal-only**: No action needed; phase names can stay.

→ **Flag for @user/@tl**: Does `experiments/` go into the public distribution?

---

## Ratification Gate

**Before execution**: @tl reviews this inventory + plan, confirms:
1. ✓ All high-priority files identified
2. ✓ Replacement mappings match the CLAUDE.md internal→external table
3. ✓ Split of work (@tech-writer docs, @swe code/generator) is clear
4. ✓ `experiments/` distribution status resolved

**On ratification**: @tech-writer + @swe proceed in parallel (docs edits, generator cleanup, CI fix).

---

## Success Criteria

- [ ] No internal phase names (Phase N, Recipe Phase N, R3.5, P5.x, V7 in contexts where they refer to phases) visible in user-facing docs/notebooks
- [ ] Generator code (`_instructions.py`, templates) produces future recipes without phase-named metadata
- [ ] Replacement phrasing is plain-language and clear to external readers
- [ ] `benchmark.yml` CI workflow cleaned (folded into round-3 fix PR)
- [ ] `experiments/` decision resolved (shipped or excluded)
