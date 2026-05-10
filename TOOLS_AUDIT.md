# `tools/` location audit

Per-file evaluation of whether each script under `bjx-bench/tools/` belongs in
this repo, in `bjx_bench/data/scripts/` (in-repo, with the package), or in a
different repo (project-level / `claude-config/`). **No moves performed** —
this is a recommendation only.

## Per-file recommendation

| File | Current location | Recommended location | Rationale |
|---|---|---|---|
| `claude_perms_audit.py` | `bjx-bench/tools/` | **`claude-config/tools/`** (parent-level, not bjx-bench) | Audits `~/.claude/settings.json` against shell commands invoked across the codebase. Has nothing to do with bjx-bench specifically; serves the entire monorepo's Claude Code permissions setup. |
| `clean_orphans.sh` | `bjx-bench/tools/` | **`bjx-bench/tools/` (stay)** | Referenced from `bjx-bench/Makefile` (`make clean-orphans`) and from the agent-mandatory pre-test discipline in `bjx-bench/CLAUDE.md`. Kills orphan Python REPLs; could be project-wide in principle, but the agent rule is bjx-bench-scoped (heavy MCMC sweeps OOM on it). Keep here. |
| `fetch_irt_2pl.py` | `bjx-bench/tools/` | **`bjx_bench/data/scripts/`** (in-repo, with the package) | Posteriordb data fetcher specific to the IRT-2PL model (`bjx_bench/model/hierarchical/irt_2pl.py`). Belongs alongside the package's data-prep code. |
| `fetch_radon.py` | `bjx-bench/tools/` | **`bjx_bench/data/scripts/`** | Posteriordb data fetcher for the radon model. Same rationale as `fetch_irt_2pl.py`. |
| `generate_gp_regression.py` | `bjx-bench/tools/` | **`bjx_bench/data/scripts/`** | Synthetic-data generator for the `gp_regression` model. Bjx-bench-specific data prep. |
| `generate_lotka_volterra.py` | `bjx-bench/tools/` | **`bjx_bench/data/scripts/`** | Synthetic-data generator for `lotka_volterra`. Same rationale. |
| `generate_stoch_vol.py` | `bjx-bench/tools/` | **`bjx_bench/data/scripts/`** | Synthetic-data generator for `stoch_vol`. Same rationale. |

## Suggested layout after moves

```
bjx-bench/
├── tools/                           # repo-orchestration scripts only
│   └── clean_orphans.sh             # referenced from Makefile + CLAUDE.md
├── bjx_bench/
│   └── data/
│       ├── scripts/                 # data prep — moved here
│       │   ├── fetch_irt_2pl.py
│       │   ├── fetch_radon.py
│       │   ├── generate_gp_regression.py
│       │   ├── generate_lotka_volterra.py
│       │   └── generate_stoch_vol.py
│       └── ...                      # raw datasets (existing)
└── ...

claude-config/                       # parent-dir cross-repo tooling
└── tools/
    └── claude_perms_audit.py        # moved here from bjx-bench/tools/
```

## Open considerations before moving

1. **Makefile reference**: `bjx-bench/Makefile` calls `uv run python tools/claude_perms_audit.py` for a `make claude-perms-audit` target. Moving the script to `claude-config/tools/` requires either deleting that Makefile target (best — bjx-bench shouldn't audit Claude Code config) or repointing it across-repo (path-fragile).

2. **`bjx_bench/data/scripts/` doesn't exist yet**. Creating it requires an `__init__.py` (or leaving it as a non-package script dir) and updating the package metadata in `pyproject.toml` if these scripts need to be packaged with the wheel.

3. **Agent allow-list**: `bash tools/clean_orphans.sh` is currently pre-approved in `.claude/settings.local.json`. Keeping `clean_orphans.sh` in `bjx-bench/tools/` preserves that allow-list path.

4. **Cross-repo move (`claude_perms_audit.py`)**: requires user approval (the original brief said "Don't actually move files. Cross-repo moves need user approval").
