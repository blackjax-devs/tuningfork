#!/usr/bin/env bash
# Guard: fail if any tracked file references internal agent-config paths or tags.
# This script itself is excluded from the check via the pathspec below.
set -euo pipefail

# The pattern strings live here only as grep arguments — this file is excluded.
MATCH_WORKLOG='worklog/'
MATCH_CONFIG='claude-config'
MATCH_TAGS='@statistician|@swe\b|@tech-writer|memoires'
# Agent-identity labels that must not reach shipped artifacts (catalog docs,
# recipes).  Specifically: "SWE agent" in prose; branch-named statistician_id
# values (swe-<branch>-YYYY-MM-DD).  The convention is stat-YYYY-MM-DD or
# tl-YYYY-MM-DD; anything prefixed swe- is an implementer label, not a
# reviewer label, and must not be committed to the catalog.
MATCH_AGENT_LABELS='SWE[- ]agent\b|"statistician_id"[[:space:]]*:[[:space:]]*"swe-'

PATTERNS="${MATCH_WORKLOG}|${MATCH_CONFIG}|${MATCH_TAGS}|${MATCH_AGENT_LABELS}"

# Run grep across all tracked text files; exclude this script and the pre-commit config.
HITS=$(git grep -nE "$PATTERNS" \
    -- . \
    ':(exclude).pre-commit-config.yaml' \
    ':(exclude)tools/check_no_internal_refs.sh' \
    ':(exclude)uv.lock' \
    2>/dev/null | grep -v "^Binary file" || true)

if [ -n "$HITS" ]; then
    echo "ERROR: internal references found — remove before committing:"
    echo "$HITS"
    exit 1
fi
