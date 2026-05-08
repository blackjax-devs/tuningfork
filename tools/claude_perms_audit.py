#!/usr/bin/env python3
"""Audit `~/blackjax-devs/.claude/settings.json` against shell commands the
codebase invokes via subprocess. Prints recommended `permissions.allow`
additions for any pattern missing.

Usage:  uv run python tools/claude_perms_audit.py
        make claude-perms-audit

Implements [META-005] from AGENT_WORKFLOW.md § Cumulative retrospective.

Heuristic: greps `subprocess.run([...])` and `subprocess.check_output([...])`
calls in bjx_bench/ and tests/, extracts the first 1-2 tokens of each command
list, and cross-references with the current `permissions.allow` list. Reports
patterns that *would not match* any existing allow rule under Claude Code's
glob semantics (where `*` matches any string of non-newline chars).

Limitations:
- Does not handle env-var-prefixed commands (those need explicit patterns
  per the Phase 2 finding).
- Ignores commands inside multi-line strings.
- May false-positive on dynamic command construction (e.g., `["git", verb]`
  where `verb` is a runtime variable).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = Path.home() / "blackjax-devs/.claude/settings.json"


def load_allow_list() -> list[str]:
    if not SETTINGS.exists():
        print(f"WARN: settings.json not found at {SETTINGS}")
        return []
    data = json.loads(SETTINGS.read_text())
    return data.get("permissions", {}).get("allow", [])


def find_subprocess_calls(root: Path) -> list[tuple[Path, int, list[str]]]:
    """Return list of (file, line, [argv tokens]) for every subprocess.run/check_output
    found in *.py under root. Naive — only handles list-literal first-arg form."""
    results: list[tuple[Path, int, list[str]]] = []
    pattern = re.compile(
        r"subprocess\.(?:run|check_output|check_call|Popen)\(\s*\[\s*"
        r'((?:"[^"]+"|\'[^\']+\'|[\w\.]+)(?:\s*,\s*(?:"[^"]+"|\'[^\']+\'|[\w\.]+))*)',
        re.MULTILINE,
    )
    for py in root.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in py.parts):
            continue
        try:
            text = py.read_text()
        except UnicodeDecodeError:
            continue
        for m in pattern.finditer(text):
            tokens_raw = m.group(1)
            # Extract string literals (quoted) only; skip variable refs (foo, bar).
            tok_re = re.compile(r'"([^"]+)"|\'([^\']+)\'')
            tokens = [a or b for a, b in tok_re.findall(tokens_raw)]
            if not tokens:
                continue
            line = text[: m.start()].count("\n") + 1
            results.append((py.relative_to(ROOT), line, tokens))
    return results


def matches_allow(tokens: list[str], allow_patterns: list[str]) -> bool:
    """Test whether the command [tokens...] is matched by any Bash() pattern.

    Approximation of Claude Code's glob: `*` matches any chars (non-newline);
    explicit literals must match. We reconstruct the command as a space-joined
    string and test against `re.escape(pattern).replace(r'\\*', r'.*')`.
    """
    cmd = " ".join(tokens)
    for raw in allow_patterns:
        # Strip "Bash(...)" wrapper
        m = re.match(r"^Bash\((.+)\)$", raw)
        if not m:
            continue
        pat = m.group(1)
        # Translate glob to regex
        re_pat = re.escape(pat).replace(r"\*", r".*")
        if re.fullmatch(re_pat, cmd):
            return True
    return False


def first_two_tokens_pattern(tokens: list[str]) -> str:
    """Generate a recommended Bash() pattern from a command's first 1-2 tokens."""
    if len(tokens) >= 2:
        return f"Bash({tokens[0]} {tokens[1]} *)"
    return f"Bash({tokens[0]} *)"


def main() -> int:
    allow = load_allow_list()
    print(f"Loaded {len(allow)} allow patterns from {SETTINGS}\n")

    calls = find_subprocess_calls(ROOT)
    print(f"Found {len(calls)} subprocess.* calls with literal commands.\n")

    missing: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for fp, line, tokens in calls:
        if not matches_allow(tokens, allow):
            missing[first_two_tokens_pattern(tokens)].append((fp, line))

    if not missing:
        print("✅ All subprocess command patterns are covered by the allow list.")
        return 0

    print(f"⚠ {len(missing)} command pattern(s) MISSING from allow list:\n")
    for pat, locs in sorted(missing.items()):
        print(f"  {pat}")
        for fp, line in locs[:3]:
            print(f"      ↳ {fp}:{line}")
        if len(locs) > 3:
            print(f"      ↳ … and {len(locs) - 3} more")
        print()

    print("To fix, add the missing patterns to permissions.allow in")
    print(f"  {SETTINGS}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
