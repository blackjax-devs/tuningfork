"""Static guard for direct sampling calls in production code.

The two baseline sets below are deliberately explicit.  The lower-level set is
the small capability layer generated programs may build on; the transitional
set is caller debt that must shrink as those callers move to generated programs.
Both sets are exact, multiplicity-preserving counters (not line-number lists).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

Hit = tuple[str, str, str]


def _qualifier(stack: list[str]) -> str:
    return ".".join(stack) if stack else "<module>"


class _Scanner(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.hits: Counter[Hit] = Counter()
        self._scope: list[str] = []
        self._run_names: set[str] = set()
        self._smc_names: set[str] = {"run_smc"}
        self._aliases: dict[str, str] = {"run_smc": "run_smc"}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "run_inference_algorithm":
                name = alias.asname or alias.name
                self._run_names.add(name)
                self._aliases[name] = "run_inference_algorithm"
            if alias.name == "run_smc":
                name = alias.asname or alias.name
                self._smc_names.add(name)
                self._aliases[name] = "run_smc"
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "run_inference_algorithm" or alias.name.endswith(
                ".run_inference_algorithm"
            ):
                name = alias.asname or alias.name.rsplit(".", 1)[-1]
                self._run_names.add(name)
                self._aliases[name] = "run_inference_algorithm"
            if alias.name == "run_smc" or alias.name.endswith(".run_smc"):
                name = alias.asname or alias.name.rsplit(".", 1)[-1]
                self._smc_names.add(name)
                self._aliases[name] = "run_smc"
        self.generic_visit(node)

    def _visit_scope(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def _primitive_for_expr(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return self._aliases.get(expr.id)
        if isinstance(expr, ast.Attribute):
            if expr.attr in {
                "run_inference_algorithm",
                "run_smc",
                "runner",
                "factory",
            }:
                return expr.attr
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        primitive = self._primitive_for_expr(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if primitive is None:
                    self._aliases.pop(target.id, None)
                else:
                    self._aliases[target.id] = primitive

    def visit_Call(self, node: ast.Call) -> None:
        primitive = self._primitive_for_expr(node.func)
        if primitive is not None:
            self.hits[(self.relative_path, _qualifier(self._scope), primitive)] += 1
        self.generic_visit(node)


def scan_source(root: Path) -> Counter[Hit]:
    """Scan production Python below *root*, excluding tests/catalog/notebooks."""
    root = Path(root)
    hits: Counter[Hit] = Counter()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(
            part in {"tests", "catalog", "notebooks"}
            for part in path.relative_to(root).parts
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        scanner = _Scanner(rel)
        scanner.visit(tree)
        hits.update(scanner.hits)
    return hits


# Filled from the checked-in production tree.  Keep these as literals so a
# removed call is visible and fails until the debt is intentionally retired.
LOWER_LEVEL_BASELINE: Counter[Hit] = Counter(
    {
        ("warmup/_vi_warmup_runner.py", "_vi_warmup_runner", "factory"): 2,
        ("warmup/_vi_warmup_runner.py", "_vi_warmup_runner._sa_one_step", "factory"): 1,
        ("warmup/mclmc_tuning.py", "_runner", "factory"): 1,
        ("warmup/no_warmup.py", "_runner", "factory"): 1,
    }
)

TRANSITIONAL_DEBT_BASELINE: Counter[Hit] = Counter(
    {
        (
            "calibration/certify_reference.py",
            "certify_reference_nuts",
            "run_inference_algorithm",
        ): 1,
        (
            "groundtruth/_nuts_multichain.py",
            "_run_nuts_multichain.one_sample",
            "run_inference_algorithm",
        ): 1,
        ("calibration/tune.py", "_run_trial", "factory"): 1,
        ("calibration/tune.py", "_run_trial", "run_inference_algorithm"): 1,
        ("calibration/tune.py", "_run_warmup", "runner"): 1,
        ("recipes/_base.py", "Recipe.from_warmup_only", "runner"): 1,
        ("recipes/_smc_runner.py", "emit_smc_recipe_for_cell", "factory"): 2,
        ("recipes/_smc_runner.py", "emit_smc_recipe_for_cell", "run_smc"): 1,
        ("recipes/emit_mclmc_lrd.py", "_run_cert_seed", "runner"): 1,
        (
            "recipes/sequential_run_recipe_pipeline.py",
            "_warmup_wge_worker",
            "runner",
        ): 3,
    }
)


def baseline() -> Counter[Hit]:
    return LOWER_LEVEL_BASELINE + TRANSITIONAL_DEBT_BASELINE


def report(root: Path) -> dict[str, Counter[Hit]]:
    """Return actual, expected, and their signed differences for tooling/tests."""
    actual = scan_source(root)
    expected = baseline()
    return {
        "actual": actual,
        "expected": expected,
        "additions": actual - expected,
        "removals": expected - actual,
    }


def check(root: Path) -> tuple[bool, str]:
    result = report(root)
    additions = result["additions"]
    removals = result["removals"]
    if not additions and not removals:
        return (
            True,
            "matches the recorded direct-sampling baseline; transitional debt remains "
            "non-admissible and must shrink",
        )
    lines = ["codegen boundary mismatch"]
    if additions:
        lines.append("additions:")
        lines.extend(f"  {count}x {hit}" for hit, count in sorted(additions.items()))
    if removals:
        lines.append("removals (shrink the baseline deliberately):")
        lines.extend(f"  {count}x {hit}" for hit, count in sorted(removals.items()))
    return False, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path(__file__).parents[1] / "tuningfork"
    )
    args = parser.parse_args(argv)
    ok, message = check(args.root)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
