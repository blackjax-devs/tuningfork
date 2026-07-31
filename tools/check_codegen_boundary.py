"""Static guard for direct sampling calls in production code."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

Hit = tuple[str, str, str]

# Constructors which return a BlackJAX sampling algorithm.  Keep this list
# deliberately small and explicit: warmup/adaptation helpers are not sampling
# paths, while an omitted constructor would let a custom scan bypass codegen.
_SAMPLING_APIS = frozenset(
    {
        "nuts",
        "hmc",
        "mhmc",
        "rmhmc",
        "dynamic_hmc",
        "dmhmc",
        "ghmc",
        "barker",
        "mala",
        "rmh",
        "irmh",
        "additive_step_random_walk",
        "mclmc",
        "adjusted_mclmc",
        "adjusted_mclmc_dynamic",
        "orbital_hmc",
        "elliptical_slice",
        "mgrad_gaussian",
        "laplace_hmc",
        "laplace_dhmc",
        "laplace_mhmc",
        "laplace_dmhmc",
        "adaptive_tempered_smc",
        "tempered_smc",
        "partial_posteriors_smc",
        "persistent_sampling_smc",
        "adaptive_persistent_sampling_smc",
    }
)
_REFERENCE_SCOPES = frozenset(
    {
        ("calibration/certify_reference.py", "certify_reference_nuts"),
        ("groundtruth/_nuts_multichain.py", "_run_nuts_multichain.one_sample"),
    }
)


def _qualifier(stack: list[str]) -> str:
    return ".".join(stack) if stack else "<module>"


class _Scanner(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.hits: Counter[Hit] = Counter()
        self._scope: list[str] = []
        self._aliases: dict[str, str] = {"run_smc": "run_smc"}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if node.module == "blackjax" and alias.name in _SAMPLING_APIS:
                name = alias.asname or alias.name
                self._aliases[name] = f"blackjax.{alias.name}"
            if alias.name == "run_inference_algorithm":
                name = alias.asname or alias.name
                self._aliases[name] = "run_inference_algorithm"
            if alias.name == "run_smc":
                name = alias.asname or alias.name
                self._aliases[name] = "run_smc"
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "run_inference_algorithm" or alias.name.endswith(
                ".run_inference_algorithm"
            ):
                name = alias.asname or alias.name.rsplit(".", 1)[-1]
                self._aliases[name] = "run_inference_algorithm"
            if alias.name == "run_smc" or alias.name.endswith(".run_smc"):
                name = alias.asname or alias.name.rsplit(".", 1)[-1]
                self._aliases[name] = "run_smc"
            if alias.name == "blackjax":
                self._aliases[alias.asname or "blackjax"] = "blackjax"
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
            parent = self._primitive_for_expr(expr.value)
            if parent is not None:
                dotted = f"{parent}.{expr.attr}"
                if parent == "blackjax" and expr.attr in _SAMPLING_APIS:
                    return dotted
                if expr.attr in {
                    "run_inference_algorithm",
                    "run_smc",
                    "runner",
                    "factory",
                }:
                    return expr.attr
                return None
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
        is_sampling_constructor = primitive is not None and (
            primitive.startswith("blackjax.")
            and primitive.rsplit(".", 1)[-1] in _SAMPLING_APIS
        )
        if primitive is not None and not (
            is_sampling_constructor
            and (self.relative_path, _qualifier(self._scope)) in _REFERENCE_SCOPES
        ):
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


# Generated recipe programs are the sole ordinary sampling route.  The canonical
# ground-truth reference paths are the only direct-call exceptions.
ALLOWED_DIRECT_CALLS: Counter[Hit] = Counter(
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
    }
)


def baseline() -> Counter[Hit]:
    return ALLOWED_DIRECT_CALLS


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
            "no alternate recipe-sampling debt remains",
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
