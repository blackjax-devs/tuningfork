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


def _is_low_level_symbol(symbol: str) -> bool:
    """Return whether *symbol* names BlackJAX's internal sampler plumbing."""
    parts = symbol.split(".")
    if len(parts) >= 4 and parts[:2] in (["blackjax", "mcmc"], ["blackjax", "smc"]):
        return parts[-1].startswith("build_") or parts[-1] == "as_top_level_api"
    return (
        len(parts) == 3
        and parts[0] == "blackjax"
        and parts[1] in _SAMPLING_APIS
        and (parts[-1].startswith("build_") or parts[-1] == "as_top_level_api")
    )


def _is_blackjax_namespace(symbol: str) -> bool:
    """Return whether *symbol* can lead to internal BlackJAX sampler plumbing."""
    return symbol in {"blackjax.mcmc", "blackjax.smc"} or symbol.startswith(
        ("blackjax.mcmc.", "blackjax.smc.")
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
            if node.module == "importlib" and alias.name == "import_module":
                self._aliases[alias.asname or alias.name] = "importlib.import_module"
            if node.module == "blackjax" and alias.name in _SAMPLING_APIS:
                name = alias.asname or alias.name
                self._aliases[name] = f"blackjax.{alias.name}"
            if node.module in {"blackjax.mcmc", "blackjax.smc"}:
                name = alias.asname or alias.name
                self._aliases[name] = f"{node.module}.{alias.name}"
            elif node.module and node.module.startswith("blackjax.mcmc."):
                symbol = f"{node.module}.{alias.name}"
                if _is_low_level_symbol(symbol):
                    name = alias.asname or alias.name
                    self._aliases[name] = symbol
            elif node.module and node.module.startswith("blackjax.smc."):
                symbol = f"{node.module}.{alias.name}"
                if _is_low_level_symbol(symbol):
                    name = alias.asname or alias.name
                    self._aliases[name] = symbol
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
            if alias.name.startswith(("blackjax.mcmc.", "blackjax.smc.")):
                if alias.asname:
                    self._aliases[alias.asname] = alias.name
                else:
                    # ``import a.b.c`` binds ``a``, not ``c``.
                    self._aliases["blackjax"] = "blackjax"
            if alias.name == "importlib":
                self._aliases[alias.asname or "importlib"] = "importlib"
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
                if _is_low_level_symbol(dotted):
                    return dotted
                if _is_blackjax_namespace(dotted):
                    return dotted
                if dotted == "importlib.import_module":
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
        if isinstance(expr, ast.Call):
            if (
                isinstance(expr.func, ast.Name)
                and expr.func.id == "getattr"
                and len(expr.args) == 2
                and isinstance(expr.args[1], ast.Constant)
                and isinstance(expr.args[1].value, str)
            ):
                parent = self._primitive_for_expr(expr.args[0])
                if parent is not None:
                    dotted = f"{parent}.{expr.args[1].value}"
                    if (
                        (parent == "blackjax" and expr.args[1].value in _SAMPLING_APIS)
                        or _is_low_level_symbol(dotted)
                        or _is_blackjax_namespace(dotted)
                    ):
                        return dotted
            importer = self._primitive_for_expr(expr.func)
            if (
                importer == "importlib.import_module"
                and len(expr.args) == 1
                and isinstance(expr.args[0], ast.Constant)
                and isinstance(expr.args[0].value, str)
            ):
                module = expr.args[0].value
                if module == "blackjax" or _is_blackjax_namespace(module):
                    return module
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
        if primitive is None:
            self.generic_visit(node)
            return
        is_sampling_constructor = primitive is not None and (
            primitive.startswith("blackjax.")
            and primitive.rsplit(".", 1)[-1] in _SAMPLING_APIS
        )
        is_low_level_builder = _is_low_level_symbol(primitive)
        if (
            is_sampling_constructor
            or is_low_level_builder
            or primitive
            in {
                "run_inference_algorithm",
                "run_smc",
                "runner",
                "factory",
            }
        ) and not (
            is_sampling_constructor
            and primitive == "blackjax.nuts"
            and (self.relative_path, _qualifier(self._scope)) in _REFERENCE_SCOPES
        ):
            self.hits[(self.relative_path, _qualifier(self._scope), primitive)] += 1
        self.generic_visit(node)


def scan_source(root: Path) -> Counter[Hit]:
    """Scan authored production Python below *root*."""
    root = Path(root)
    hits: Counter[Hit] = Counter()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or "_cache" in relative.parts:
            continue
        rel = relative.as_posix()
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
