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

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_TOOL = Path(__file__).parents[2] / "tools" / "check_codegen_boundary.py"
_SPEC = importlib.util.spec_from_file_location("check_codegen_boundary", _TOOL)
assert _SPEC and _SPEC.loader
boundary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(boundary)

_PRODUCTION = Path(__file__).parents[2] / "tuningfork"


def test_repository_matches_exact_baseline() -> None:
    assert boundary.scan_source(_PRODUCTION) == boundary.baseline()
    assert boundary.check(_PRODUCTION) == (
        True,
        "no alternate recipe-sampling debt remains",
    )


def test_new_call_in_temp_source_is_reported(tmp_path: Path) -> None:
    (tmp_path / "new.py").write_text(
        "from blackjax.util import run_inference_algorithm as ria\n"
        "def generated_gap():\n"
        "    return ria(kernel, state, 1)\n"
    )
    ok, message = boundary.check(tmp_path)
    assert not ok
    assert "new.py" in message
    assert "run_inference_algorithm" in message


def test_qualified_run_inference_algorithm_is_reported(tmp_path: Path) -> None:
    (tmp_path / "qualified.py").write_text(
        "import blackjax\n"
        "def sample():\n"
        "    return blackjax.util.run_inference_algorithm(kernel, state, 1)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("qualified.py", "sample", "run_inference_algorithm")] == 1


def test_aliased_run_smc_is_reported(tmp_path: Path) -> None:
    (tmp_path / "smc.py").write_text(
        "from another.module import run_smc as execute_smc\n"
        "def sample():\n"
        "    return execute_smc(kernel)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("smc.py", "sample", "run_smc")] == 1


def test_qualified_run_smc_is_reported(tmp_path: Path) -> None:
    (tmp_path / "qualified_smc.py").write_text(
        "import tuningfork.runner.smc as smc\n"
        "def sample():\n"
        "    return smc.run_smc(kernel)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("qualified_smc.py", "sample", "run_smc")] == 1


def test_direct_blackjax_constructor_is_reported(tmp_path: Path) -> None:
    (tmp_path / "custom.py").write_text(
        "import blackjax\n"
        "def sample():\n"
        "    kernel = blackjax.nuts(logdensity_fn)\n"
        "    return scan(kernel)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("custom.py", "sample", "blackjax.nuts")] == 1


def test_sampling_constructor_alias_is_reported(tmp_path: Path) -> None:
    (tmp_path / "aliased.py").write_text(
        "from blackjax import hmc as build_sampler\n"
        "def sample():\n"
        "    return build_sampler(logdensity_fn)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("aliased.py", "sample", "blackjax.hmc")] == 1


@pytest.mark.parametrize(
    ("name", "source", "primitive"),
    [
        (
            "from_module.py",
            "from blackjax.mcmc import hmc as internal_hmc\n"
            "internal_hmc.build_kernel()\n",
            "blackjax.mcmc.hmc.build_kernel",
        ),
        (
            "import_module.py",
            "import blackjax.mcmc.hmc as internal_hmc\n"
            "internal_hmc.build_kernel()\n",
            "blackjax.mcmc.hmc.build_kernel",
        ),
        (
            "import_module_unaliased.py",
            "import blackjax.mcmc.hmc\n" "blackjax.mcmc.hmc.build_kernel()\n",
            "blackjax.mcmc.hmc.build_kernel",
        ),
        (
            "import_smc_module_unaliased.py",
            "import blackjax.smc.adaptive_tempered_smc\n"
            "blackjax.smc.adaptive_tempered_smc.build_kernel()\n",
            "blackjax.smc.adaptive_tempered_smc.build_kernel",
        ),
        (
            "from_function.py",
            "from blackjax.mcmc.random_walk import build_rmh\n" "build_rmh()\n",
            "blackjax.mcmc.random_walk.build_rmh",
        ),
        (
            "top_level_module.py",
            "import blackjax\nblackjax.nuts.build_kernel()\n",
            "blackjax.nuts.build_kernel",
        ),
        (
            "top_level_api.py",
            "from blackjax.mcmc.hmc import as_top_level_api\n" "as_top_level_api()\n",
            "blackjax.mcmc.hmc.as_top_level_api",
        ),
        (
            "smc_builder.py",
            "from blackjax.smc.adaptive_tempered_smc import build_kernel\n"
            "build_kernel()\n",
            "blackjax.smc.adaptive_tempered_smc.build_kernel",
        ),
    ],
)
def test_low_level_sampling_builder_import_forms_are_reported(
    tmp_path: Path, name: str, source: str, primitive: str
) -> None:
    (tmp_path / name).write_text(source)
    hits = boundary.scan_source(tmp_path)
    assert hits[(name, "<module>", primitive)] == 1


@pytest.mark.parametrize(
    ("name", "source", "primitive"),
    [
        (
            "getattr_constructor.py",
            "import blackjax\ngetattr(blackjax, 'nuts')(logdensity_fn)\n",
            "blackjax.nuts",
        ),
        (
            "getattr_builder.py",
            "import blackjax.mcmc.hmc\n"
            "getattr(blackjax.mcmc.hmc, 'build_kernel')()\n",
            "blackjax.mcmc.hmc.build_kernel",
        ),
        (
            "dynamic_module.py",
            "import importlib\n"
            "importlib.import_module('blackjax.mcmc.hmc').build_kernel()\n",
            "blackjax.mcmc.hmc.build_kernel",
        ),
        (
            "aliased_dynamic_module.py",
            "from importlib import import_module as load\n"
            "load('blackjax.smc.adaptive_tempered_smc').build_kernel()\n",
            "blackjax.smc.adaptive_tempered_smc.build_kernel",
        ),
    ],
)
def test_constant_dynamic_sampling_forms_are_reported(
    tmp_path: Path, name: str, source: str, primitive: str
) -> None:
    (tmp_path / name).write_text(source)
    hits = boundary.scan_source(tmp_path)
    assert hits[(name, "<module>", primitive)] == 1


def test_unrelated_build_function_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text(
        "from project.helpers import build_sampler\n" "build_sampler(logdensity_fn)\n"
    )
    assert not boundary.scan_source(tmp_path)


def test_emitted_string_is_ignored_and_reference_scope_is_exempt(
    tmp_path: Path,
) -> None:
    emit = tmp_path / "recipes"
    emit.mkdir()
    (emit / "_emit_script.py").write_text("source = 'blackjax.nuts(logdensity_fn)'\n")
    (emit / "bad.py").write_text("import blackjax\nblackjax.nuts(logdensity_fn)\n")
    reference = tmp_path / "calibration"
    reference.mkdir()
    (reference / "certify_reference.py").write_text(
        "import blackjax\n"
        "def certify_reference_nuts():\n"
        "    blackjax.hmc(logdensity_fn)\n"
        "    return blackjax.nuts(logdensity_fn)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits == {
        (
            "calibration/certify_reference.py",
            "certify_reference_nuts",
            "blackjax.hmc",
        ): 1,
        ("recipes/bad.py", "<module>", "blackjax.nuts"): 1,
    }


def test_catalog_python_is_inside_the_boundary(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "custom.py").write_text(
        "import blackjax\nblackjax.nuts(logdensity_fn)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("catalog/custom.py", "<module>", "blackjax.nuts")] == 1


def test_generated_cache_is_outside_the_authored_source_boundary(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "catalog" / "model" / "_cache" / "generated_runs"
    generated.mkdir(parents=True)
    (generated / "program.py").write_text(
        "import blackjax\nblackjax.nuts(logdensity_fn)\n"
    )
    assert not boundary.scan_source(tmp_path)


def test_chained_assignment_alias_is_reported(tmp_path: Path) -> None:
    (tmp_path / "aliases.py").write_text(
        "def sample(base_method):\n"
        "    first = base_method.factory\n"
        "    second = first\n"
        "    return second(logdensity_fn)\n"
    )
    hits = boundary.scan_source(tmp_path)
    assert hits[("aliases.py", "sample", "factory")] == 1


def test_unknown_assignment_alias_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "unknown.py").write_text(
        "def sample():\n" "    helper = object()\n" "    return helper(logdensity_fn)\n"
    )
    assert not boundary.scan_source(tmp_path)


def test_scanner_preserves_call_multiplicity(tmp_path: Path) -> None:
    path = tmp_path / "generated.py"
    path.write_text(
        "def generated():\n"
        "    base_method.factory(logdensity_fn)\n"
        "    base_method.factory(logdensity_fn)\n"
    )
    key = ("generated.py", "generated", "factory")
    assert boundary.scan_source(tmp_path)[key] == 2


def test_comments_strings_and_docstrings_do_not_trigger(tmp_path: Path) -> None:
    (tmp_path / "noise.py").write_text(
        '"""run_smc(); obj.runner(); obj.factory()"""\n'
        "# run_smc(); obj.runner(); obj.factory()\n"
        "text = 'run_inference_algorithm(obj); obj.runner()'\n"
    )
    assert not boundary.scan_source(tmp_path)
