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
"""Guards on the re-emit driver's configuration reconstruction.

The driver decides which committed recipes may be re-measured and under what
call.  A reconstruction error does not fail loudly — it produces a plausible
number for the wrong configuration — so the decisions worth testing are the ones
that REFUSE.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_TOOLS = Path(__file__).parent.parent.parent / "tools" / "reemit_sweep.py"


def _load_driver():
    """Import the driver from tools/, which is a script directory, not a package.

    The module must be registered in ``sys.modules`` BEFORE execution: its
    ``@dataclass`` declarations resolve annotations by looking themselves up
    there, and fail with an opaque AttributeError if absent.
    """
    import sys

    spec = importlib.util.spec_from_file_location("reemit_sweep", _TOOLS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reemit_sweep"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


driver = _load_driver()


class TestFilenameRoundTrip:
    """A re-emit must write back to the file it came from."""

    def test_inner_kernel_tag_is_recomposed(self) -> None:
        """An explicit inner kernel that differs from the default earns a tag.

        Without this the emit lands on the untagged neighbour: it would overwrite
        a different cell's recipe and leave the source orphaned, with both files
        still parsing cleanly so nothing downstream would notice.
        """
        assert driver._combined_filename_tag("hmc", "nuts", None) == "inner_nuts"
        assert driver._combined_filename_tag("mhmc", "nuts", None) == "inner_nuts"

    def test_no_tag_when_inner_kernel_matches_the_implicit_default(self) -> None:
        """nuts-warmed nuts, and the substitute family, carry no inner tag."""
        assert driver._combined_filename_tag("nuts", "nuts", None) is None
        # dynamic_hmc is warmed by nuts implicitly, so naming it changes nothing.
        assert driver._combined_filename_tag("dynamic_hmc", "nuts", None) is None

    def test_tags_compose_in_the_runner_order(self) -> None:
        assert (
            driver._combined_filename_tag("hmc", "nuts", "policy_v2-long")
            == "inner_nuts__policy_v2-long"
        )

    def test_policy_tag_is_read_off_the_filename(self) -> None:
        assert (
            driver._policy_tag(
                "medium__dmhmc__window_adaptation_diag_imm__policy_v2-long.json"
            )
            == "policy_v2-long"
        )
        assert driver._policy_tag("low__nuts__window_adaptation_diag_imm.json") is None


class TestSeedRecovery:
    """The master seed never reaches the artifact; only its derivative does."""

    def test_default_seed_derives_the_corpus_wide_tuning_seed(self) -> None:
        """682737 is what the documented default stamps, on every jax version tried."""
        from tuningfork.recipes._recipe_runner import RECIPE_SEED

        assert driver._predicted_tuning_seed(RECIPE_SEED) == 682737

    def test_candidates_include_the_second_generation_seed(self) -> None:
        """A re-emitted cell sits one generation down the seed chain.

        The rerun path feeds a recipe's own tuning_seed back in as the master
        seed, so cells re-emitted that way record a tuning_seed the default never
        produces.  Dropping this candidate silently skips 19 committed cells.
        """
        from tuningfork.recipes._recipe_runner import RECIPE_SEED

        candidates = driver._seed_candidates()
        assert RECIPE_SEED in candidates
        assert driver._predicted_tuning_seed(RECIPE_SEED) in candidates
        assert driver._predicted_tuning_seed(682737) == 4089912102


class TestKernelKwargReconstruction:
    """Only caller overrides may be replayed; derived values must re-derive."""

    def test_warmup_derived_trajectory_length_is_not_replayed(self) -> None:
        """Pinning a warmup-derived value would freeze a measurement as a setting."""
        from tuningfork.base_method import BASE_METHODS

        recipe = {
            "base_method_name": "hmc",
            "base_method_params": {"step_size": 0.1, "num_integration_steps": 37},
        }
        override, notes, blocker = driver._reconstruct_sampler_kwargs(
            recipe, BASE_METHODS["hmc"], "nuts"
        )
        assert blocker is None
        assert override is None or "num_integration_steps" not in override
        assert any("warmup-derived" in n for n in notes)

    def test_adapted_parameters_are_never_replayed(self) -> None:
        """step_size and the mass matrix come from warmup, not from the caller."""
        from tuningfork.base_method import BASE_METHODS

        recipe = {
            "base_method_name": "nuts",
            "base_method_params": {"step_size": 0.003, "inverse_mass_matrix": [1.0]},
        }
        override, _, blocker = driver._reconstruct_sampler_kwargs(
            recipe, BASE_METHODS["nuts"], None
        )
        assert blocker is None
        assert override is None

    def test_non_registry_kwarg_is_replayed_and_named(self) -> None:
        """A pinned kwarg outside the default space can only be an override."""
        from tuningfork.base_method import BASE_METHODS

        recipe = {
            "base_method_name": "nuts",
            "base_method_params": {"step_size": 0.003, "max_num_doublings": 15},
        }
        override, notes, blocker = driver._reconstruct_sampler_kwargs(
            recipe, BASE_METHODS["nuts"], None
        )
        assert blocker is None
        assert override == {"max_num_doublings": 15}
        assert any("max_num_doublings" in n for n in notes)


class TestWarmupParameterSchemas:
    """Two on-disk warmup schemas are in use; both must yield the same call."""

    def test_target_acceptance_survives_the_warmups_list_schema(self) -> None:
        """A curvature-sensitive cell must not be replayed at the default 0.8.

        Most committed recipes record warmup settings as a ``warmups`` list with
        no flat ``warmup_params`` dict.  Reading the flat key directly returns
        an empty dict for those, silently dropping ``target_acceptance`` — which
        does not fail loudly, it just reruns the sampler at a different setting.
        Observed on banana: 1593 divergences out of 4000 at the default, 0 at the
        recorded 0.99.
        """
        catalog = Path(__file__).parent.parent.parent / "tuningfork" / "catalog"
        recipe = (
            catalog
            / "banana"
            / "recipes"
            / "medium__dynamic_hmc__window_adaptation_diag_imm__policy_v1-medium.json"
        )
        if not recipe.exists():
            pytest.skip("banana policy recipe not in the catalog")

        raw = json.loads(recipe.read_text())
        assert "warmup_params" not in raw, (
            "fixture no longer exercises the warmups-list schema; pick another "
            "recipe that does, or this test proves nothing"
        )

        config = driver.reconstruct(recipe)
        assert isinstance(config, driver.CellConfig)
        assert config.target_acceptance == 0.99
        assert config.n_warmup == 2000
        assert config.num_chains == 4


class TestScopeDecisions:
    """Cells outside the migration are declined, not silently emitted."""

    @pytest.mark.parametrize(
        "recipe_name,expected",
        [
            ("smc__adaptive_tempered_smc__rwm.json", "importance-weight ESS"),
            ("failed__elliptical_slice__no_warmup.json", "failed recipe"),
        ],
    )
    def test_out_of_scope_families_are_skipped(
        self, tmp_path: Path, recipe_name: str, expected: str
    ) -> None:
        model_dir = tmp_path / "some_model" / "recipes"
        model_dir.mkdir(parents=True)
        p = model_dir / recipe_name
        p.write_text(json.dumps({"effort": "failed", "base_method_name": "nuts"}))
        result = driver.reconstruct(p)
        assert isinstance(result, driver.Skip)
        assert expected in result.reason

    def test_config_correction_cells_are_flagged_not_pooled(self) -> None:
        """Cells emitted under the standard protocol must be separable downstream.

        Their movement is dominated by a corrected gradient budget, so pooling
        them with genuine replays would read as an estimator effect.
        """
        assert driver.CONFIG_CORRECTION_CELLS
        for key, reason in driver.CONFIG_CORRECTION_CELLS.items():
            assert "/" in key, f"{key!r} should be '<model>/<filename>'"
            assert reason.strip(), f"{key} has no recorded reason"
