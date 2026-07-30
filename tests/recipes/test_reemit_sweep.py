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


class TestLowRankPilotBudget:
    """The pilot budget is a setting, and a silent default collapses the rank."""

    def test_low_rank_cells_carry_rank_and_pilot_budget(self) -> None:
        """k_rank and the pilot budget must be reconstructed, not defaulted.

        The pilot's effective sample size gates how much rank the rank-safety
        check permits.  Running a 10x-too-small pilot on a 50-dimensional target
        drops n_eff to about 4, clamps the usable rank from 40 to 1-2, and the
        preconditioner collapses to near-diagonal — so the cell fails
        certification for a reason that has nothing to do with the cell.  The
        harness defaults are 1000/1000; several cells record 10000/10000.
        """
        catalog = Path(__file__).parent.parent.parent / "tuningfork" / "catalog"
        recipe = (
            catalog
            / "ill_cond_50"
            / "recipes"
            / "low__mclmc_lrd__mclmc_lrd_tuning.json"
        )
        if not recipe.exists():
            pytest.skip("low-rank-diagonal recipe not in the catalog")

        config = driver.reconstruct(recipe)
        assert isinstance(config, driver.CellConfig)
        assert config.harness == "mclmc_lrd"
        assert config.k_rank == 40
        assert config.pilot_n_warmup == 10000
        assert config.pilot_n_samples == 10000

    def test_low_rank_cell_without_a_recorded_pilot_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Better to skip than to silently substitute the harness default."""
        model_dir = tmp_path / "some_model" / "recipes"
        model_dir.mkdir(parents=True)
        p = model_dir / "low__mclmc_lrd__mclmc_lrd_tuning.json"
        p.write_text(
            json.dumps(
                {
                    "effort": "low",
                    "base_method_name": "mclmc_lrd",
                    "warmup_name": "mclmc_lrd_tuning",
                    "base_method_params": {},
                    "warmup_params": {"n_warmup": 1000, "num_chains": 4},
                    "calibration_budget": {"n_samples": 1000},
                }
            )
        )
        result = driver.reconstruct(p)
        assert isinstance(result, driver.Skip)


class TestConfigFidelityIsAStandingGate:
    """Every committed recipe must record a configuration the emit path can reproduce.

    Kept in the suite rather than in the sweep script because the check is
    general: the next warmup family that grows a hyperparameter will drop it the
    same way the variational warmups' num_optimization_steps was dropped, and
    nothing else in the repo would notice.  Baseline-free by construction — it
    reconstructs each recipe from itself, so it asserts the artifact is
    self-reproducible rather than comparing against a revision.
    """

    def test_every_recipe_records_a_reproducible_config(self) -> None:
        catalog = Path(__file__).parent.parent.parent / "tuningfork" / "catalog"
        violations = []
        checked = 0
        for p in sorted(catalog.glob("*/recipes/*.json")):
            cfg = driver.reconstruct(p)
            if isinstance(cfg, driver.Skip):
                continue
            checked += 1
            bad = driver.config_fidelity_violations(cfg, json.loads(p.read_text()))
            if bad:
                violations.append(f"{p.parent.parent.name}/{p.name}: " + "; ".join(bad))
        assert checked > 100, (
            f"only {checked} recipes reconstructed — the gate has gone nearly "
            f"vacuous and is no longer testing the corpus"
        )
        assert not violations, (
            f"recipes record a configuration the emit path would not reproduce "
            f"({checked} checked):\n" + "\n".join(violations)
        )


def _minimal_cfg(**overrides):
    """A CellConfig for a real (sampler, warmup) pair, for fidelity-check tests."""
    fields = {
        "recipe_path": Path("tuningfork/catalog/mvn_10/recipes/low__nuts__w.json"),
        "model_name": "mvn_10",
        "warmup_name": "window_adaptation_diag_imm",
        "sampler_name": "nuts",
        "effort": "low",
        "harness": "recipe_runner",
        "n_warmup": 1000,
        "n_samples": 1000,
        "num_chains": 4,
        "seed": 20260517,
    }
    fields.update(overrides)
    return driver.CellConfig(**fields)


def _committed(x64, **overrides):
    """A committed artifact recording ``x64`` as the precision it ran at."""
    doc = {
        "warmups": [{"params": {}}],
        "base_method_params": {},
        "calibration_budget": {"machine_info": {"jax_x64_enabled": x64}},
    }
    doc.update(overrides)
    return doc


class TestPrecisionFidelity:
    """A replay must run at the float precision of the run it reproduces.

    Every other comparison in ``config_fidelity_violations`` is of a recorded
    PARAMETER.  x64 is not one — it follows the model's ``requires_x64`` and
    otherwise the ambient environment — so a cell committed under
    ``JAX_ENABLE_X64=1`` can be replayed in float32 with every parameter check
    green.  That is not hypothetical: it happened to 15 cells while both gates
    reported 0 mismatches on all 138.
    """

    def test_an_unpinned_precision_flip_is_a_violation(self) -> None:
        cfg = _minimal_cfg(recorded_x64=False)
        assert (
            cfg.key not in driver.PRECISION_FLIP_CELLS
        ), "fixture cell is pinned, so this test would pass for the wrong reason"
        bad = driver.config_fidelity_violations(cfg, _committed(True))
        assert any("jax_x64_enabled" in v for v in bad), bad

    def test_a_missing_baseline_flag_is_a_violation(self) -> None:
        """An absent flag means the committed precision is unknown, not float32."""
        cfg = _minimal_cfg(recorded_x64=False)
        committed = _committed(True)
        committed["calibration_budget"]["machine_info"] = {}
        bad = driver.config_fidelity_violations(cfg, committed)
        assert any("jax_x64_enabled" in v for v in bad), bad

    def test_matching_precision_is_not_a_violation(self) -> None:
        cfg = _minimal_cfg(recorded_x64=False)
        bad = driver.config_fidelity_violations(cfg, _committed(False))
        assert not any("jax_x64_enabled" in v for v in bad), bad

    def test_pinned_flips_are_accepted_and_each_records_its_transition(self) -> None:
        """The pin is an acceptance list, so an empty or unreasoned one is a lie.

        Emptying it would make every known flip re-appear as a gate failure, and
        an entry with no recorded transition tells a reader nothing about what was
        accepted — the same discipline CONFIG_CORRECTION_CELLS is held to.
        """
        assert driver.PRECISION_FLIP_CELLS
        for key, reason in driver.PRECISION_FLIP_CELLS.items():
            assert "/" in key, f"{key!r} should be '<model>/<filename>'"
            assert "jax_x64_enabled" in reason, f"{key} does not record its transition"

    def test_every_pinned_flip_is_accepted_rather_than_reported(self) -> None:
        assert driver.PRECISION_FLIP_CELLS, "nothing pinned; the test proves nothing"
        for pinned in driver.PRECISION_FLIP_CELLS:
            model, filename = pinned.split("/", 1)
            cfg = _minimal_cfg(
                model_name=model,
                recipe_path=Path(f"tuningfork/catalog/{model}/recipes/{filename}"),
                recorded_x64=False,
            )
            assert cfg.key == pinned
            bad = driver.config_fidelity_violations(cfg, _committed(True))
            assert not any("jax_x64_enabled" in v for v in bad), f"{pinned}: {bad}"


class TestEnumerationDirection:
    """Every field family is enumerated from the COMMITTED side, not cfg's.

    ``recertify()`` and the verify gate both depend on ``reconstruct()``, which
    raises an obvious question: if a field ``reconstruct()`` fails to extract
    is also a field ``config_fidelity_violations`` doesn't know to look for,
    the two share a blind spot and neither would ever see the drop. That is
    only true if the check iterates the fields CFG believes exist. It does
    not: ``warmup_params`` and ``base_method_params`` both loop over the
    COMMITTED dict's own keys (``for key, want in committed_warmup.items()``,
    ``for key, want in committed_kernel.items()``) and ask whether the replay
    built from cfg has each one -- so a key cfg has never heard of is still
    flagged, which is exactly the shape of the horseshoe defect this suite
    guards against (below). The three ``structural`` fields (``step_policy``,
    ``warmup_inner_kernel``, ``init_strategy``) are a narrower case: the KEY
    SET checked is a fixed, hand-written list of exactly those three names
    (not derived from committed's own keys the way the two dicts above are),
    but each comparison re-reads ``committed.get(key)`` fresh rather than
    trusting cfg's belief about it, so a wrong/stale cfg value for one of
    those three specific fields is still caught (also tested below). A
    hypothetical FOURTH structural field added to the schema without a
    matching line in ``structural`` would be invisible to this check -- that
    residual gap is real and is not what these tests claim to close.
    """

    def test_base_method_params_key_unknown_to_cfg_is_still_caught(self) -> None:
        """The exact shape of the horseshoe defect, reproduced directly.

        cfg carries NO sampler_kwargs_override at all -- as if reconstruct()
        itself, not just a hand-typed call, had never heard of this key. The
        check still catches it because it iterates committed's keys, not
        cfg's.
        """
        cfg = _minimal_cfg(sampler_kwargs_override=None)
        committed = _committed(False)
        committed["base_method_params"] = {"max_num_doublings": 15}
        bad = driver.config_fidelity_violations(cfg, committed)
        assert any("max_num_doublings" in v for v in bad), bad

    def test_warmup_params_key_unknown_to_cfg_is_still_caught(self) -> None:
        cfg = _minimal_cfg(warmup_kwargs_override=None)
        committed = _committed(False)
        committed["warmups"] = [{"params": {"__unknown_test_kwarg__": "sentinel"}}]
        bad = driver.config_fidelity_violations(cfg, committed)
        assert any("__unknown_test_kwarg__" in v for v in bad), bad

    def test_base_method_params_value_drift_is_caught_not_just_absence(self) -> None:
        """A key present on BOTH sides with a different value is still a violation.

        TL review question: does base_method_params check key PRESENCE only
        (a replay that carries the key at all "passes", drifted value or not)
        or VALUE too? cfg here WOULD replay max_num_doublings -- just at a
        different value than committed records -- so the presence check alone
        would pass this silently. It doesn't: the ``elif replayed_kernel[key]
        != want`` branch fires.
        """
        cfg = _minimal_cfg(sampler_kwargs_override={"max_num_doublings": 15})
        committed = _committed(False)
        committed["base_method_params"] = {"max_num_doublings": 20}
        bad = driver.config_fidelity_violations(cfg, committed)
        assert any("max_num_doublings" in v and "committed 20" in v for v in bad), bad

    def test_structural_field_comparison_does_not_trust_cfgs_own_belief(self) -> None:
        """init_strategy is re-read from committed, not assumed from cfg.

        cfg believes there is no init_strategy (None); committed disagrees.
        If the comparison trusted cfg's own value instead of re-reading
        committed, this would pass silently -- it doesn't.
        """
        cfg = _minimal_cfg(init_strategy=None)
        committed = _committed(False)
        committed["init_strategy"] = {"type": "zero_perchain", "jitter": 0.5}
        bad = driver.config_fidelity_violations(cfg, committed)
        assert any("init_strategy" in v for v in bad), bad


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
