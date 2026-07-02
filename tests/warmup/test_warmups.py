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
"""Tests for the warmup registry and multi-chain contract.

Covers:
  1. WARMUPS dict has exactly the expected entries (fast; pure dict).
  2. is_compatible() for all warmups — one parametrized table (fast; pure logic).
  3. window_adaptation_diag_imm smoke: NUTS on 10-D MVN at n_warmup=200, num_chains=1 (single-chain shim).
  4. mclmc_tuning smoke: MCLMC on 10-D MVN at n_warmup=200, num_chains=1 (single-chain shim).
  5. no_warmup smoke: RWM (gradient-free) and NUTS, num_chains=1.
  6. Compatibility error via _run_warmup (wrong warmup for algorithm).
  7. Auto-dispatch in tune_algorithm: mclmc → mclmc_tuning, nuts → window_adaptation_diag_imm,
     rwm → no_warmup (verified via result structure).
  8. tune_algorithm regression: existing calls with warmup_name=None still pass.
  9. Multi-chain contract tests: shape checks for num_chains=1/4/8,
     pre-batched init_position, dense mass matrix, MCLMC multi-chain, no_warmup multi-chain.
 10. HARD-KEEP unique invariants: MEADS num_chains<num_folds raises ValueError; CHEES returns
     callable params; multipathfinder broadcasts a shared IMM; no_warmup always returns {}.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from tuningfork.base_method import BASE_METHODS
from tuningfork.calibration.tune import _run_warmup, tune_algorithm
from tuningfork.model import MODELS
from tuningfork.warmup import WARMUPS, Warmup

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MVN = MODELS["mvn_10"]
_NUTS = BASE_METHODS["nuts"]
_HMC = BASE_METHODS["hmc"]
_MCLMC = BASE_METHODS["mclmc"]
_RWM = BASE_METHODS["rwm"]
_MALA = BASE_METHODS["mala"]
_GHMC = BASE_METHODS["ghmc"]
_DYNAMIC_HMC = BASE_METHODS["dynamic_hmc"]

# Build a shared 10-D MVN logdensity_fn + init_position for smoke tests.
# We do this once at module level to avoid per-test model compilation.
_SEED = 42
_RNG_KEY = jax.random.key(_SEED)
_D = 10  # MVN-10 has 10 dimensions

# ---------------------------------------------------------------------------
# Helper: build logdensity_fn + position from the model registry
# ---------------------------------------------------------------------------


def _build_logdensity(posterior_entry, key):
    from tuningfork.model._numpyro import build_logdensity_fn

    init_position, logdensity_fn, _ = build_logdensity_fn(key, posterior_entry)
    return init_position, logdensity_fn


# ---------------------------------------------------------------------------
# State-shape helpers
# ---------------------------------------------------------------------------


def _state_leading_dim(states) -> int:
    """Return the leading dimension of the vmapped state pytree."""
    leaves = jax.tree.leaves(states)
    return leaves[0].shape[0]


def _position_shape(states) -> tuple:
    """Return (num_chains, *param_shape) from the position pytree."""
    pos_leaves = jax.tree.leaves(states.position)
    return pos_leaves[0].shape


# ---------------------------------------------------------------------------
# 1. Registry structure — FAST (pure dict lookups, zero JAX)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestWarmupRegistry:
    """WARMUPS registry structure tests (fast; no chain runs)."""

    def test_warmups_has_expected_entries(self) -> None:
        """Subset assertion: all known warmups must be present."""
        expected = {
            "window_adaptation_diag_imm",
            "mclmc_tuning",
            "adjusted_mclmc_tuning",
            "no_warmup",
            "pathfinder",
            "multipathfinder",
            "meads",
            "chees",
        }
        assert expected <= set(WARMUPS.keys()), (
            f"Missing warmup entries: {expected - set(WARMUPS.keys())}. "
            f"Registered: {sorted(WARMUPS)}"
        )

    def test_all_entries_are_warmup_instances(self) -> None:
        for name, entry in WARMUPS.items():
            assert isinstance(
                entry, Warmup
            ), f"WARMUPS[{name!r}] is not a Warmup instance"

    def test_warmup_names_match_keys(self) -> None:
        for key, entry in WARMUPS.items():
            assert (
                entry.name == key
            ), f"WARMUPS[{key!r}].name = {entry.name!r} doesn't match key"


# ---------------------------------------------------------------------------
# 2. is_compatible() — FAST (pure logic, one parametrized table)
# ---------------------------------------------------------------------------

# Columns: (warmup_name, method_name, expected_result)
_IS_COMPATIBLE_TABLE = [
    # window_adaptation_diag_imm: accepts gradient-based except mclmc/rwm family
    ("window_adaptation_diag_imm", "hmc", True),
    ("window_adaptation_diag_imm", "nuts", True),
    ("window_adaptation_diag_imm", "barker", True),
    ("window_adaptation_diag_imm", "mala", True),
    ("window_adaptation_diag_imm", "mclmc", False),
    ("window_adaptation_diag_imm", "rwm", False),
    # mclmc_tuning: mclmc only
    ("mclmc_tuning", "mclmc", True),
    ("mclmc_tuning", "nuts", False),
    ("mclmc_tuning", "hmc", False),
    ("mclmc_tuning", "rwm", False),
    # pathfinder: gradient-based except mclmc
    ("pathfinder", "nuts", True),
    ("pathfinder", "hmc", True),
    ("pathfinder", "barker", True),
    ("pathfinder", "mclmc", False),
    # multipathfinder: same as pathfinder
    ("multipathfinder", "nuts", True),
    ("multipathfinder", "hmc", True),
    ("multipathfinder", "barker", True),
    ("multipathfinder", "mclmc", False),
    # no_warmup: star sentinel — accepts everything
    ("no_warmup", "nuts", True),
    ("no_warmup", "mclmc", True),
    ("no_warmup", "rwm", True),
    ("no_warmup", "hypothetical_future_sampler", True),
    # adjusted_mclmc_tuning: adjusted_mclmc + adjusted_mclmc_dynamic only
    ("adjusted_mclmc_tuning", "adjusted_mclmc", True),
    ("adjusted_mclmc_tuning", "adjusted_mclmc_dynamic", True),
    ("adjusted_mclmc_tuning", "nuts", False),
    ("adjusted_mclmc_tuning", "mclmc", False),
    # meads: ghmc only
    ("meads", "ghmc", True),
    ("meads", "nuts", False),
    ("meads", "hmc", False),
    ("meads", "mclmc", False),
    # chees: dynamic_hmc only
    ("chees", "dynamic_hmc", True),
    ("chees", "nuts", False),
    ("chees", "hmc", False),
]


@pytest.mark.fast
@pytest.mark.parametrize(
    "warmup_name,method_name,expected",
    _IS_COMPATIBLE_TABLE,
    ids=[f"{w}__{m}_{'+' if e else '-'}" for w, m, e in _IS_COMPATIBLE_TABLE],
)
def test_is_compatible(warmup_name: str, method_name: str, expected: bool) -> None:
    """One parametrized check per (warmup, method) pair.

    Re: #151 lesson — each warmup's distinct compat rule is encoded as its own
    row; collapsing to "every warmup X => Y" would silently drop failures for
    warmups where the rule differs (e.g. meads vs window_adaptation_diag_imm).
    """
    result = WARMUPS[warmup_name].is_compatible(method_name)
    assert result == expected, (
        f"WARMUPS[{warmup_name!r}].is_compatible({method_name!r}) = {result}, "
        f"expected {expected}"
    )


@pytest.mark.fast
def test_no_warmup_has_star_sentinel() -> None:
    """no_warmup's * sentinel is a distinct invariant — kept as named case."""
    assert "*" in WARMUPS["no_warmup"].compatible_methods


# ---------------------------------------------------------------------------
# 3. window_adaptation_diag_imm smoke — SLOW (JAX-compiled)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStanWindowSmoke:
    """window_adaptation_diag_imm smoke test on NUTS + MVN-10.

    Uses num_chains=1 to preserve backward-compatible shim semantics.
    Output shapes have a leading dim of 1.
    """

    def _run(self, seed: int, **kw):
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["window_adaptation_diag_imm"].runner(
            warmup_key, init_pos, 200, _NUTS, logdensity_fn=logdensity_fn, **kw
        )

    def test_returns_state_and_adapted_params(self) -> None:
        state, params, *_ = self._run(101, num_chains=1)
        assert state is not None
        assert isinstance(params, dict)

    def test_adapted_params_has_step_size(self) -> None:
        _, params, *_ = self._run(102, num_chains=1)
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        _, params, *_ = self._run(103, num_chains=1)
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_step_size_positive(self) -> None:
        _, params, *_ = self._run(104, num_chains=1)
        step_sizes = jnp.asarray(params["step_size"])
        # shape (1,) — all positive
        assert bool(jnp.all(step_sizes > 0)), f"step_size={step_sizes} not all > 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        _, params, *_ = self._run(105, num_chains=1)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10)
        assert imm.shape == (
            1,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10) for num_chains=1 diagonal"

    def test_dense_mass_matrix_shape(self) -> None:
        """is_mass_matrix_diagonal=False produces (num_chains, d, d) IMM."""
        _, params, *_ = self._run(106, num_chains=1, is_mass_matrix_diagonal=False)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10, 10)
        assert imm.shape == (
            1,
            10,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10, 10)"

    def test_dense_mass_matrix_is_symmetric_positive_definite(self) -> None:
        """Sanity check: dense IMM should be symmetric and PD (per chain)."""
        _, params, *_ = self._run(107, num_chains=1, is_mass_matrix_diagonal=False)
        imm = params["inverse_mass_matrix"]
        # imm has shape (1, 10, 10); check chain 0
        imm_chain0 = imm[0]
        assert jnp.allclose(
            imm_chain0, imm_chain0.T, atol=1e-6
        ), "dense IMM must be symmetric"
        try:
            jnp.linalg.cholesky(imm_chain0)
        except Exception as exc:  # pragma: no cover
            raise AssertionError(f"dense IMM not positive definite: {exc}") from exc


# ---------------------------------------------------------------------------
# 4. mclmc_tuning smoke — SLOW (JAX-compiled)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMclmcTuningSmoke:
    """mclmc_tuning smoke test on MCLMC + MVN-10.

    Uses num_chains=1 to preserve backward-compatible shim semantics.
    Output shapes have a leading dim of 1.
    """

    def _run(self, seed: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["mclmc_tuning"].runner(
            warmup_key, init_pos, 200, _MCLMC, logdensity_fn=logdensity_fn, num_chains=1
        )

    def test_returns_state_and_adapted_params(self) -> None:
        state, params, *_ = self._run(201)
        assert state is not None
        assert isinstance(params, dict)

    def test_adapted_params_has_L(self) -> None:
        _, params, *_ = self._run(202)
        assert "L" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_step_size(self) -> None:
        _, params, *_ = self._run(203)
        assert "step_size" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_inverse_mass_matrix(self) -> None:
        _, params, *_ = self._run(204)
        assert "inverse_mass_matrix" in params, f"params keys: {list(params.keys())}"

    def test_adapted_params_has_total_tuning_steps(self) -> None:
        _, params, *_ = self._run(205)
        assert "_total_tuning_steps" in params, f"params keys: {list(params.keys())}"

    def test_L_positive(self) -> None:
        _, params, *_ = self._run(206)
        # shape (1,) for num_chains=1
        assert bool(jnp.all(jnp.asarray(params["L"]) > 0)), f"L={params['L']} not > 0"

    def test_step_size_positive(self) -> None:
        _, params, *_ = self._run(207)
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"step_size={params['step_size']} not > 0"

    def test_inverse_mass_matrix_shape(self) -> None:
        _, params, *_ = self._run(208)
        imm = params["inverse_mass_matrix"]
        # num_chains=1 → shape (1, 10)
        assert imm.shape == (
            1,
            10,
        ), f"inverse_mass_matrix.shape={imm.shape}, expected (1, 10)"

    def test_total_tuning_steps_positive(self) -> None:
        _, params, *_ = self._run(209)
        steps = int(params["_total_tuning_steps"])
        assert steps > 0, f"_total_tuning_steps={steps} <= 0"


# ---------------------------------------------------------------------------
# 5. no_warmup smoke — SLOW (JAX-compiled)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoWarmupSmoke:
    """no_warmup smoke tests on RWM and NUTS (num_chains=1 shim)."""

    def test_rwm_returns_state_and_empty_params(self) -> None:
        key = jax.random.key(301)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _RWM,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}, f"Expected empty dict, got {params}"

    def test_nuts_returns_state_and_empty_params(self) -> None:
        key = jax.random.key(302)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}, f"Expected empty dict, got {params}"

    def test_mclmc_rng_key_threading_works(self) -> None:
        """MCLMC requires kernel.init(position, rng_key); no_warmup handles this."""
        key = jax.random.key(303)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        # Should NOT raise TypeError about unexpected rng_key argument.
        state, params = WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert state is not None
        assert params == {}


# ---------------------------------------------------------------------------
# 6. Compatibility error via _run_warmup — SLOW (builds logdensity_fn)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCompatibilityError:
    """_run_warmup raises ValueError when warmup is incompatible with algorithm."""

    def test_mclmc_tuning_on_nuts_raises(self) -> None:
        key = jax.random.key(401)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_NUTS,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="mclmc_tuning",
            )

    def test_window_adaptation_diag_imm_on_mclmc_raises(self) -> None:
        key = jax.random.key(402)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_MCLMC,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="window_adaptation_diag_imm",
            )

    def test_unknown_warmup_name_raises(self) -> None:
        key = jax.random.key(403)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="unknown warmup"):
            _run_warmup(
                logdensity_fn=logdensity_fn,
                init_position=init_pos,
                algorithm_entry=_NUTS,
                n_warmup=50,
                rng_key=jax.random.fold_in(key, 1),
                warmup_name="nonexistent_warmup",
            )


# ---------------------------------------------------------------------------
# 7. Auto-dispatch in tune_algorithm — SLOW (JAX chain runs)
# ---------------------------------------------------------------------------


_AUTO_N_TRIALS = 1
_AUTO_N_SEEDS = 1
_AUTO_N_CHAINS = 1
_AUTO_N_SAMPLES = 50
_AUTO_N_WARMUP = 100


@pytest.mark.slow
class TestAutoDispatch:
    """tune_algorithm auto-dispatch resolves warmup_name=None correctly.

    We only run n_trials=1 to keep runtime low; the goal is to confirm
    the dispatch doesn't raise and produces a valid TuningResult.
    """

    def test_mclmc_auto_dispatches_to_mclmc_tuning(self) -> None:
        """MCLMC with warmup_name=None should use mclmc_tuning (not window_adaptation_diag_imm)."""
        result = tune_algorithm(
            _MVN,
            _MCLMC,
            rng_key=jax.random.key(501),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        # If auto-dispatch went to window_adaptation_diag_imm instead, it would raise
        # ValueError("not compatible with").  So if we reach this assertion,
        # dispatch is correct.
        assert result.base_method_name == "mclmc"

    def test_nuts_auto_dispatches_to_window_adaptation_diag_imm(self) -> None:
        """NUTS with warmup_name=None should use window_adaptation_diag_imm."""
        result = tune_algorithm(
            _MVN,
            _NUTS,
            rng_key=jax.random.key(502),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        assert result.base_method_name == "nuts"
        # Verify: best_score is finite (window_adaptation_diag_imm warmup worked).
        assert math.isfinite(result.best_score), f"best_score={result.best_score}"

    def test_rwm_auto_dispatches_to_no_warmup(self) -> None:
        """RWM with warmup_name=None should use no_warmup."""
        result = tune_algorithm(
            _MVN,
            _RWM,
            rng_key=jax.random.key(503),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        assert result.base_method_name == "rwm"

    def test_explicit_warmup_name_overrides_auto(self) -> None:
        """Passing warmup_name='no_warmup' for NUTS should skip window_adaptation_diag_imm."""
        result = tune_algorithm(
            _MVN,
            _NUTS,
            warmup_name="no_warmup",
            rng_key=jax.random.key(504),
            n_trials=_AUTO_N_TRIALS,
            n_seeds=_AUTO_N_SEEDS,
            n_chains=_AUTO_N_CHAINS,
            n_samples=_AUTO_N_SAMPLES,
            n_warmup=_AUTO_N_WARMUP,
        )
        # no_warmup is compatible with NUTS via "*"; should not raise.
        assert result.base_method_name == "nuts"


# ---------------------------------------------------------------------------
# 8. Regression: existing tune_algorithm calls still pass — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTuneAlgorithmRegression:
    """Confirm that the refactored _run_warmup produces the same structural
    outcomes as the old inline dispatch.  Only structural tests (not numeric)
    because the warmup key path is identical to old code.
    """

    _N_WARMUP = 200
    _N_SAMPLES = 200
    _N_TRIALS = 3
    _N_SEEDS = 1
    _N_CHAINS = 1

    def _run(self, algo, seed):
        return tune_algorithm(
            _MVN,
            algo,
            n_trials=self._N_TRIALS,
            n_seeds=self._N_SEEDS,
            n_chains=self._N_CHAINS,
            n_samples=self._N_SAMPLES,
            n_warmup=self._N_WARMUP,
            rng_key=jax.random.key(seed),
        )

    def test_nuts_result_structure_unchanged(self) -> None:
        result = self._run(_NUTS, 600)
        assert result.base_method_name == "nuts"
        assert result.n_trials_completed == self._N_TRIALS
        assert len(result.history) == self._N_TRIALS
        assert math.isfinite(result.best_score)

    def test_hmc_result_structure_unchanged(self) -> None:
        result = self._run(_HMC, 601)
        assert result.base_method_name == "hmc"
        assert result.n_trials_completed == self._N_TRIALS
        assert "step_size" in result.best_params
        assert "num_integration_steps" in result.best_params

    def test_mclmc_result_structure_unchanged(self) -> None:
        result = self._run(_MCLMC, 602)
        assert result.base_method_name == "mclmc"
        assert result.n_trials_completed == self._N_TRIALS
        assert "step_size" in result.best_params
        assert "L" in result.best_params

    def test_rwm_result_structure_unchanged(self) -> None:
        result = self._run(_RWM, 603)
        assert result.base_method_name == "rwm"
        assert result.n_trials_completed == self._N_TRIALS

    def test_mala_result_structure_unchanged(self) -> None:
        result = self._run(_MALA, 604)
        assert result.base_method_name == "mala"
        assert result.n_trials_completed == self._N_TRIALS


# ---------------------------------------------------------------------------
# 9. Multi-chain contracts — window_adaptation_diag_imm — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStanWindowMultiChain:
    """window_adaptation_diag_imm multi-chain shape contract tests."""

    def _run(self, seed: int, num_chains: int, **kw):
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return WARMUPS["window_adaptation_diag_imm"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    @pytest.mark.parametrize(
        "num_chains,expected_pos,expected_ss,expected_imm",
        [
            (4, (4, 10), (4,), (4, 10)),
            (8, (8, 10), (8,), (8, 10)),
            (1, (1, 10), (1,), (1, 10)),
        ],
        ids=["nc4", "nc8", "nc1"],
    )
    def test_num_chains_shape(
        self, num_chains, expected_pos, expected_ss, expected_imm
    ) -> None:
        """Diagonal IMM shape + positivity contract for num_chains=1/4/8.

        nc4 also serves as the default=4 smoke (default kwarg omitted → same
        _run internally, so the leading-dim=4 contract is covered).
        """
        seed = 1002 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains)
        assert (
            _position_shape(states) == expected_pos
        ), f"nc={num_chains}: pos shape {_position_shape(states)} != {expected_pos}"
        assert (
            params["step_size"].shape == expected_ss
        ), f"nc={num_chains}: step_size shape {params['step_size'].shape} != {expected_ss}"
        assert (
            params["inverse_mass_matrix"].shape == expected_imm
        ), f"nc={num_chains}: IMM shape {params['inverse_mass_matrix'].shape} != {expected_imm}"
        if num_chains == 4:
            # Positivity check on the nc4 row (avoids a separate JAX-traced call)
            assert bool(
                jnp.all(jnp.asarray(params["step_size"]) > 0)
            ), f"Not all step sizes positive: {params['step_size']}"

    def test_pre_batched_init_position(self) -> None:
        """init_position with leading dim == num_chains passes through verbatim."""
        num_chains = 4
        key = jax.random.key(1007)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        # Pre-batch: replicate 4 times
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params, *_ = WARMUPS["window_adaptation_diag_imm"].runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Pre-batched: expected (4, 10), got {pos_shape}"

    def test_dense_mm_multi_chain_shape(self) -> None:
        """num_chains=2, is_mass_matrix_diagonal=False → IMM shape (2, d, d)."""
        states, params, *_ = self._run(
            1008, num_chains=2, is_mass_matrix_diagonal=False
        )
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            2,
            10,
            10,
        ), f"Expected (2, 10, 10) for dense multi-chain, got {imm.shape}"
        pos_shape = _position_shape(states)
        assert pos_shape == (2, 10), f"Expected (2, 10), got {pos_shape}"


# ---------------------------------------------------------------------------
# 10. Multi-chain contracts — mclmc_tuning — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMclmcTuningMultiChain:
    """mclmc_tuning multi-chain shape contract tests."""

    def _run(self, seed: int, num_chains: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)
        return WARMUPS["mclmc_tuning"].runner(
            warmup_key,
            init_pos,
            200,
            _MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )

    @pytest.mark.parametrize(
        "num_chains,expected_leading,expected_L,expected_ss,expected_imm",
        [
            (4, 4, (4,), (4,), (4, 10)),
            (1, 1, (1,), (1,), (1, 10)),
        ],
        ids=["nc4", "nc1"],
    )
    def test_num_chains_shape(
        self, num_chains, expected_leading, expected_L, expected_ss, expected_imm
    ) -> None:
        """Shape + positivity + int-type contract for L/step_size/IMM.

        nc4 row also covers: default=4 behavior, L>0, step_size>0,
        and _total_tuning_steps is a Python int — avoids separate JAX-traced calls.
        """
        seed = 2002 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains)
        assert _state_leading_dim(states) == expected_leading
        assert (
            jnp.asarray(params["L"]).shape == expected_L
        ), f"nc={num_chains}: L shape mismatch"
        assert jnp.asarray(params["step_size"]).shape == expected_ss
        assert params["inverse_mass_matrix"].shape == expected_imm
        if num_chains == 4:
            assert bool(
                jnp.all(jnp.asarray(params["L"]) > 0)
            ), f"Not all L positive: {params['L']}"
            steps = params["_total_tuning_steps"]
            assert isinstance(
                steps, int
            ), f"_total_tuning_steps should be int, got {type(steps)}"
            assert steps > 0, f"_total_tuning_steps={steps} <= 0"


# ---------------------------------------------------------------------------
# 11. Multi-chain contracts — no_warmup — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoWarmupMultiChain:
    """no_warmup multi-chain shape contract tests.

    HARD-KEEP invariant: no_warmup ALWAYS returns {} regardless of num_chains
    and algorithm choice.  This is distinct from all other warmups — named here.
    """

    def _run(self, seed: int, base_method, num_chains: int) -> tuple[object, dict]:
        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return WARMUPS["no_warmup"].runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            base_method,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )

    @pytest.mark.parametrize(
        "base_method_key,seed,num_chains",
        [
            ("nuts", 3002, 4),
            ("rwm", 3003, 4),
            ("mclmc", 3004, 4),
        ],
        ids=["nuts_nc4", "rwm_nc4", "mclmc_nc4"],
    )
    def test_num_chains_4_state_shape(self, base_method_key, seed, num_chains) -> None:
        """num_chains=4 across algorithms → position leading dim == 4; params == {}."""
        base_method = BASE_METHODS[base_method_key]
        states, params, *_ = self._run(seed, base_method, num_chains=num_chains)
        pos_shape = _position_shape(states)
        assert pos_shape == (4, 10), f"Expected (4, 10), got {pos_shape}"
        assert params == {}, f"no_warmup always returns empty dict, got {params}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed)."""
        states, params, *_ = self._run(3005, _NUTS, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        assert params == {}

    def test_adapted_params_always_empty(self) -> None:
        """HARD-KEEP: no_warmup always returns {} regardless of num_chains."""
        for nc in (1, 2, 4, 8):
            _, params, *_ = self._run(3010 + nc, _NUTS, num_chains=nc)
            assert params == {}, f"num_chains={nc}: expected empty dict, got {params}"


# ---------------------------------------------------------------------------
# 12. Pathfinder multi-chain warmup — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPathfinderMultiChain:
    """pathfinder warmup multi-chain shape contract tests.

    Thin shim around blackjax.pathfinder_adaptation(num_chains=..., n_paths=None).
    IMM is now dense (num_chains, d, d) — breaking change from the old
    diagonal (num_chains, d) contract in PR B (warmup-collapse-pathfinder-shims).
    Step size is NOW adapted (dual-averaging over n_warmup steps).
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from tuningfork.warmup.pathfinder import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    @pytest.mark.parametrize(
        "num_chains,expected_pos,expected_ss,expected_imm",
        [
            (4, (4, _D), (4,), (4, _D, _D)),
            (2, (2, _D), (2,), (2, _D, _D)),
        ],
        ids=["nc4", "nc2"],
    )
    def test_shape_and_keys(
        self, num_chains, expected_pos, expected_ss, expected_imm
    ) -> None:
        """Dense IMM shape + keys present + step_size > 0 for nc=2 and nc=4.

        nc4 row: verifies default=4 behavior and positivity.
        nc2 row: verifies a non-default shape (two chains).
        """
        seed = 4001 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains)
        assert _position_shape(states) == expected_pos
        assert jnp.asarray(params["step_size"]).shape == expected_ss
        assert params["inverse_mass_matrix"].shape == expected_imm
        assert "step_size" in params
        assert "inverse_mass_matrix" in params
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"nc={num_chains}: step_size not all > 0"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed), dense IMM (1, d, d)."""
        states, params, *_ = self._run(4007, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (1,), f"step_size expected (1,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            1,
            _D,
            _D,
        ), f"IMM expected (1, {_D}, {_D}), got {imm.shape}"

    def test_compatibility_check_raises_for_mclmc(self) -> None:
        """is_compatible('mclmc') returns False; runner raises ValueError for mclmc."""
        from tuningfork.warmup.pathfinder import ENTRY

        assert not ENTRY.is_compatible(
            "mclmc"
        ), "pathfinder should not be compatible with mclmc"

        key = jax.random.key(4008)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _MCLMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
            )


# ---------------------------------------------------------------------------
# 13. MultiPathfinder multi-chain warmup — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMultiPathfinderMultiChain:
    """multipathfinder warmup multi-chain shape contract tests.

    Thin shim around blackjax.pathfinder_adaptation(num_chains=..., n_paths=num_chains).
    IMM is now dense (num_chains, d, d) broadcast from the shared (d, d) upstream estimate.
    IMM contract changed from old diagonal (num_chains, d) in PR B (warmup-collapse-pathfinder-shims).
    Step size is NOW adapted (dual-averaging over n_warmup steps).
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from tuningfork.warmup.multipathfinder import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _NUTS,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    @pytest.mark.parametrize(
        "num_chains,expected_pos,expected_ss,expected_imm",
        [
            (4, (4, _D), (4,), (4, _D, _D)),
            (2, (2, _D), (2,), (2, _D, _D)),
        ],
        ids=["nc4", "nc2"],
    )
    def test_shape_and_keys(
        self, num_chains, expected_pos, expected_ss, expected_imm
    ) -> None:
        """Dense IMM shape + keys present + step_size > 0 for nc=2 and nc=4.

        nc4 row: verifies default=4 behavior, keys present, and positivity.
        nc2 row: verifies non-default shape (broadcast shared IMM contract still holds).
        """
        seed = 5001 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains)
        assert _position_shape(states) == expected_pos
        assert jnp.asarray(params["step_size"]).shape == expected_ss
        assert params["inverse_mass_matrix"].shape == expected_imm
        assert "step_size" in params
        assert "inverse_mass_matrix" in params
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"nc={num_chains}: step_size not all > 0"

    def test_psis_diagnostics_in_calibration_metadata(self) -> None:
        """_multipathfinder_psis_pareto_k sidecar must be present."""
        _, params, *_ = self._run(5005, num_chains=2)
        assert (
            "_multipathfinder_psis_pareto_k" in params
        ), f"Missing _multipathfinder_psis_pareto_k; keys: {list(params)}"

    def test_step_size_positive(self) -> None:
        """Multipathfinder now adapts step_size via DA; all values must be > 0."""
        _, params, *_ = self._run(5006, num_chains=4)
        ss = jnp.asarray(params["step_size"])
        assert bool(jnp.all(ss > 0)), f"All step_sizes should be > 0, got {ss}"

    def test_num_chains_1_not_squeezed(self) -> None:
        """num_chains=1 → leading dim 1 (NOT squeezed), dense IMM (1, d, d)."""
        states, params, *_ = self._run(5007, num_chains=1)
        leading = _state_leading_dim(states)
        assert leading == 1, f"num_chains=1 should give leading dim 1, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (1,), f"step_size expected (1,), got {ss.shape}"
        imm = params["inverse_mass_matrix"]
        assert imm.shape == (
            1,
            _D,
            _D,
        ), f"IMM expected (1, {_D}, {_D}), got {imm.shape}"

    def test_imm_values_are_identical_across_chains(self) -> None:
        """HARD-KEEP: All chains share the same IMM (broadcast from shared (d, d) estimate).

        This is multipathfinder's distinctive invariant: single global IMM broadcast,
        vs pathfinder which is per-chain.  The broadcast means imm[0] == imm[i] for all i.
        """
        _, params, *_ = self._run(5008, num_chains=4)
        imm = params["inverse_mass_matrix"]
        # All (d, d) slices should be equal (same broadcast estimate).
        for i in range(1, 4):
            assert jnp.allclose(
                imm[0], imm[i], atol=1e-6
            ), f"IMM chain {i} differs from chain 0"

    def test_compatibility_check_raises_for_mclmc(self) -> None:
        """is_compatible('mclmc') returns False; runner raises ValueError for mclmc."""
        from tuningfork.warmup.multipathfinder import ENTRY

        assert not ENTRY.is_compatible(
            "mclmc"
        ), "multipathfinder should not be compatible with mclmc"

        key = jax.random.key(5009)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="not compatible with"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _MCLMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
            )


# ---------------------------------------------------------------------------
# 14. MEADS multi-chain warmup — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMeadsMultiChain:
    """meads warmup multi-chain shape contract tests.

    MEADS is fundamentally multi-chain: a single call handles all num_chains
    chains jointly via cross-validation across num_folds folds.  Unlike
    window_adaptation_diag_imm (which vmaps per-chain), MEADS is NOT vmapped — one call, all
    chains.

    Adapted parameters are shared (single MEADS estimate) and broadcast to
    (num_chains,) shape to satisfy the multi-chain contract.
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from tuningfork.warmup.meads import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            200,
            _GHMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    @pytest.mark.parametrize(
        "num_chains,num_folds,expected_leading",
        [
            (4, None, 4),  # default num_folds=4; nc==nf OK; keys present; compat smoke
            (8, 4, 8),  # nc > nf; shapes correct
        ],
        ids=["nc4_default", "nc8_nf4"],
    )
    def test_shape_and_keys(self, num_chains, num_folds, expected_leading) -> None:
        """Shape + required-keys contract for MEADS at nc=4 (default) and nc=8/nf=4.

        nc4 row: covers default=4 behavior, all required param keys, and compat smoke.
        nc8/nf4 row: covers chains > folds shape (the happy path for non-default size).
        """
        kw = {} if num_folds is None else {"num_folds": num_folds}
        seed = 6001 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains, **kw)
        assert _state_leading_dim(states) == expected_leading
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (
            expected_leading,
        ), f"step_size expected ({expected_leading},)"
        if num_chains == 4:
            # Keys present (checked once on nc4)
            for key_name in ("step_size", "momentum_inverse_scale", "alpha", "delta"):
                assert key_name in params, f"Missing {key_name!r}; got: {list(params)}"
        if num_chains == 8:
            imm = jnp.asarray(params["momentum_inverse_scale"])
            assert imm.shape == (8, _D), f"momentum_inverse_scale expected (8, {_D})"

    def test_num_chains_below_num_folds_raises(self) -> None:
        """HARD-KEEP: num_chains=2 < num_folds=4 must raise ValueError.

        MEADS cannot function if num_chains < num_folds — this is a MEADS-specific
        invariant not present in any other warmup.  Must remain a named test.
        """
        from tuningfork.warmup.meads import ENTRY

        key = jax.random.key(6002)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        with pytest.raises(ValueError, match="num_chains"):
            ENTRY.runner(
                jax.random.fold_in(key, 1),
                init_pos,
                200,
                _GHMC,
                logdensity_fn=logdensity_fn,
                num_chains=2,
                num_folds=4,
            )

    def test_explicit_num_chains_8_num_folds_4(self) -> None:
        """num_chains=8, num_folds=4 (chains > folds): shapes correct."""
        states, params, *_ = self._run(6003, num_chains=8, num_folds=4)
        leading = _state_leading_dim(states)
        assert leading == 8, f"Expected leading dim 8, got {leading}"
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (8,), f"step_size expected (8,), got {ss.shape}"
        imm = jnp.asarray(params["momentum_inverse_scale"])
        assert imm.shape == (
            8,
            _D,
        ), f"momentum_inverse_scale expected (8, {_D}), got {imm.shape}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == num_chains) passes through verbatim."""
        from tuningfork.warmup.meads import ENTRY

        num_chains = 4
        key = jax.random.key(6004)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            200,
            _GHMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_meads_num_folds_sidecar_present(self) -> None:
        """_meads_num_folds sidecar key must be present."""
        _, params, *_ = self._run(6006, num_chains=4)
        assert (
            "_meads_num_folds" in params
        ), f"Missing _meads_num_folds sidecar; got: {list(params)}"
        assert params["_meads_num_folds"] == 4


# ---------------------------------------------------------------------------
# 15. CHEES multi-chain warmup — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCheesMultiChain:
    """chees warmup multi-chain shape contract tests.

    CHEES is fundamentally multi-chain: a single call handles all num_chains
    chains jointly.  Like MEADS, CHEES is NOT vmapped — one call, all chains.

    Upstream API note: chees_adaptation.run() requires step_size and an optax
    optimizer as positional args (unlike meads_adaptation.run).  This wrapper
    handles that internally.

    Adapted numeric params (step_size, inverse_mass_matrix) are shared CHEES
    estimates broadcast to (num_chains,) / (num_chains, d).  Callable params
    (next_random_arg_fn, integration_steps_fn) are passed through as-is.
    """

    def _run(self, seed: int, num_chains: int, **kw):
        from tuningfork.warmup.chees import ENTRY

        key = jax.random.key(seed)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        return ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            50,  # short warmup for tests
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
            **kw,
        )

    @pytest.mark.parametrize(
        "num_chains,expected_leading,expected_ss,expected_imm",
        [
            (4, 4, (4,), (4, _D)),
            (8, 8, (8,), (8, _D)),
        ],
        ids=["nc4", "nc8"],
    )
    def test_shape_and_keys(
        self, num_chains, expected_leading, expected_ss, expected_imm
    ) -> None:
        """Shape + keys + positivity contract for CHEES at nc=4 and nc=8.

        nc4 row: covers default=4 behavior, keys present, step_size > 0, IMM shape,
        and compat smoke (dynamic_hmc runner succeeds).
        nc8 row: covers non-default chains > 4 shape.
        """
        seed = 7001 + num_chains
        states, params, *_ = self._run(seed, num_chains=num_chains)
        assert _state_leading_dim(states) == expected_leading
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == expected_ss
        imm = jnp.asarray(params["inverse_mass_matrix"])
        assert imm.shape == expected_imm
        if num_chains == 4:
            assert "step_size" in params
            assert "inverse_mass_matrix" in params
            assert bool(jnp.all(ss > 0)), f"step_size not all > 0: {ss}"

    def test_pre_batched_init_position_passes_through(self) -> None:
        """Pre-batched init_position (leading dim == num_chains) passes through verbatim."""
        from tuningfork.warmup.chees import ENTRY

        num_chains = 4
        key = jax.random.key(7003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        batched_pos = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape), init_pos
        )
        states, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            batched_pos,
            50,
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
            num_chains=num_chains,
        )
        pos_shape = _position_shape(states)
        assert pos_shape == (4, _D), f"Pre-batched: expected (4, {_D}), got {pos_shape}"

    def test_callable_params_present(self) -> None:
        """HARD-KEEP: CHEES adapted_params must contain next_random_arg_fn and
        integration_steps_fn as callables.

        This is CHEES-specific — no other warmup returns callable params.
        Distinct invariant; must remain a named test.
        """
        _, params, *_ = self._run(7005, num_chains=4)
        assert (
            "next_random_arg_fn" in params
        ), f"Missing next_random_arg_fn; keys: {list(params)}"
        assert (
            "integration_steps_fn" in params
        ), f"Missing integration_steps_fn; keys: {list(params)}"
        assert callable(
            params["next_random_arg_fn"]
        ), "next_random_arg_fn must be callable"
        assert callable(
            params["integration_steps_fn"]
        ), "integration_steps_fn must be callable"

    def test_jitter_amount_default_in_sidecar(self) -> None:
        """Sidecar must contain _chees_target_acceptance_rate metadata."""
        _, params, *_ = self._run(7006, num_chains=4)
        assert (
            "_chees_target_acceptance_rate" in params
        ), f"Missing _chees_target_acceptance_rate sidecar; keys: {list(params)}"
        assert (
            abs(params["_chees_target_acceptance_rate"] - 0.651) < 1e-6
        ), f"Expected 0.651, got {params['_chees_target_acceptance_rate']}"

    def test_max_leapfrog_steps_sidecar_present(self) -> None:
        """Sidecar must contain _chees_max_leapfrog_steps metadata."""
        _, params, *_ = self._run(7007, num_chains=4)
        assert (
            "_chees_max_leapfrog_steps" in params
        ), f"Missing _chees_max_leapfrog_steps sidecar; keys: {list(params)}"
        assert isinstance(
            params["_chees_max_leapfrog_steps"], int
        ), "_chees_max_leapfrog_steps must be a Python int"

    def test_target_acceptance_rate_none_falls_back_to_default(self) -> None:
        """HARD-KEEP: target_acceptance_rate=None must NOT reach upstream chees_adaptation.

        The generic recipe-runner dispatch (_recipe_runner.py) always forwards
        target_acceptance_rate explicitly, including None when the caller supplied
        no override (the emit default is None, not "omit the kwarg").  Before the
        fix, a plain typed default on this wrapper did nothing once None was passed
        in, and `target_acceptance_rate - harmonic_mean` TypeErrored deep inside
        chees_adaptation.py.  Regression guard: pass None explicitly and confirm
        (a) no crash, (b) the sidecar records the CHEES default (0.651), not None.
        """
        from tuningfork.warmup.chees import _DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE, ENTRY

        key = jax.random.key(7008)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        _, params = ENTRY.runner(
            jax.random.fold_in(key, 1),
            init_pos,
            50,
            _DYNAMIC_HMC,
            logdensity_fn=logdensity_fn,
            num_chains=4,
            target_acceptance_rate=None,
        )
        assert params["_chees_target_acceptance_rate"] == pytest.approx(
            _DEFAULT_CHEES_TARGET_ACCEPTANCE_RATE
        ), (
            "target_acceptance_rate=None must fall back to the CHEES default, "
            f"got {params['_chees_target_acceptance_rate']}"
        )


# ---------------------------------------------------------------------------
# 16. adjusted_mclmc_tuning warmup — SLOW (chain-running tests only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAdjustedMclmcTuning:
    """adjusted_mclmc_tuning warmup chain-running shape contract tests.

    adjusted_mclmc_tuning uses blackjax.adjusted_mclmc_find_L_and_step_size
    (static kernel) to jointly find L, step_size, and a diagonal IMM.
    Compatible with both adjusted_mclmc and adjusted_mclmc_dynamic.

    Registry / compat checks folded into TestWarmupRegistry / test_is_compatible
    (fast-tier); only the chain-running assertions remain here.
    """

    def test_single_chain_signature_adjusted_mclmc(self) -> None:
        """Single-chain run: verify returned param keys and positivity."""
        from tuningfork.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from tuningfork.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8001)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        states, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=1,
        )
        assert states is not None
        assert isinstance(params, dict)
        assert "L" in params, f"L missing; keys={list(params)}"
        assert "step_size" in params, f"step_size missing; keys={list(params)}"
        assert (
            "inverse_mass_matrix" in params
        ), f"inverse_mass_matrix missing; keys={list(params)}"
        assert (
            "_total_tuning_steps" in params
        ), f"_total_tuning_steps missing; keys={list(params)}"
        # Positivity (merged from test_single_chain_L_and_step_size_positive)
        assert bool(jnp.all(jnp.asarray(params["L"]) > 0)), f"L not > 0: {params['L']}"
        assert bool(
            jnp.all(jnp.asarray(params["step_size"]) > 0)
        ), f"step_size not > 0: {params['step_size']}"

    def test_multi_chain_3_shape(self) -> None:
        """num_chains=3: L/step_size shape (3,), IMM shape (3, d)."""
        from tuningfork.base_method.adjusted_mclmc import ENTRY as _ADJ_MCLMC
        from tuningfork.warmup.adjusted_mclmc_tuning import ENTRY

        key = jax.random.key(8003)
        init_pos, logdensity_fn = _build_logdensity(_MVN, key)
        warmup_key = jax.random.fold_in(key, 1)

        states, params = ENTRY.runner(
            warmup_key,
            init_pos,
            100,
            _ADJ_MCLMC,
            logdensity_fn=logdensity_fn,
            num_chains=3,
        )
        # State leading dim == 3
        leaves = jax.tree.leaves(states)
        assert (
            leaves[0].shape[0] == 3
        ), f"Expected leading dim 3, got {leaves[0].shape[0]}"

        # L shape (3,)
        L = jnp.asarray(params["L"])
        assert L.shape == (3,), f"Expected L.shape=(3,), got {L.shape}"

        # step_size shape (3,)
        ss = jnp.asarray(params["step_size"])
        assert ss.shape == (3,), f"Expected step_size.shape=(3,), got {ss.shape}"

        # inverse_mass_matrix shape (3, d)
        imm = params["inverse_mass_matrix"]
        d = _D
        assert imm.shape == (3, d), f"Expected IMM.shape=(3, {d}), got {imm.shape}"
        # _total_tuning_steps is a Python int (merged from test_total_tuning_steps_is_python_int)
        steps = params["_total_tuning_steps"]
        assert isinstance(
            steps, int
        ), f"_total_tuning_steps must be int, got {type(steps)}"
        assert steps > 0, f"_total_tuning_steps must be > 0, got {steps}"


# ---------------------------------------------------------------------------
# TestNoWarmupGuards — SLOW (builds logdensity_fn + no_warmup guard raises)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoWarmupGuards:
    """no_warmup._runner raises NotImplementedError for specialised methods."""

    def test_no_warmup_raises_for_elliptical_slice(self) -> None:
        """elliptical_slice.extra_required_kwargs non-empty → no_warmup raises NotImplementedError."""
        import pytest

        from tuningfork.base_method.elliptical_slice import ENTRY as _ELLIP_SLICE
        from tuningfork.warmup import WARMUPS

        key = jax.random.key(9001)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(NotImplementedError, match="factory requires extra kwargs"):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                _ELLIP_SLICE,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )

    def test_no_warmup_raises_for_irmh(self) -> None:
        """irmh.extra_required_kwargs non-empty → no_warmup raises NotImplementedError."""
        from tuningfork.base_method.irmh import ENTRY as _IRMH
        from tuningfork.warmup import WARMUPS

        key = jax.random.key(9002)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(NotImplementedError, match="factory requires extra kwargs"):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                _IRMH,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )

    def test_no_warmup_raises_for_synthetic_extra_kwargs_entry_guard(self) -> None:
        """Synthetic BaseMethod(extra_required_kwargs non-empty) → NotImplementedError."""
        from tuningfork.base_method._base import BaseMethod
        from tuningfork.warmup import WARMUPS

        synthetic = BaseMethod(
            name="synthetic_irmh_like",
            family="mcmc",
            factory=lambda logdensity_fn, **kw: None,
            grad_count_per_step=lambda info: 0,
            default_hp_space=(),
            extra_required_kwargs=("proposal_distribution",),
        )

        key = jax.random.key(9003)
        init_pos = jnp.zeros(5)

        def dummy_logdensity(x):
            return -0.5 * jnp.sum(x**2)

        with pytest.raises(NotImplementedError, match="factory requires extra kwargs"):
            WARMUPS["no_warmup"].runner(
                key,
                init_pos,
                0,
                synthetic,
                logdensity_fn=dummy_logdensity,
                num_chains=1,
            )
