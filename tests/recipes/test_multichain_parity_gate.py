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
"""Tests for the two-part statistical-equivalence multichain parity gate.

Covers the gate function from tuningfork.calibration.multichain_parity_gate.

Test catalogue
--------------
1. test_correct_keys_vmap_ok
   Correct per-chain keys fed to both loop and vmap → VMAP_OK.
   Structural sample-1 |d| ≈ 1.2e-14 (machine-eps scale; < 1e-10 threshold).

2. test_injected_bug_shift_block_bug
   Off-by-one bug: vmap chain-0 slot fed chain-1's key (internal bug).
   Structural sample-1 |d| ≈ 3.63 (O(1) → BLOCK_BUG).

3. test_injected_bug_transpose_block_bug
   Reversed-keys bug: vmap keys reversed (internal bug).
   Structural sample-1 |d| ≈ 2.14 (O(1) → BLOCK_BUG).

4. test_key_identity_assertion_fires
   Keys passed to the gate (keys_loop vs keys_stacked) do NOT match →
   AssertionError from the key-identity assertion (guards against a stacking bug
   in the caller before sampling even starts).

5. test_parity_result_to_dict
   ParityGateResult.to_dict() returns expected keys for JSON serialisation.

6. test_statistical_equivalence_part_b
   After struct_ok passes, part (B) metrics (ks_D, ks_p, acc, nsteps, div)
   are populated in the result.

Reference numbers (pre-validated by statistician via diag_injected_bug.py,
banana adjusted_mclmc_dynamic float64, seed 101):
  correct   sample-1 |d| ≈ 1.199e-14 → VMAP_OK
  shift     sample-1 |d| ≈ 3.634     → BLOCK_BUG
  transpose sample-1 |d| ≈ 2.141     → BLOCK_BUG

Gotchas (statistician memo):
  - measure sample-1 ONLY (sample index 0), not full trajectory
  - inject at the KEY level (wrong key to vmap chain slot), not at positions
  - the gate's key-identity assertion checks keys_loop vs keys_stacked;
    inject the bug INSIDE sample_vmap (wrong internal key), NOT in keys_stacked,
    so that (A) the key assertion passes and (B) the structural check fires
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Banana config (must match diag_injected_bug.py exactly for ref-number repro)
_CH = 4
_AVG = 18
_BANANA_VAR = np.array([8.0, 9.0])
_SEED = 101

# Pre-validated step_size (from diag_injected_bug_results.json, seed 101)
_REF_STEP = 0.19870453313343764


def _load_banana_ld():
    """Return (logdensity_fn, init_pos, imm_diag)."""
    from tuningfork.model._numpyro import build_logdensity_fn
    from tuningfork.model._registry import MODELS

    # x64 must be enabled; tests/conftest.py should handle this but ensure here.
    jax.config.update("jax_enable_x64", True)

    entry = MODELS["banana"]
    init_dict, ld_raw, _ = build_logdensity_fn(jax.random.key(7), entry)
    flat_init, unravel = ravel_pytree(init_dict)
    ld = lambda xf: ld_raw(unravel(xf))
    imm = jnp.asarray(_BANANA_VAR, dtype=jnp.float64)
    return ld, jnp.zeros(2, dtype=jnp.float64), imm


def _make_dyn_kernel(imm):
    """Build an adjusted_mclmc_dynamic low-level kernel with fixed IMM."""
    import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod
    from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

    _steps_fn = make_random_trajectory_length_fn(True)
    base_kernel = adj_dyn_mod.build_kernel(integration_steps_fn=_steps_fn)

    def kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        L_proposal_factor,
        inverse_mass_matrix,
        integration_steps_params,
    ):
        return base_kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            L_proposal_factor=L_proposal_factor,
            inverse_mass_matrix=imm,  # always fixed
            integration_steps_params=integration_steps_params,
        )

    return kernel


def _chain_keys(seed, ch):
    """Per-chain init keys: key(seed * 1000 + ci + 1) — matches diag_injected_bug.py."""
    return [jax.random.key(seed * 1000 + ci + 1) for ci in range(ch)]


def _scan_chain(ld, init, imm, dyn, step, avg, sk, n):
    """Scan a single chain for n steps; return flat positions (n, d)."""
    import blackjax.mcmc.adjusted_mclmc_dynamic as adj_dyn_mod

    s = adj_dyn_mod.init(init, ld, sk)

    def stp(c, key):
        nx, info = dyn(
            rng_key=key,
            state=c,
            logdensity_fn=ld,
            step_size=step,
            L_proposal_factor=jnp.inf,
            inverse_mass_matrix=imm,
            integration_steps_params=(float(avg),),
        )
        return nx, (
            nx.position,
            info.is_divergent,
            info.acceptance_rate,
            info.num_integration_steps,
        )

    _, (pt, dv, ac, ns) = jax.lax.scan(stp, s, jax.random.split(sk, n))
    flat = jax.vmap(lambda q: ravel_pytree(q)[0])(pt)
    # Return JAX arrays (not np.array) — safe inside jax.vmap contexts.
    # Callers do the eager np.asarray conversion at their level.
    return flat, dv, ac, ns


def _build_sample_functions(
    ld, init, imm, step, avg, n, ch, seed, *, vmap_key_mode="correct"
):
    """Return (sample_loop, sample_vmap, keys_loop, keys_stacked).

    Bugs are injected INSIDE sample_vmap via vmap_key_mode:
      "correct"   — correct keys everywhere
      "shift"     — chain-0 slot gets chain-1's key (off-by-one)
      "transpose" — reversed keys

    keys_loop and keys_stacked ALWAYS carry the correct per-chain keys so
    that the gate's key-identity assertion passes regardless of vmap_key_mode.
    The bug is in how sample_vmap APPLIES the keys internally.
    """
    dyn = _make_dyn_kernel(imm)
    correct_keys = _chain_keys(seed, ch)

    # Keys exposed to the gate (always correct, for key-identity assertion)
    keys_loop_list = correct_keys
    keys_stacked_arr = jnp.stack(correct_keys)

    # Internal keys for the vmap path (may be bugged)
    if vmap_key_mode == "correct":
        internal_keys = correct_keys
    elif vmap_key_mode == "shift":
        # chain-0 gets chain-1's key (off-by-one injection)
        internal_keys = correct_keys[1:] + correct_keys[:1]
    elif vmap_key_mode == "transpose":
        # reversed
        internal_keys = correct_keys[::-1]
    else:
        raise ValueError(f"Unknown vmap_key_mode: {vmap_key_mode!r}")

    def sample_loop(probe_n):
        pos_list, dv_list, ac_list, ns_list = [], [], [], []
        for sk in correct_keys:
            p, dv, ac, ns = _scan_chain(ld, init, imm, dyn, step, avg, sk, probe_n)
            pos_list.append(p)
            dv_list.append(dv)
            ac_list.append(ac)
            ns_list.append(ns)
        pos = np.stack(pos_list, axis=0)  # (ch, probe_n, d)
        return (
            pos,
            float(np.mean(np.concatenate(dv_list))),
            float(np.mean(np.concatenate(ac_list))),
            float(np.mean(np.concatenate(ns_list))),
        )

    def sample_vmap(probe_n):
        int_keys = jnp.stack(internal_keys)

        def run_one(sk):
            p, dv, ac, ns = _scan_chain(ld, init, imm, dyn, step, avg, sk, probe_n)
            return p, dv, ac, ns

        flats, dvs, acs, nss = jax.vmap(run_one)(int_keys)
        pos = np.asarray(flats)  # (ch, probe_n, d)
        return (
            pos,
            float(np.mean(np.asarray(dvs))),
            float(np.mean(np.asarray(acs))),
            float(np.mean(np.asarray(nss))),
        )

    return sample_loop, sample_vmap, keys_loop_list, keys_stacked_arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def banana_setup():
    """Shared banana model + kernel, loaded once per module."""
    jax.config.update("jax_enable_x64", True)
    ld, init, imm = _load_banana_ld()
    return ld, init, imm


class TestMultichainParityGate:
    """Parity gate verdict tests using banana adjusted_mclmc_dynamic."""

    def test_correct_keys_vmap_ok(self, banana_setup):
        """Correct keys: struct sample-1 |d| ≈ 1.2e-14 → VMAP_OK."""
        from tuningfork.calibration.multichain_parity_gate import (
            VMAP_OK,
            multichain_parity_gate,
        )

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="correct",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)

        assert result.verdict == VMAP_OK, (
            f"Expected VMAP_OK, got {result.verdict!r} "
            f"(struct_sample1_abs={result.struct_sample1_abs:.3e})"
        )
        assert result.struct_ok is True
        # Reference: ~1.2e-14 (machine-eps scale, float64)
        assert (
            result.struct_sample1_abs < 1e-10
        ), f"Correct-keys struct check should be machine-eps; got {result.struct_sample1_abs:.3e}"
        # Broad check: close to the statistician's pre-validated value
        assert result.struct_sample1_abs < 1e-12, (
            f"Expected struct_sample1_abs near 1.2e-14; got {result.struct_sample1_abs:.3e} — "
            "possible JAX version mismatch"
        )

    def test_injected_bug_shift_block_bug(self, banana_setup):
        """Off-by-one bug: vmap chain-0 gets chain-1's key → BLOCK_BUG.

        Reference: struct_sample1_abs ≈ 3.63 (O(1) signal at sample 1).
        Bug is injected INSIDE sample_vmap (internal key shuffle);
        keys_stacked passed to the gate are always correct so the key-identity
        assertion passes — the structural check is what catches the defect.
        """
        from tuningfork.calibration.multichain_parity_gate import (
            BLOCK_BUG,
            multichain_parity_gate,
        )

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="shift",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)

        assert result.verdict == BLOCK_BUG, (
            f"Expected BLOCK_BUG for off-by-one key injection, got {result.verdict!r} "
            f"(struct_sample1_abs={result.struct_sample1_abs:.3e})"
        )
        assert result.struct_ok is False
        # Reference: ≈ 3.63 — must be O(1)
        assert (
            result.struct_sample1_abs > 1.0
        ), f"Expected O(1) struct signal for shift bug; got {result.struct_sample1_abs:.3e}"
        # Broad check against the statistician's reference number (allow ±20%)
        assert (
            abs(result.struct_sample1_abs - 3.634) < 0.8
        ), f"Expected struct_sample1_abs ≈ 3.63; got {result.struct_sample1_abs:.4f}"

    def test_injected_bug_transpose_block_bug(self, banana_setup):
        """Reversed keys: vmap uses reversed per-chain keys → BLOCK_BUG.

        Reference: struct_sample1_abs ≈ 2.14 (O(1) signal at sample 1).
        """
        from tuningfork.calibration.multichain_parity_gate import (
            BLOCK_BUG,
            multichain_parity_gate,
        )

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="transpose",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)

        assert result.verdict == BLOCK_BUG, (
            f"Expected BLOCK_BUG for reversed-keys injection, got {result.verdict!r} "
            f"(struct_sample1_abs={result.struct_sample1_abs:.3e})"
        )
        assert result.struct_ok is False
        # Reference: ≈ 2.14 — must be O(1)
        assert (
            result.struct_sample1_abs > 1.0
        ), f"Expected O(1) struct signal for transpose bug; got {result.struct_sample1_abs:.3e}"
        # Broad check against the statistician's reference number (allow ±20%)
        assert (
            abs(result.struct_sample1_abs - 2.141) < 0.5
        ), f"Expected struct_sample1_abs ≈ 2.14; got {result.struct_sample1_abs:.4f}"

    def test_key_identity_assertion_fires(self, banana_setup):
        """Mismatched keys_loop vs keys_stacked → AssertionError from key-identity check.

        This guards against a caller-side stacking bug (e.g. wrong axis, wrong order).
        The assertion fires BEFORE any sampling occurs.
        """
        from tuningfork.calibration.multichain_parity_gate import multichain_parity_gate

        ld, init, imm = banana_setup

        # Build correct sample functions
        sl, sv, kl_correct, ks_correct = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="correct",
        )

        # Pass reversed stacked keys to the gate → mismatch with keys_loop
        wrong_stacked = jnp.stack(_chain_keys(_SEED, _CH)[::-1])  # reversed

        with pytest.raises(AssertionError, match="key-identity assertion failed"):
            multichain_parity_gate(sl, sv, kl_correct, wrong_stacked, probe_n=50)

    def test_stat_ok_populated_on_vmap_ok(self, banana_setup):
        """When (A) passes, part (B) metrics (ks_D, ks_p, acc, nsteps, div) are populated."""
        from tuningfork.calibration.multichain_parity_gate import multichain_parity_gate

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="correct",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)

        assert result.stat_ok is not None
        assert result.ks_D is not None and result.ks_D >= 0
        assert result.ks_p is not None and 0 <= result.ks_p <= 1
        assert result.acc_abs is not None and result.acc_abs >= 0
        assert result.nsteps_abs is not None and result.nsteps_abs >= 0
        assert result.div_abs is not None and result.div_abs >= 0
        assert result.full_traj_max_abs is not None

    def test_stat_none_on_block_bug(self, banana_setup):
        """When (A) fails → BLOCK_BUG and stat_ok/ks fields are None (skipped)."""
        from tuningfork.calibration.multichain_parity_gate import (
            BLOCK_BUG,
            multichain_parity_gate,
        )

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="shift",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)

        assert result.verdict == BLOCK_BUG
        assert result.stat_ok is None
        assert result.ks_D is None
        assert result.ks_p is None

    def test_parity_result_to_dict_keys(self, banana_setup):
        """ParityGateResult.to_dict() returns all expected keys for JSON storage."""
        from tuningfork.calibration.multichain_parity_gate import multichain_parity_gate

        ld, init, imm = banana_setup
        sl, sv, kl, ks = _build_sample_functions(
            ld,
            init,
            imm,
            _REF_STEP,
            _AVG,
            n=50,
            ch=_CH,
            seed=_SEED,
            vmap_key_mode="correct",
        )
        result = multichain_parity_gate(sl, sv, kl, ks, probe_n=50)
        d = result.to_dict()

        expected_keys = {
            "verdict",
            "struct_ok",
            "struct_sample1_abs",
            "stat_ok",
            "ks_D",
            "ks_p",
            "acc_abs",
            "acc_threshold",
            "nsteps_abs",
            "div_abs",
            "full_traj_max_abs",
        }
        assert set(d.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(d.keys())}; "
            f"Extra keys: {set(d.keys()) - expected_keys}"
        )
        # All values must be JSON-serialisable (no JAX arrays)
        import json

        json.dumps(d)  # raises TypeError if non-serialisable
