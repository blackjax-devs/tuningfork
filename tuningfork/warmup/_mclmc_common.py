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
"""Shared unpacking logic for MCLMC adaptation state."""

from typing import Any

import jax
import jax.numpy as jnp


def _unpack_mclmc_adaptation(
    states: Any,
    adaptation_states: Any,
    total_tuning_steps_per_chain: Any,
) -> tuple[Any, dict[str, Any]]:
    """Unpack and finalize MCLMC/adjusted-MCLMC adaptation state.

    Parameters
    ----------
    states
        Post-adaptation states (batched, shape (num_chains, ...)).
    adaptation_states
        Adaptation state objects from the tuning routine (batched).
    total_tuning_steps_per_chain
        Per-chain total tuning steps (from vmapped tuning output).

    Returns
    -------
    states
        The post-adaptation states (pass-through).
    adapted
        Dict with keys:
        - "L": (num_chains,) adapted trajectory lengths
        - "step_size": (num_chains,) adapted step sizes
        - "inverse_mass_matrix": (num_chains, d) adapted preconditioners
        - "_total_tuning_steps": int — total gradient evals (summed across chains)
    """
    # SYNC: block until vmapped tuning completes before host-materialising the
    # step count.  Without this, int() goes through the buffer protocol on an
    # unsynced JAX future — same deadlock risk as the calibration/ subtree.
    jax.block_until_ready((states, adaptation_states, total_tuning_steps_per_chain))
    # total_tuning_steps is the same for all chains (same num_steps).
    # Take the value from chain 0 and convert to Python int.
    total_tuning_steps = int(jnp.asarray(total_tuning_steps_per_chain)[0])

    # MCLMCAdaptationState._fields = ('L', 'step_size', 'inverse_mass_matrix')
    adapted: dict[str, Any] = {
        "L": adaptation_states.L,  # shape (num_chains,)
        "step_size": adaptation_states.step_size,  # shape (num_chains,)
        "inverse_mass_matrix": adaptation_states.inverse_mass_matrix,  # (num_chains, d)
        # will fold this into Recipe.calibration_budget
        "_total_tuning_steps": total_tuning_steps,
    }
    return states, adapted


__all__ = ["_unpack_mclmc_adaptation"]
