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
"""Binding checks for generated evidence used by recipe certification."""

from __future__ import annotations

from collections.abc import Mapping

from tuningfork.catalog.emit import RECIPE_EVIDENCE_KEY, canonical_recipe_snapshot
from tuningfork.recipes._certification_intent import CertificationIntent
from tuningfork.recipes._execution_plan import ExecutionOverrides
from tuningfork.recipes._ground_truth_reference import GroundTruthReference
from tuningfork.recipes._launcher import LaunchResult
from tuningfork.recipes._resolve_execution_plan import resolve_execution_plan


def verify_launch_binding(
    result: LaunchResult,
    intent: CertificationIntent,
    reference: GroundTruthReference,
    *,
    n_warmup: int,
    n_samples: int,
    num_chains: int,
    seed: int,
    progress_bar: bool,
) -> None:
    """Reject evidence not bound to this exact intent, plan, and GT object."""
    expected = resolve_execution_plan(
        intent.recipe,
        ExecutionOverrides(
            tuning_seed=seed,
            num_samples=n_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            num_warmup=n_warmup,
            warmup_num_chains=intent.recipe.warmup_num_chains,
        ),
    )
    if (
        result.manifest.plan_hash != expected.plan_hash
        or result.manifest.executable_config_hash != expected.executable_config_hash
        or result.manifest.recipe_ref != expected.recipe_ref
    ):
        raise ValueError(
            "generated receipt does not match the requested execution plan"
        )
    if result.receipt.manifest != result.manifest:
        raise ValueError("launch result and receipt contain different manifests")

    # ExecutionReceipt freezes JSON arrays to tuples in its immutable in-memory
    # view.  Compare the public, thawed receipt representation that was hashed
    # and persisted, otherwise a valid list-valued recipe snapshot compares
    # unequal solely because of the storage wrapper.
    identity = result.receipt.as_dict().get("reference_identity")
    envelope = (
        identity.get(RECIPE_EVIDENCE_KEY) if isinstance(identity, Mapping) else None
    )
    if not isinstance(envelope, Mapping):
        raise ValueError("receipt lacks the recipe evidence envelope")
    expected_snapshot = canonical_recipe_snapshot(intent.recipe)
    if envelope.get("snapshot") != expected_snapshot:
        raise ValueError("receipt recipe snapshot does not match certification intent")
    if envelope.get("caller_reference_identity") != reference.identity:
        raise ValueError(
            "receipt ground-truth identity does not match evaluation input"
        )
