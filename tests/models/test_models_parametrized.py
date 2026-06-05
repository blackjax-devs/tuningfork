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
"""Parametrized tests for model registration and basic schema.

Consolidates identical registration/schema pattern from 11 per-model test files.
Per-model analytic moment tests (model correctness) are preserved separately in
test_posterior_entry.py.
"""

import pytest

from tuningfork.model import MODELS

pytestmark = pytest.mark.fast

# All 14 models in the test suite
_MODELS = sorted(MODELS.keys())


# ===========================================================================
# Parametrized Registration tests
# ===========================================================================


@pytest.mark.parametrize("model_name", _MODELS)
def test_model_registered(model_name: str) -> None:
    """Model is registered in MODELS."""
    assert (
        model_name in MODELS
    ), f"Model '{model_name}' not found in MODELS; registered: {sorted(MODELS)}"


@pytest.mark.parametrize("model_name", _MODELS)
def test_model_has_dim_and_class(model_name: str) -> None:
    """ENTRY has dim (positive int) and class_ (non-empty string label)."""
    entry = MODELS[model_name]
    assert isinstance(entry.dim, int) and entry.dim > 0
    assert (
        isinstance(entry.class_, str) and entry.class_
    ), f"Model '{model_name}' must have a non-empty string class_; got {entry.class_!r}"
