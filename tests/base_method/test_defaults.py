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

"""Pure default-value helpers for sampler hyperparameter spaces."""

import pytest

from tuningfork.base_method import (
    BASE_METHODS,
    HyperparamSpace,
    default_params_for,
    default_value_for_space,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        (
            HyperparamSpace("x", "loguniform", 1e-3, 1e1),
            pytest.approx(0.6309573444801934),
        ),
        (HyperparamSpace("x", "uniform", 2.0, 8.0), 5.0),
        (HyperparamSpace("x", "int", 1, 9), 5),
        (HyperparamSpace("x", "categorical", choices=("a", "b")), "a"),
    ],
)
def test_default_value_for_space(space: HyperparamSpace, expected) -> None:
    assert default_value_for_space(space) == expected


def test_default_params_for_registry_entry() -> None:
    params = default_params_for(BASE_METHODS["nuts"])
    assert set(params) == {
        space.name for space in BASE_METHODS["nuts"].default_hp_space
    }
    assert params["step_size"] == pytest.approx(0.12589254117941667)
