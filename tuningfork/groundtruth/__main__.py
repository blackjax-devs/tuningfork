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
"""Entry point for ``python -m tuningfork.groundtruth``.

Sets ``JAX_ENABLE_X64=1`` from the ``GT_X64`` environment variable before any
JAX import so that models requiring 64-bit floats (``gp_regression``,
``lotka_volterra``) work correctly.  Must happen here, at process start, before
tuningfork.model (which imports JAX) is loaded.
"""

import os

if os.environ.get("GT_X64") == "1":
    os.environ.setdefault("JAX_ENABLE_X64", "1")

from tuningfork.groundtruth._cli import main  # noqa: E402

if __name__ == "__main__":
    main()
