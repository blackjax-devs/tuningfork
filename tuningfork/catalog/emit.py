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
"""Public wrapper for recipe-to-standalone-script code-gen.

The implementation lives in ``tuningfork.recipes._emit_script``. This module
re-exports the function for user-facing convenience under
``tuningfork.catalog.emit``.

Example usage::

    from tuningfork.catalog import load_recipe, emit_script
    from pathlib import Path

    recipe = load_recipe("tuningfork/catalog/eight_schools_ncp/groundtruth.json")
    script = emit_script(recipe, num_samples=500)
    Path("reproduce_eight_schools.py").write_text(script)
    # Then: python reproduce_eight_schools.py

The emitted script has zero ``import tuningfork`` in its inference path
(locked decision D8 STRICT).  The caller writes the returned string to
whatever location they prefer (locked decision D9: pure return-string).
"""

from tuningfork.recipes._emit_script import emit_script

__all__ = ["emit_script"]
