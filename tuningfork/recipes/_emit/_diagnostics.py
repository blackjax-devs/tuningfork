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

"""Emit the optional tap-diagnostics wrapper for generated programs."""

from __future__ import annotations

from typing import Any

_TAP_ENV = "TUNINGFORK_TAP_DIAGNOSTICS"


def emit_diagnostics(ctx: dict[str, Any]) -> str:
    """Return standalone source that lazily enters the tap context.

    The generated source remains inert unless ``TUNINGFORK_TAP_DIAGNOSTICS``
    is enabled.  Compatibility checks and tap wiring stay in the canonical
    diagnostics module; this block only supplies the recipe-specific policy.
    """
    model = ctx["model_name"]
    method = ctx["base_method_name"]
    seed = ctx["sampler_seed"]
    max_doublings = ctx.get("max_num_doublings", 10)
    skipped_message = (
        f"[tuningfork tap] tap diagnostics skipped for {method!r}: "
        "incompatible algorithm"
    )
    return "\n".join(
        [
            "import atexit as _tap_atexit",
            "import contextlib as _tap_contextlib",
            "import logging as _tap_logging",
            "import os as _tap_os",
            "_tap_stack = _tap_contextlib.ExitStack()",
            "_tap_atexit.register(_tap_stack.close)",
            "if _tap_os.environ.get('TUNINGFORK_TAP_DIAGNOSTICS', '0') not in {'', '0'}:",
            "    from tuningfork.diagnostics._tap import is_algorithm_tap_compatible as _tap_compatible",
            "    from tuningfork.diagnostics._tap import tap_diagnostics_context as _tap_context",
            f"    if _tap_compatible({method!r}):",
            f"        _tap_stack.enter_context(_tap_context(run_tag={model!r} + '__' + {method!r} + '__seed' + str({seed}), base_method_name={method!r}, max_num_doublings={max_doublings!r}))",
            "    else:",
            f"        _tap_logging.getLogger(__name__).warning({skipped_message!r})",
            "",
        ]
    )


def emit_diagnostics_close() -> str:
    """Return the explicit close line to place after sampling synchronizes."""
    return "_tap_stack.close()"


__all__ = ["emit_diagnostics", "emit_diagnostics_close"]
