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
"""Machine-info snapshot for recipe and groundtruth metadata provenance.

Captured at recipe-write time so that wall-time comparisons across machines
are interpretable: a recipe stamped with ``machine_info`` can be compared
against a re-run on a different host with known hardware differences.

The snapshot is intentionally lightweight — no network calls, no slow probes,
all values from stdlib or already-imported packages.  Returns a plain dict
so it round-trips through ``json.dumps`` without a custom encoder.

Usage::

    from tuningfork._machine_info import get_machine_info
    info = get_machine_info()
    # {"cpu_model": "...", "cpu_count_logical": 8, "os": "...", ...}
"""

from __future__ import annotations

import platform
import sys
from typing import Any

__all__ = ["get_machine_info"]


def get_machine_info() -> dict[str, Any]:
    """Return a lightweight hardware + software snapshot as a plain dict.

    Fields
    ------
    cpu_model : str
        Processor model string from ``platform.processor()``, e.g.
        ``"x86_64"`` on Linux or ``"Intel(R) Core(TM) i7-..."`` on macOS.
        May be an empty string on some Linux builds — fall back to
        ``platform.machine()`` in that case.
    cpu_count_logical : int | None
        Number of logical CPUs (hyperthreaded cores) via ``os.cpu_count()``.
        None if undetermined.
    os : str
        ``"Linux"`` / ``"macOS"`` / ``"Windows"`` / ``"<platform.system()>"``
        with the kernel release appended, e.g. ``"Linux 5.15.0-..."``
    python_version : str
        ``sys.version_info`` as ``"3.13.2"``.
    jax_version : str
        Installed JAX version string; ``"unavailable"`` if JAX not importable.
    blackjax_version : str
        Installed BlackJAX version string; ``"unavailable"`` if not importable.
    jax_x64_enabled : bool | None
        ``jax.config.x64_enabled`` at call time; None if JAX not importable.
    jax_default_device : str | None
        ``str(jax.default_backend())`` at call time, e.g. ``"cpu"`` or ``"gpu"``;
        None if JAX not importable.
    gpu_info : str | None
        Short GPU description if a CUDA/ROCm device is visible; ``None`` on
        CPU-only builds.  Obtained cheaply via ``jax.devices("gpu")`` — no
        subprocess or nvidia-smi call.

    Returns
    -------
    dict[str, Any]
        All values are JSON-serialisable (str, int, bool, or None).
    """
    import os

    cpu_model = platform.processor() or platform.machine()
    cpu_count = os.cpu_count()

    os_name = platform.system()
    os_release = platform.release()
    os_str = f"{os_name} {os_release}".strip()

    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    jax_version: str = "unavailable"
    blackjax_version: str = "unavailable"
    x64_enabled: bool | None = None
    default_device: str | None = None
    gpu_info: str | None = None

    try:
        import jax

        jax_version = jax.__version__
        x64_enabled = bool(jax.config.x64_enabled)
        default_device = str(jax.default_backend())

        # Probe GPU without causing a full JAX init if GPU is absent.
        # jax.devices("gpu") returns [] on CPU-only builds — no error.
        try:
            gpu_devs = jax.devices("gpu")
            if gpu_devs:
                gpu_info = str(gpu_devs[0])
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    try:
        import blackjax

        blackjax_version = getattr(blackjax, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_count_logical": cpu_count,
        "os": os_str,
        "python_version": py_ver,
        "jax_version": jax_version,
        "blackjax_version": blackjax_version,
        "jax_x64_enabled": x64_enabled,
        "jax_default_device": default_device,
        "gpu_info": gpu_info,
    }
