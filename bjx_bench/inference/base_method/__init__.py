"""bjx-bench algorithm registry.

``BASE_METHODS`` maps algorithm name strings to ``BaseMethod`` instances.
The runner, Optuna BO loop, and CLI all iterate over this dict to discover
available algorithms without hard-coding names.

Adding a new algorithm
----------------------
1. Create ``bjx_bench/algorithms/<name>.py`` with an ``ENTRY`` module-level
   variable of type ``BaseMethod``.
2. Import it here and add it to ``BASE_METHODS``.
"""

from __future__ import annotations

from bjx_bench.inference.base_method._base import BaseMethod, HyperparamSpace
from bjx_bench.inference.base_method.barker import ENTRY as _barker_entry
from bjx_bench.inference.base_method.hmc import ENTRY as _hmc_entry
from bjx_bench.inference.base_method.mala import ENTRY as _mala_entry
from bjx_bench.inference.base_method.mclmc import ENTRY as _mclmc_entry
from bjx_bench.inference.base_method.nuts import ENTRY as _nuts_entry
from bjx_bench.inference.base_method.rwm import ENTRY as _rwm_entry

BASE_METHODS: dict[str, BaseMethod] = {
    _hmc_entry.name: _hmc_entry,
    _nuts_entry.name: _nuts_entry,
    _mala_entry.name: _mala_entry,
    _barker_entry.name: _barker_entry,
    _rwm_entry.name: _rwm_entry,
    _mclmc_entry.name: _mclmc_entry,
}

__all__ = ["BaseMethod", "HyperparamSpace", "BASE_METHODS"]
