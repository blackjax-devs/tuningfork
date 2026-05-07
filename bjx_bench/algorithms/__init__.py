"""bjx-bench algorithm registry.

``ALGORITHMS`` maps algorithm name strings to ``AlgorithmEntry`` instances.
The runner, Optuna BO loop, and CLI all iterate over this dict to discover
available algorithms without hard-coding names.

Adding a new algorithm
----------------------
1. Create ``bjx_bench/algorithms/<name>.py`` with an ``ENTRY`` module-level
   variable of type ``AlgorithmEntry``.
2. Import it here and add it to ``ALGORITHMS``.
"""

from __future__ import annotations

from bjx_bench.algorithms._base import AlgorithmEntry, HyperparamSpace
from bjx_bench.algorithms.barker import ENTRY as _barker_entry
from bjx_bench.algorithms.hmc import ENTRY as _hmc_entry
from bjx_bench.algorithms.mala import ENTRY as _mala_entry
from bjx_bench.algorithms.nuts import ENTRY as _nuts_entry
from bjx_bench.algorithms.rwm import ENTRY as _rwm_entry

ALGORITHMS: dict[str, AlgorithmEntry] = {
    _hmc_entry.name: _hmc_entry,
    _nuts_entry.name: _nuts_entry,
    _mala_entry.name: _mala_entry,
    _barker_entry.name: _barker_entry,
    _rwm_entry.name: _rwm_entry,
}

__all__ = ["AlgorithmEntry", "HyperparamSpace", "ALGORITHMS"]
