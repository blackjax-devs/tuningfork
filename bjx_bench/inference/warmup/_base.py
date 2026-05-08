"""Warmup procedures.

Each Warmup wraps a BlackJAX adaptation routine into a uniform shape:
``runner(rng_key, init_position, n_warmup, base_method, **kwargs) -> (state, params)``.

Phase 2.5 lands only the Warmup dataclass and an empty WARMUPS dict so
the inference/ namespace is in place. Real wrapper modules
(stan_window, mclmc_tuning, no_warmup) land in Phase 3 alongside the
Tier-C warmup-isolation runs. The existing
``bjx_bench/calibration/tier_b.py:_run_warmup`` retains its current
inline dispatch; refactoring it to use WARMUPS is a Phase 3 task.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Warmup:
    """A warmup procedure: produces ``(state, params)`` before sampling."""

    name: str
    runner: Callable[..., tuple[Any, dict]]
    compatible_methods: tuple[str, ...]
    notes: str = ""
