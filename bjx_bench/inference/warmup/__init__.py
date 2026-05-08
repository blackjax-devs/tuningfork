"""bjx-bench warmup registry.

Phase 3 (P3.1) lands three core warmups:
- ``stan_window``: blackjax.window_adaptation; compatible with hmc, nuts,
  barker, mala.
- ``mclmc_tuning``: blackjax.mclmc_find_L_and_step_size; compatible with
  mclmc only.
- ``no_warmup``: identity warmup returning default init state + empty params;
  compatible with all algorithms (sentinel ``"*"``).

MEADS / ChEES / Pathfinder warmups land in Phase 5 alongside the VI
integration.

Usage::

    from bjx_bench.inference.warmup import WARMUPS, Warmup

    warmup = WARMUPS["stan_window"]
    state, params = warmup.runner(rng_key, position, n_warmup, base_method,
                                  logdensity_fn=logdensity_fn)
"""

from bjx_bench.inference.warmup._base import Warmup
from bjx_bench.inference.warmup.mclmc_tuning import ENTRY as _mclmc_tuning
from bjx_bench.inference.warmup.no_warmup import ENTRY as _no_warmup
from bjx_bench.inference.warmup.stan_window import ENTRY as _stan_window

WARMUPS: dict[str, Warmup] = {
    _stan_window.name: _stan_window,
    _mclmc_tuning.name: _mclmc_tuning,
    _no_warmup.name: _no_warmup,
}

__all__ = ["WARMUPS", "Warmup"]
