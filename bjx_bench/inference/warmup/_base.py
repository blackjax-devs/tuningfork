"""Warmup procedures.

Each Warmup wraps a BlackJAX adaptation routine into a uniform shape:
``runner(rng_key, init_position, n_warmup, base_method, *, logdensity_fn, **kwargs)
-> (state, params)``.

Phase 2.5 landed the Warmup dataclass stub.  Phase 3 (P3.1) adds the
real wrapper modules (stan_window, mclmc_tuning, no_warmup) and the
``is_compatible`` method so the registry can guard against mismatched
pairings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Warmup:
    """A warmup procedure: produces ``(state, params)`` before sampling.

    Parameters
    ----------
    name
        Unique registry key, e.g. ``"stan_window"``, ``"mclmc_tuning"``,
        ``"no_warmup"``.
    runner
        Callable with signature::

            runner(rng_key, init_position, n_warmup, base_method,
                   *, logdensity_fn, **kwargs) -> (state, dict)

        Returns the post-warmup kernel state and a dict of adapted
        parameters (e.g. ``{"step_size": ..., "inverse_mass_matrix": ...}``).
        Empty dict means "use default params from BO / recipe".
    compatible_methods
        Tuple of ``BaseMethod.name`` values this warmup supports.
        The special sentinel ``"*"`` means "compatible with all algorithms".
    notes
        Free-form documentation string.
    """

    name: str
    runner: Callable[..., tuple[Any, dict]]
    compatible_methods: tuple[str, ...]
    notes: str = ""

    def is_compatible(self, base_method_name: str) -> bool:
        """Return True if this warmup supports the named algorithm.

        Parameters
        ----------
        base_method_name
            ``BaseMethod.name`` string, e.g. ``"nuts"``, ``"mclmc"``.

        Returns
        -------
        bool
            ``True`` if ``"*"`` is in ``self.compatible_methods`` (meaning
            this warmup works with any algorithm) or if ``base_method_name``
            is listed explicitly in ``self.compatible_methods``.
        """
        return (
            "*" in self.compatible_methods
            or base_method_name in self.compatible_methods
        )
