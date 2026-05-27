# flake8: noqa: F401
"""Timing context helpers for recipe timing table display.

Functions to compute and format timing metadata from recipe calibration_budget.
Used by the catalog_explorer notebook to contextualize measured timings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tuningfork.recipes._base import Recipe


def compute_total_warmup_steps(warmups: list[dict] | None) -> int | None:
    """Sum n_warmup across all warmup phases.

    When a recipe uses multi-phase warmup (e.g., pathfinder + window_adaptation),
    this returns the total number of warmup steps across all phases.

    Parameters
    ----------
    warmups
        List of warmup stage dicts, each with "params" containing "n_warmup".
        If None or empty, returns None.

    Returns
    -------
    int | None
        Sum of all n_warmup values across phases, or None if warmups is
        None/empty or any phase lacks n_warmup.

    Examples
    --------
    Single-phase warmup:

    >>> warmups_single = [
    ...     {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 500}}
    ... ]
    >>> compute_total_warmup_steps(warmups_single)
    500

    Multi-phase warmup:

    >>> warmups_dual = [
    ...     {"name": "pathfinder", "params": {"n_warmup": 100}},
    ...     {"name": "window_adaptation_diag_imm", "params": {"n_warmup": 400}},
    ... ]
    >>> compute_total_warmup_steps(warmups_dual)
    500

    Legacy recipe with no warmups list:

    >>> compute_total_warmup_steps(None)
    """
    if not warmups:
        return None

    total = 0
    for phase in warmups:
        params = phase.get("params", {})
        n_warmup = params.get("n_warmup")
        if n_warmup is None:
            return None
        total += n_warmup

    return total if total > 0 else None


def format_timing_context(recipe: Recipe) -> dict[str, str]:
    """Format context strings for the timing metadata table.

    Extracts calibration_budget and warmups fields from a recipe and
    produces human-readable context for each row of the measured-timings table
    (e.g., "1000 steps × 4 chains" for warmup wall, etc.).

    Parameters
    ----------
    recipe
        A Recipe object with calibration_budget and warmups fields.

    Returns
    -------
    dict[str, str]
        Keys: "warmup_wall", "sampling_wall", "per_draw", "total_wall", "machine".
        Values: context strings (e.g., "1000 steps × 4 chains"), or "—" if
        the required data is missing from calibration_budget.

    Examples
    --------
    >>> # Recipe with complete timing data
    >>> ctx = format_timing_context(recipe)
    >>> ctx["warmup_wall"]
    '1000 steps × 4 chains'
    >>> ctx["sampling_wall"]
    '1000/chain × 4 chains = 4000 draws'
    >>> ctx["per_draw"]
    'per chain·draw (4000 draws)'
    >>> ctx["total_wall"]
    'warmup + sampling'
    >>> ctx["machine"]
    ''
    """
    budget = recipe.calibration_budget or {}

    n_warmup = budget.get("n_warmup")
    n_samples = budget.get("n_samples")
    num_chains = budget.get("num_chains")
    warmup_wall = budget.get("warmup_wall_seconds")
    sampling_wall = budget.get("sampling_wall_seconds")

    total_warmup_steps = compute_total_warmup_steps(recipe.warmups)

    context: dict[str, str] = {}

    # Warmup wall: "{total_warmup_steps} steps × {num_chains} chains"
    if total_warmup_steps is not None and num_chains is not None:
        context["warmup_wall"] = f"{total_warmup_steps} steps × {num_chains} chains"
    else:
        context["warmup_wall"] = "—"

    # Sampling wall: "{n_samples}/chain × {num_chains} chains = {n_samples*num_chains} draws"
    if n_samples is not None and num_chains is not None:
        total_draws = n_samples * num_chains
        context["sampling_wall"] = (
            f"{n_samples}/chain × {num_chains} chains = {total_draws} draws"
        )
    else:
        context["sampling_wall"] = "—"

    # Per-draw: "per chain·draw ({n_samples*num_chains} draws)"
    if n_samples is not None and num_chains is not None:
        total_draws = n_samples * num_chains
        context["per_draw"] = f"per chain·draw ({total_draws} draws)"
    else:
        context["per_draw"] = "—"

    # Total wall est.: "warmup + sampling" or blank
    if warmup_wall is not None and sampling_wall is not None:
        context["total_wall"] = "warmup + sampling"
    else:
        context["total_wall"] = ""

    # Machine: blank (no scale context)
    context["machine"] = ""

    return context
