# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tests for stationary initialization data used by generated recipes."""

import json
from pathlib import Path

import numpy as np
import pytest

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tuningfork" / "catalog"


@pytest.mark.fast
def test_stationary_init_shape() -> None:
    """Stationary positions are batched by chain with model-shaped leaves."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=4, catalog_root=_CATALOG_ROOT
    )
    assert positions["mu"].shape == (4,)
    assert positions["tau"].shape == (4,)
    assert positions["theta_raw"].shape == (4, 8)


@pytest.mark.fast
def test_stationary_init_offsets_correct() -> None:
    """Each chain receives a distinct deterministic offset from the reference mean."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=4, catalog_root=_CATALOG_ROOT
    )
    summary = json.loads(
        (_CATALOG_ROOT / "eight_schools_ncp" / "reference" / "summary.json").read_text()
    )
    gt_mean = float(summary["mean"]["mu"])
    gt_std = float(summary["std"]["mu"])
    for value, offset in zip(np.asarray(positions["mu"]), (0.1, -0.1, 0.05, -0.05)):
        assert float(value) == pytest.approx(gt_mean + offset * gt_std, abs=1e-5)


@pytest.mark.fast
def test_stationary_init_num_chains_cycling() -> None:
    """Offset schedule cycles when fewer chains are requested."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    positions = _build_stationary_init_positions(
        "eight_schools_ncp", num_chains=2, catalog_root=_CATALOG_ROOT
    )
    assert positions["mu"].shape == (2,)
    assert float(positions["mu"][0]) != float(positions["mu"][1])


@pytest.mark.fast
def test_stationary_init_missing_summary_raises() -> None:
    """Missing reference summaries are reported clearly."""
    from tuningfork.recipes._recipe_runner import _build_stationary_init_positions

    with pytest.raises(FileNotFoundError, match="Reference summary not found"):
        _build_stationary_init_positions(
            "nonexistent_model", num_chains=4, catalog_root=_CATALOG_ROOT
        )
