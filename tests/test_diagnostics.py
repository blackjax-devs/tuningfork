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
"""Smoke tests for tuningfork.catalog.diagnostics family renderers.

Each family renderer is tested independently to catch axis-shape and
ArviZ-import regressions cheaply.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tuningfork.catalog.diagnostics import (
    render_gradient_mh,
    render_universal_summary,
    samples_to_idata,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def mock_samples_multichain():
    """Mock multi-chain samples: (n_chains=2, n_draws=100, dim=5)."""
    rng = np.random.default_rng(42)
    return {
        "x": rng.standard_normal((2, 100, 5)),
        "y": rng.standard_normal((2, 100, 3)),
    }


@pytest.fixture
def mock_samples_singlechain():
    """Mock single-chain samples: (n_draws=100, dim=5)."""
    rng = np.random.default_rng(42)
    return {
        "x": rng.standard_normal((100, 5)),
        "y": rng.standard_normal((100, 3)),
    }


@pytest.fixture
def mock_info_with_divergence():
    """Mock info struct with divergence array."""

    class MockInfo:
        def __init__(self):
            self.is_divergent = np.array([[False] * 100 for _ in range(2)])

    return MockInfo()


@pytest.fixture
def mock_gate_verdict():
    """Mock gate verdict dict."""
    return {
        "rhat_max": 1.003,
        "min_bulk_ess": 456,
        "n_divergences": 0,
        "max_abs_mean_z": 1.5,
        "verdict": "PASS",
        "margins": {},
    }


def test_samples_to_idata_multichain(mock_samples_multichain):
    """Test ArviZ conversion from multi-chain samples."""
    idata = samples_to_idata(mock_samples_multichain, is_multichain=True)
    assert idata is not None
    assert "x" in idata.posterior
    assert "y" in idata.posterior
    assert idata.posterior["x"].shape == (2, 100, 5)
    assert idata.posterior["y"].shape == (2, 100, 3)


def test_samples_to_idata_singlechain(mock_samples_singlechain):
    """Test ArviZ conversion from single-chain samples."""
    idata = samples_to_idata(mock_samples_singlechain, is_multichain=False)
    assert idata is not None
    assert "x" in idata.posterior
    assert "y" in idata.posterior
    # Should be reshaped to (1, n_draws, *event)
    assert idata.posterior["x"].shape[0] == 1
    assert idata.posterior["x"].shape[1] == 100


def test_render_universal_summary(
    mock_samples_multichain, mock_info_with_divergence, mock_gate_verdict
):
    """Test Section 1 universal summary table rendering."""
    idata = samples_to_idata(mock_samples_multichain, is_multichain=True)
    fig = render_universal_summary(
        idata,
        mock_info_with_divergence,
        mock_gate_verdict,
        wall_time_seconds=1.5,
    )
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_render_gradient_mh(mock_samples_multichain, mock_info_with_divergence):
    """Test gradient MH renderer (semantic name)."""
    idata = samples_to_idata(mock_samples_multichain, is_multichain=True)
    figs = render_gradient_mh(idata, mock_info_with_divergence, sampler_name="nuts")
    assert isinstance(figs, list)
    assert len(figs) > 0
    for fig in figs:
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
