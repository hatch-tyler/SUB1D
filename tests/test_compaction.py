"""Tests for the compaction module."""
from __future__ import annotations

import numpy as np
import pytest

from sub1d.compaction import (
    scale_by_varying_thickness,
    aggregate_deformation,
)


class TestScaleByVaryingThickness:
    """Test thickness-scaling of deformation."""

    def test_constant_thickness_no_change(self):
        """If thickness ratio is 1.0 everywhere, no scaling."""
        n = 100
        deformation = np.linspace(0, -1.0, n)
        dates = np.arange(n, dtype=float)

        # All one ratio -> same thickness
        layer_thicknesses = {"pre-2000": 100.0, "2000-": 100.0}
        initial_thickness = 100.0

        result = scale_by_varying_thickness(
            deformation, layer_thicknesses, initial_thickness, dates,
        )

        np.testing.assert_allclose(result, deformation, atol=1e-10)


class TestAggregateDeformation:
    """Test aggregation of deformation across layers."""

    def test_sums_correctly(self):
        """Total should be the sum of individual layers."""
        import matplotlib.dates as mdates
        import datetime

        n = 50
        # Create numeric dates (matplotlib format)
        base_date = mdates.date2num(datetime.datetime(2000, 1, 1))
        dates_num = base_date + np.arange(n, dtype=float)

        deformation_by_layer = {
            "Layer1": {
                "total": np.vstack([dates_num, np.ones(n) * -0.5]),
            },
            "Layer2": {
                "total": np.vstack([dates_num, np.ones(n) * -0.3]),
            },
        }

        layer_compaction_switch = {"Layer1": True, "Layer2": True}

        result = aggregate_deformation(deformation_by_layer, layer_compaction_switch)

        assert "Total" in result.columns
        np.testing.assert_allclose(result["Total"].values, -0.8, atol=1e-10)
