"""Tests for head data I/O functions."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from sub1d.head_io import (
    read_head_data,
    clip_head_timeseries,
    compute_overburden_stress,
    interpolate_head_series,
    _validate_head_data,
)
from sub1d.exceptions import InputDataError


@pytest.fixture
def sample_head_csv(tmp_path):
    """Create sample CSV head data files."""
    dates = pd.date_range("2000-01-01", periods=100, freq="D")
    heads = 10.0 + np.sin(np.arange(100) * 0.1)

    filepath = tmp_path / "test_heads.csv"
    df = pd.DataFrame({"date": dates, "head": heads})
    df.to_csv(filepath, index=False)

    return str(filepath), dates, heads


class TestReadHeadData:
    """Tests for read_head_data."""

    def test_reads_csv_successfully(self, sample_head_csv):
        filepath, dates, heads = sample_head_csv
        result = read_head_data({"TestAquifer": filepath})

        assert "TestAquifer" in result
        assert len(result["TestAquifer"]) == 100

    def test_raises_on_missing_file(self):
        with pytest.raises(InputDataError, match="not found"):
            read_head_data({"TestAquifer": "/nonexistent/path.csv"})

    def test_copies_to_output_dir(self, sample_head_csv, tmp_path):
        filepath, _, _ = sample_head_csv
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)

        read_head_data({"TestAquifer": filepath}, output_dir=output_dir)

        input_copy = os.path.join(output_dir, "input_data")
        assert os.path.isdir(input_copy)


class TestClipHeadTimeseries:
    """Tests for clip_head_timeseries."""

    def test_clips_to_common_dates(self):
        dates1 = pd.date_range("2000-01-01", periods=100, freq="D")
        dates2 = pd.date_range("2000-02-01", periods=100, freq="D")

        head_data = {
            "Aquifer1": pd.DataFrame({"date": dates1, "head": np.ones(100)}),
            "Aquifer2": pd.DataFrame({"date": dates2, "head": np.ones(100)}),
        }

        clipped, start, end = clip_head_timeseries(head_data)

        # Start should be the later of the two starts
        assert start == pd.Timestamp("2000-02-01")
        # End should be the earlier of the two ends
        assert end == pd.Timestamp("2000-04-09")

    def test_respects_explicit_bounds(self):
        dates = pd.date_range("2000-01-01", periods=100, freq="D")
        head_data = {
            "Aquifer1": pd.DataFrame({"date": dates, "head": np.ones(100)}),
        }

        start = pd.Timestamp("2000-02-01")
        end = pd.Timestamp("2000-03-01")
        clipped, s, e = clip_head_timeseries(head_data, start=start, end=end)

        assert s == start
        assert e == end
        # All dates should be within bounds
        df = clipped["Aquifer1"]
        assert df.iloc[:, 0].min() >= start
        assert df.iloc[:, 0].max() <= end


class TestComputeOverburdenStress:
    """Tests for compute_overburden_stress."""

    def test_zero_at_initial_head(self):
        dates = pd.date_range("2000-01-01", periods=10, freq="D")
        heads = np.full(10, 100.0)  # Constant head

        head_data = {
            "TopAquifer": pd.DataFrame({"date": dates, "head": heads}),
        }

        ob = compute_overburden_stress(
            head_data, "TopAquifer",
            specific_yield=0.2, rho_w=1000.0, g=9.81,
        )

        # With constant head, overburden change should be zero
        np.testing.assert_allclose(ob.iloc[:, 1].values, 0.0, atol=1e-10)

    def test_positive_for_declining_head(self):
        dates = pd.date_range("2000-01-01", periods=10, freq="D")
        heads = np.linspace(100.0, 90.0, 10)  # Declining head

        head_data = {
            "TopAquifer": pd.DataFrame({"date": dates, "head": heads}),
        }

        ob = compute_overburden_stress(
            head_data, "TopAquifer",
            specific_yield=0.2, rho_w=1000.0, g=9.81,
        )

        # Declining head -> negative overburden change (Sy * rho * g * delta_h < 0)
        assert ob.iloc[-1, 1] < 0


class TestInterpolateHeadSeries:
    """Tests for interpolate_head_series."""

    def test_interpolates_to_finer_timestep(self):
        dates = np.array([0.0, 1.0, 2.0, 3.0])
        heads = np.array([10.0, 8.0, 6.0, 4.0])

        result = interpolate_head_series(dates, heads, dt=0.5)

        assert result.shape[1] == 2
        assert len(result) == 7  # 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        # Check interpolated values
        np.testing.assert_allclose(result[1, 1], 9.0, atol=0.01)  # at t=0.5


class TestValidateHeadData:
    """Tests for _validate_head_data."""

    def test_clean_data_passes_through(self):
        """Clean data should be returned unchanged."""
        dates = pd.date_range("2000-01-01", periods=10, freq="D")
        df = pd.DataFrame({"date": dates, "head": np.arange(10, dtype=float)})

        result = _validate_head_data(df, "TestAquifer")
        assert len(result) == 10

    def test_nan_interpolated(self):
        """NaN values should be interpolated."""
        dates = pd.date_range("2000-01-01", periods=5, freq="D")
        heads = [1.0, np.nan, 3.0, np.nan, 5.0]
        df = pd.DataFrame({"date": dates, "head": heads})

        result = _validate_head_data(df, "TestAquifer")
        assert not result["head"].isna().any()
        assert result["head"].iloc[1] == pytest.approx(2.0, abs=0.01)

    def test_duplicates_removed(self):
        """Duplicate dates should be dropped."""
        dates = pd.to_datetime(["2000-01-01", "2000-01-01", "2000-01-02"])
        df = pd.DataFrame({"date": dates, "head": [1.0, 2.0, 3.0]})

        result = _validate_head_data(df, "TestAquifer")
        assert len(result) == 2

    def test_non_monotonic_sorted(self):
        """Non-monotonic dates should be sorted."""
        dates = pd.to_datetime(["2000-01-03", "2000-01-01", "2000-01-02"])
        df = pd.DataFrame({"date": dates, "head": [3.0, 1.0, 2.0]})

        result = _validate_head_data(df, "TestAquifer")
        assert result.iloc[0, 0] == pd.Timestamp("2000-01-01")
        assert result.iloc[-1, 0] == pd.Timestamp("2000-01-03")

    def test_inf_handled(self):
        """Inf values should be interpolated like NaN."""
        dates = pd.date_range("2000-01-01", periods=5, freq="D")
        heads = [1.0, np.inf, 3.0, -np.inf, 5.0]
        df = pd.DataFrame({"date": dates, "head": heads})

        result = _validate_head_data(df, "TestAquifer")
        assert np.all(np.isfinite(result["head"].values))
