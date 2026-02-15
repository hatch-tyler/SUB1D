"""Read, clip, and interpolate head data for the SUB1D subsidence model.

This module extracts and deduplicates the head-reading, clipping,
overburden-stress, interpolation, and resume-mode logic that was previously
copy-pasted in two places inside the monolithic ``model.py``.

Functions
---------
read_head_data
    Read head CSV files for every aquifer and optionally archive them.
clip_head_timeseries
    Trim all aquifer head series to a common date window.
compute_overburden_stress
    Derive overburden stress from water-level changes in the unconfined aquifer.
interpolate_head_series
    Re-sample a head time series onto a uniform timestep.
apply_resume_constant_head
    Lock selected aquifer heads to their first value (resume mode).
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from sub1d.exceptions import InputDataError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_head_data(df: pd.DataFrame, aquifer: str) -> pd.DataFrame:
    """Validate and clean a head-data DataFrame in place.

    Checks performed (all non-fatal unless data is unrecoverable):

    1. NaN / Inf values in the head column — interpolated with warning.
    2. Duplicate dates — dropped with warning.
    3. Non-monotonic dates — sorted with warning.
    4. Large temporal gaps (>30 days) — warning only.
    5. Extreme head values (|head| > 1000 m) — warning only.

    Parameters
    ----------
    df : pd.DataFrame
        Two-column DataFrame (dates, heads).
    aquifer : str
        Aquifer name (for log messages).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    date_col = df.columns[0]
    head_col = df.columns[1]

    # 1. NaN / Inf
    vals = pd.to_numeric(df[head_col], errors="coerce").to_numpy()
    bad_mask = ~np.isfinite(vals)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        logger.warning(
            "%s: %d NaN/Inf values detected in head data; interpolating.",
            aquifer, n_bad,
        )
        # Replace Inf/-Inf with NaN so interpolation works
        df[head_col] = df[head_col].replace([np.inf, -np.inf], np.nan)
        df[head_col] = df[head_col].interpolate(method="linear", limit_direction="both")
        # If still NaN (e.g. all-NaN), forward/back fill
        df[head_col] = df[head_col].ffill().bfill()

    # 2. Duplicate dates
    dup_mask = df[date_col].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        logger.warning(
            "%s: %d duplicate dates dropped.", aquifer, n_dup,
        )
        df = df[~dup_mask].reset_index(drop=True)

    # 3. Non-monotonic dates
    dates = pd.to_datetime(df[date_col])
    if not dates.is_monotonic_increasing:
        logger.warning(
            "%s: dates are not monotonically increasing; sorting.",
            aquifer,
        )
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df[date_col])
        if not dates.is_monotonic_increasing:
            raise InputDataError(
                f"{aquifer}: dates are still non-monotonic after sorting. "
                "Check for duplicate dates with different values."
            )

    # 4. Large gaps (>30 days)
    if len(dates) > 1:
        diffs = dates.diff().dt.days.iloc[1:]
        max_gap = diffs.max()
        if max_gap > 30:
            n_gaps = int((diffs > 30).sum())
            logger.warning(
                "%s: %d gap(s) > 30 days detected (max gap: %.0f days).",
                aquifer, n_gaps, max_gap,
            )

    # 5. Extreme head values
    heads = df[head_col].to_numpy(dtype=float)
    n_extreme = int(np.sum(np.abs(heads) > 1000))
    if n_extreme > 0:
        logger.warning(
            "%s: %d head value(s) with |head| > 1000 m detected.",
            aquifer, n_extreme,
        )

    return df


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_head_data(
    head_data_files: dict[str, str],
    output_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Read head-data CSV files for each aquifer.

    Each CSV is expected to have a date/datetime in column 0 and head values
    in column 1.  The date column is parsed automatically by *pandas*.

    Parameters
    ----------
    head_data_files : dict[str, str]
        Mapping of aquifer name to the file-system path of its head CSV.
    output_dir : str, optional
        When provided the raw CSV files are copied into an ``input_data``
        sub-directory under *output_dir* so that every model run carries a
        snapshot of its inputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of aquifer name to a two-column DataFrame whose first column
        contains parsed datetimes and whose second column contains head values.

    Raises
    ------
    InputDataError
        If any of the specified CSV files does not exist on disk.
    """
    head_data: dict[str, pd.DataFrame] = {}

    if output_dir:
        input_copy = os.path.normpath(os.path.join(output_dir, "input_data"))
        os.makedirs(input_copy, exist_ok=True)

    for aquifer, filepath in head_data_files.items():
        logger.info("Reading head data for %s from %s", aquifer, filepath)

        if not os.path.isfile(filepath):
            raise InputDataError(f"Head data file not found: {filepath}")

        data = pd.read_csv(filepath, parse_dates=[0])
        data = _validate_head_data(data, aquifer)
        head_data[aquifer] = data

        if output_dir:
            shutil.copy2(
                filepath,
                os.path.join(input_copy, os.path.basename(filepath)),
            )

        logger.info(
            "Successfully read head data for %s (%d records)",
            aquifer,
            len(data),
        )

    return head_data


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_head_timeseries(
    head_data: dict[str, pd.DataFrame],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp | None, pd.Timestamp | None]:
    """Clip all aquifer head series to a common start/end date window.

    When multiple aquifers have different observation periods the overlapping
    window is determined automatically: the *latest* first date becomes the
    start and the *earliest* last date becomes the end.  Explicit *start* /
    *end* arguments override the automatic detection.

    Parameters
    ----------
    head_data : dict[str, pd.DataFrame]
        Head data for every aquifer.  Each DataFrame must have datetimes in
        column 0.
    start : pd.Timestamp, optional
        Forced start date.  If ``None`` the latest first date across all
        aquifers is used.
    end : pd.Timestamp, optional
        Forced end date.  If ``None`` the earliest last date across all
        aquifers is used.

    Returns
    -------
    clipped : dict[str, pd.DataFrame]
        New DataFrames (with reset indices) trimmed to the common window.
    start : pd.Timestamp or None
        The resolved start date (``None`` only when *head_data* is empty).
    end : pd.Timestamp or None
        The resolved end date (``None`` only when *head_data* is empty).
    """
    if not head_data:
        return head_data, None, None

    if start is None:
        starttimes = [df.iloc[:, 0].min() for df in head_data.values()]
        start = max(starttimes)

    if end is None:
        endtimes = [df.iloc[:, 0].max() for df in head_data.values()]
        end = min(endtimes)

    logger.info(
        "Clipping head timeseries to %s - %s",
        start.strftime("%d-%b-%Y"),
        end.strftime("%d-%b-%Y"),
    )

    clipped: dict[str, pd.DataFrame] = {}
    for aquifer, df in head_data.items():
        mask = (df.iloc[:, 0] >= start) & (df.iloc[:, 0] <= end)
        clipped[aquifer] = df[mask].reset_index(drop=True)
        logger.info("Clipped %s: %d records", aquifer, len(clipped[aquifer]))

    return clipped, start, end


# ---------------------------------------------------------------------------
# Overburden stress
# ---------------------------------------------------------------------------

def compute_overburden_stress(
    head_data: dict[str, pd.DataFrame],
    uppermost_aquifer: str,
    specific_yield: float,
    rho_w: float,
    g: float,
) -> pd.DataFrame:
    r"""Compute overburden stress from water-level changes in the unconfined aquifer.

    The overburden stress increment is calculated as:

    .. math::

        \Delta\sigma_{v} = S_y \, \rho_w \, g \, (h - h_0)

    where :math:`h_0` is the head at the first observation.

    Parameters
    ----------
    head_data : dict[str, pd.DataFrame]
        Head data per aquifer.
    uppermost_aquifer : str
        Name of the uppermost (unconfined) aquifer whose water-level
        fluctuations drive the overburden change.
    specific_yield : float
        Specific yield of the unconfined aquifer (dimensionless).
    rho_w : float
        Density of water in kg/m\ :sup:`3` (typically 1000).
    g : float
        Gravitational acceleration in m/s\ :sup:`2` (typically 9.81).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame.  Column 0 contains the dates (same name as the
        source date column) and column 1 is ``"Overburden Stress"`` in Pa.
    """
    df = head_data[uppermost_aquifer]
    dates = df.iloc[:, 0]
    heads = df.iloc[:, 1]

    stress = specific_yield * rho_w * g * (heads - heads.iloc[0])

    result = pd.DataFrame(
        {
            dates.name or "date": dates.values,
            "Overburden Stress": stress.values,
        }
    )

    logger.info("Computed overburden stress for %s", uppermost_aquifer)
    return result


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_head_series(
    dates: np.ndarray,
    heads: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Linearly interpolate a head time series onto a uniform timestep.

    The routine uses scaled-integer arithmetic (factor of 10 000) when building
    the new time vector so that the resulting sample points are identical to
    those produced by the legacy code, avoiding any floating-point drift.

    Parameters
    ----------
    dates : np.ndarray
        1-D array of numeric dates (e.g. output of
        ``matplotlib.dates.date2num``).
    heads : np.ndarray
        1-D array of head values, same length as *dates*.
    dt : float
        Desired uniform timestep in the same units as *dates*.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 2)`` where column 0 is the new time vector and
        column 1 is the linearly interpolated head.
    """
    t_new = 0.0001 * np.arange(
        10000 * dates.min(),
        10000 * dates.max() + 1,
        10000 * dt,
    )
    h_new = np.interp(t_new, dates, heads)
    return np.column_stack([t_new, h_new])


# ---------------------------------------------------------------------------
# Resume-mode helper
# ---------------------------------------------------------------------------

def apply_resume_constant_head(
    head_data: dict[str, pd.DataFrame],
    resume_head_value: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Set aquifer heads to a constant for resume-mode runs.

    When a model run is resumed and a particular aquifer is flagged with
    ``"cst"`` in *resume_head_value*, every head observation for that aquifer
    is replaced by the value of the first observation.  This effectively
    freezes the head so that no further stress changes are imposed.

    Parameters
    ----------
    head_data : dict[str, pd.DataFrame]
        Head data per aquifer.  The input is **not** modified; a deep copy is
        returned.
    resume_head_value : dict[str, str]
        Mapping of aquifer name to a mode string.  Only the value ``"cst"`` is
        acted upon; all other values are silently ignored.

    Returns
    -------
    dict[str, pd.DataFrame]
        Deep copy of *head_data* with the requested aquifers set to constant
        head.
    """
    result = copy.deepcopy(head_data)
    for aquifer, mode in resume_head_value.items():
        if mode == "cst" and aquifer in result:
            first_val = result[aquifer].iloc[0, 1]
            result[aquifer].iloc[:, 1] = first_val
            logger.info(
                "Set constant head for %s at %.2f", aquifer, first_val
            )
    return result
