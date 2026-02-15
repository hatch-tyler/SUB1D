"""Unified file output for the SUB1D model.

This module replaces the 6+ repeated save patterns previously scattered
throughout ``model.py``.  Every pattern followed the same logic:

1. If the array size >= a threshold, save as raw binary (and optionally
   convert to netCDF via GMT's ``xyz2grd``).
2. Otherwise, save as CSV.

By centralising that logic here we eliminate duplication and make the
output behaviour easier to test and extend.

Functions
---------
save_array
    Save a NumPy array to disk (CSV, binary, or auto-select).
save_head_outputs
    Persist all head-related arrays for every layer.
save_compaction_outputs
    Persist all compaction-related arrays for every layer.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size thresholds matching legacy behaviour
# ---------------------------------------------------------------------------
LARGE_ARRAY_THRESHOLD: int = 1_000_000
"""Arrays with at least this many elements are saved as binary by default."""

VERY_LARGE_ARRAY_THRESHOLD: int = 3_000_000
"""Threshold used for very large diagnostic arrays (e.g. inelastic flags)."""


# ===================================================================== #
#                          Core save routine                             #
# ===================================================================== #

def save_array(
    data: np.ndarray,
    path: str | Path,
    fmt: str = "auto",
    size_threshold: int = LARGE_ARRAY_THRESHOLD,
    dtype_binary: type = np.float32,
    gmt_convert: bool = False,
    gmt_params: dict | None = None,
) -> None:
    """Save a NumPy array to disk.

    The function supports three concrete formats and an *auto* mode that
    picks the format based on array size:

    * **csv** -- human-readable comma-separated values.
    * **binary** -- raw binary dump (fast, compact).
    * **auto** -- binary if ``data.size >= size_threshold``, else CSV.

    When *binary* is selected and ``gmt_convert`` is *True*, the file is
    additionally converted to netCDF using GMT's ``xyz2grd`` utility and
    the intermediate binary is removed on success.

    Parameters
    ----------
    data : np.ndarray
        Array to save.  May be 1-D or 2-D; CSV output writes one row per
        array row.
    path : str or Path
        Destination file path.  When *fmt* is ``"auto"`` the appropriate
        extension (``.csv`` or none) is appended automatically.
    fmt : str, optional
        Output format.  One of ``"auto"`` (default), ``"csv"``, or
        ``"binary"``.
    size_threshold : int, optional
        Element count above which *auto* selects binary.  Defaults to
        :data:`LARGE_ARRAY_THRESHOLD`.
    dtype_binary : type, optional
        NumPy dtype used when writing binary files.  Defaults to
        ``np.float32``.
    gmt_convert : bool, optional
        If *True* and the chosen format is binary, attempt to convert the
        output to netCDF via ``gmt xyz2grd``.  Defaults to *False*.
    gmt_params : dict or None, optional
        Parameters forwarded to :func:`_convert_gmt`.  Required keys are
        ``'dt'``, ``'dz'``, ``'tmin'``, ``'tmax'``, ``'zmin'``, and
        ``'zmax'``.  The optional key ``'type_flag'`` (default ``"f"``)
        controls the ``-ZTL`` flag sent to GMT.

    Raises
    ------
    ValueError
        If *fmt* is not one of the recognised format strings.
    """
    path = Path(path)

    if fmt == "auto":
        fmt = "binary" if data.size >= size_threshold else "csv"

    if fmt == "csv":
        _save_csv(data, path.with_suffix(".csv"))
    elif fmt == "binary":
        _save_binary(data, path, dtype_binary, gmt_convert, gmt_params)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ===================================================================== #
#                         Private helpers                                #
# ===================================================================== #

def _save_csv(data: np.ndarray, path: Path) -> None:
    """Save an array as a CSV file.

    Parameters
    ----------
    data : np.ndarray
        Array to write.  A 1-D array is treated as a single-column table.
    path : Path
        Destination path (should already carry the ``.csv`` suffix).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)
    logger.info("Saved CSV: %s (%d elements)", path, data.size)


def _save_binary(
    data: np.ndarray,
    path: Path,
    dtype: type,
    gmt_convert: bool,
    gmt_params: dict | None,
) -> None:
    """Save an array as a raw binary file.

    Parameters
    ----------
    data : np.ndarray
        Array to write.
    path : Path
        Destination path for the binary file.
    dtype : type
        NumPy dtype to cast to before writing.
    gmt_convert : bool
        Whether to invoke GMT conversion after writing.
    gmt_params : dict or None
        Parameters for :func:`_convert_gmt`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(dtype).tofile(path)
    logger.info(
        "Saved binary: %s (%d elements, dtype=%s)",
        path,
        data.size,
        dtype.__name__,
    )

    if gmt_convert and gmt_params:
        _convert_gmt(path, gmt_params)


def _convert_gmt(binary_path: Path, params: dict) -> None:
    """Convert a raw binary file to netCDF using ``gmt xyz2grd``.

    On success the intermediate binary file is deleted.  On failure
    (missing GMT installation, bad parameters, etc.) a warning is logged
    and the binary file is kept.

    Parameters
    ----------
    binary_path : Path
        Path to the binary file produced by :func:`_save_binary`.
    params : dict
        GMT grid parameters.  Expected keys:

        * ``dt`` (*float*) -- time increment.
        * ``dz`` (*float*) -- depth increment.
        * ``tmin``, ``tmax`` (*float*) -- time range.
        * ``zmin``, ``zmax`` (*float*) -- depth range.
        * ``type_flag`` (*str*, optional) -- binary type flag for the
          ``-ZTL`` option (default ``"f"`` for 4-byte float).
    """
    nc_path = binary_path.with_suffix(".nc")
    type_flag = params.get("type_flag", "f")
    cmd = [
        "gmt",
        "xyz2grd",
        str(binary_path),
        f"-G{nc_path}",
        f"-I{params['dt']:.3f}/{params['dz']:.5f}",
        (
            f"-R{params['tmin']:.3f}/{params['tmax']:.3f}"
            f"/{params['zmin']:.3f}/{params['zmax']:.3f}"
        ),
        f"-ZTL{type_flag}",
    ]
    logger.info("Converting to netCDF: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        binary_path.unlink()
        logger.info("Converted to netCDF: %s", nc_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("GMT conversion failed (keeping binary): %s", exc)


# ===================================================================== #
#              High-level output routines (head data)                    #
# ===================================================================== #

def save_head_outputs(
    head_series: dict,
    groundwater_solution_dates: dict,
    inelastic_flag: dict,
    effective_stress: dict,
    layer_types: dict,
    Z: dict,
    t_gwflow: dict,
    dt_master: dict,
    interbeds_distributions: dict,
    outdestination: str,
    config: Any,
) -> None:
    """Save all head-related outputs for every layer.

    This function replaces the repeated per-layer save patterns that were
    formerly embedded in ``model.py``.  It delegates to either
    :func:`_save_aquitard_head_outputs` or
    :func:`_save_aquifer_head_outputs` depending on the layer type.

    Parameters
    ----------
    head_series : dict
        Mapping of layer name to head solution matrix.
    groundwater_solution_dates : dict
        Mapping of layer name to solution date arrays.
    inelastic_flag : dict
        Mapping of layer name to inelastic-flag arrays.
    effective_stress : dict
        Mapping of layer name to effective-stress arrays.
    layer_types : dict
        Mapping of layer name to ``"Aquitard"`` or ``"Aquifer"``.
    Z : dict
        Mapping of layer name to depth-node arrays.
    t_gwflow : dict
        Mapping of layer name to groundwater-flow time arrays.
    dt_master : dict
        Mapping of layer name to master time-step sizes.
    interbeds_distributions : dict
        Mapping of layer name to interbed thickness distributions
        (relevant for aquifer layers only).
    outdestination : str
        Root output directory.
    config : object
        Run configuration object.  Must expose at least
        ``config.output.save_output_head_timeseries`` and
        ``config.output.save_effective_stress``.
    """
    out_head = os.path.join(outdestination, "head_outputs")
    os.makedirs(out_head, exist_ok=True)

    for layer in head_series:
        if layer_types.get(layer) == "Aquitard":
            _save_aquitard_head_outputs(
                layer,
                head_series[layer],
                inelastic_flag.get(layer),
                effective_stress.get(layer),
                Z.get(layer),
                t_gwflow.get(layer),
                dt_master.get(layer),
                outdestination,
                config,
            )
        elif layer_types.get(layer) == "Aquifer":
            _save_aquifer_head_outputs(
                layer,
                head_series[layer],
                inelastic_flag.get(layer, {}),
                effective_stress.get(layer, {}),
                Z.get(layer, {}),
                t_gwflow.get(layer, {}),
                dt_master.get(layer),
                interbeds_distributions.get(layer, {}),
                outdestination,
                config,
            )


def _save_aquitard_head_outputs(
    layer: str,
    hmat: np.ndarray,
    inelastic_flag: np.ndarray | None,
    effective_stress: np.ndarray | None,
    z: np.ndarray | None,
    t: np.ndarray | None,
    dt: float | None,
    outdestination: str,
    config: Any,
) -> None:
    """Save head outputs for a single aquitard layer.

    Parameters
    ----------
    layer : str
        Human-readable layer name (spaces will be replaced by
        underscores in file names).
    hmat : np.ndarray
        Head solution matrix for this layer.
    inelastic_flag : np.ndarray or None
        Inelastic-flag array; saved only when present and the
        configuration requests head time-series output.
    effective_stress : np.ndarray or None
        Effective-stress array; saved only when present and the
        configuration requests it.
    z : np.ndarray or None
        Depth-node array.
    t : np.ndarray or None
        Time array for the groundwater-flow solution.
    dt : float or None
        Master time-step size.
    outdestination : str
        Root output directory.
    config : object
        Run configuration object.
    """
    layer_safe = layer.replace(" ", "_")
    head_dir = os.path.join(outdestination, "head_outputs")

    if inelastic_flag is not None and config.output.save_output_head_timeseries:
        save_array(
            inelastic_flag,
            os.path.join(head_dir, f"{layer_safe}inelastic_flag_GWFLOW"),
            size_threshold=VERY_LARGE_ARRAY_THRESHOLD,
            dtype_binary=np.int8,
        )

    if config.output.save_effective_stress and effective_stress is not None:
        save_array(
            effective_stress,
            os.path.join(outdestination, f"{layer_safe}effective_stress"),
        )


def _save_aquifer_head_outputs(
    layer: str,
    head_series_layer: np.ndarray,
    inelastic_flag_layer: dict,
    effective_stress_layer: dict,
    Z_layer: dict,
    t_layer: dict,
    dt: float | None,
    interbeds_dist: dict,
    outdestination: str,
    config: Any,
) -> None:
    """Save head outputs for a single aquifer layer with interbeds.

    Aquifer layers may contain multiple interbed thickness classes.  This
    function iterates over each class and saves the corresponding
    inelastic-flag and effective-stress arrays.

    Parameters
    ----------
    layer : str
        Human-readable layer name.
    head_series_layer : np.ndarray
        Head solution matrix for this layer.
    inelastic_flag_layer : dict
        Mapping of thickness key to inelastic-flag arrays.
    effective_stress_layer : dict
        Mapping of thickness key to effective-stress arrays.
    Z_layer : dict
        Mapping of thickness key to depth-node arrays.
    t_layer : dict
        Mapping of thickness key to time arrays.
    dt : float or None
        Master time-step size.
    interbeds_dist : dict
        Interbed thickness distribution for this layer.
    outdestination : str
        Root output directory.
    config : object
        Run configuration object.
    """
    layer_safe = layer.replace(" ", "_")
    head_dir = os.path.join(outdestination, "head_outputs")

    for thickness_key, ifl in inelastic_flag_layer.items():
        if config.output.save_output_head_timeseries and ifl is not None:
            save_array(
                ifl,
                os.path.join(
                    head_dir,
                    f"{layer_safe}_{thickness_key}inelastic_flag_GWFLOW",
                ),
                size_threshold=VERY_LARGE_ARRAY_THRESHOLD,
                dtype_binary=np.int8,
            )

        if config.output.save_effective_stress:
            eff = effective_stress_layer.get(thickness_key)
            if eff is not None:
                save_array(
                    eff,
                    os.path.join(
                        outdestination,
                        f"{layer_safe}_{thickness_key}effective_stress",
                    ),
                )


# ===================================================================== #
#           High-level output routines (compaction data)                 #
# ===================================================================== #

def save_compaction_outputs(
    deformation_output: dict,
    deformation: dict,
    layer_names: list,
    layer_types: dict,
    layer_compaction_switch: dict,
    outdestination: str,
    save_s: bool,
) -> None:
    """Save all compaction-related outputs.

    For each layer whose compaction switch is enabled:

    * A total-deformation CSV summary (from a :class:`pandas.DataFrame`)
      is always written.
    * If ``save_s`` is *True*, the raw deformation array for aquitard
      layers is saved via :func:`save_array`.

    Parameters
    ----------
    deformation_output : dict
        Mapping of layer name to a :class:`pandas.DataFrame` containing
        total deformation results.
    deformation : dict
        Mapping of layer name to a dict whose ``"total"`` key holds the
        raw deformation :class:`numpy.ndarray`.
    layer_names : list
        Ordered list of layer names.
    layer_types : dict
        Mapping of layer name to ``"Aquitard"`` or ``"Aquifer"``.
    layer_compaction_switch : dict
        Mapping of layer name to *bool* indicating whether compaction
        output is requested.
    outdestination : str
        Root output directory.
    save_s : bool
        If *True*, save the raw deformation arrays in addition to the
        summary CSV.
    """
    os.makedirs(os.path.join(outdestination, "s_outputs"), exist_ok=True)

    for layer in layer_names:
        if not layer_compaction_switch.get(layer, False):
            continue

        layer_safe = layer.replace(" ", "_")

        if layer in deformation_output:
            deformation_output[layer].to_csv(
                os.path.join(
                    outdestination,
                    f"{layer}_Total_Deformation_Out.csv",
                ),
                index=False,
            )

        if save_s and layer in deformation:
            if layer_types[layer] == "Aquitard":
                save_array(
                    deformation[layer]["total"],
                    os.path.join(outdestination, f"{layer_safe}_s"),
                )
