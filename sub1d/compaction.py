"""Compaction solver and aggregation routines for the SUB1D model.

This module orchestrates the compaction workflow: iterating over layers,
calling the finite-difference compaction solver for each clay interbed or
aquitard, computing interconnected-matrix (sand) deformation, scaling by
time-varying thicknesses, and aggregating results across all layers.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.dates as mdates

from sub1d.solver import solve_compaction_elastic_inelastic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Aquifer-layer compaction
# ---------------------------------------------------------------------------

def solve_layer_compaction_aquifer(
    layer_name: str,
    head_series: Dict[str, np.ndarray],
    interbeds_distributions: Dict[float, float],
    sand_Sse: float,
    compressibility_of_water: float,
    layer_thickness: Union[float, Dict[str, float]],
    layer_thickness_type: str,
    initial_thickness: float,
    clay_Sse: float,
    clay_Ssv: float,
    dz_clays: float,
    dt_master: float,
    groundwater_solution_dates: Dict[str, np.ndarray],
    overburden_stress_compaction: bool,
    overburden_dates: Optional[np.ndarray],
    overburden_data: Optional[np.ndarray],
    rho_w: float,
    g: float,
    compaction_solver_debug_include_endnodes: bool,
    preset_precons: bool,
    initial_condition_precons: Dict[str, np.ndarray],
    unconfined_aquifer_name: Optional[str],
) -> Dict[str, Any]:
    """Compute compaction for an aquifer layer with interbedded clays.

    The total deformation has two components:

    1. **Interconnected matrix** (sand skeleton) -- instantaneous elastic
       response proportional to the change in head from the initial value.
    2. **Clay interbeds** -- delayed, potentially inelastic response solved
       by the 1-D compaction solver for each distinct interbed thickness.

    Parameters
    ----------
    layer_name : str
        Name of the aquifer layer (used only for logging).
    head_series : dict
        ``head_series[layer]`` for this aquifer.  Must contain the key
        ``"Interconnected matrix"`` (a 2-D array with columns
        ``[date_num, head]``) and ``"{thickness:.2f} clays"`` keys holding
        head matrices (nodes x time-steps) for each interbed thickness.
    interbeds_distributions : dict
        Mapping ``{half_thickness_m: n_interbeds}`` for this layer.
    sand_Sse : float
        Elastic skeletal specific storage of the sand (1/m).
    compressibility_of_water : float
        Compressibility of water (1/m), subtracted from ``sand_Sse`` and
        ``clay_Sse``/``clay_Ssv`` before use.
    layer_thickness : float or dict
        Layer thickness.  A single float for ``"constant"`` thickness type,
        or a dict of ``{period_key: thickness}`` for ``"step_changes"``.
    layer_thickness_type : str
        Either ``"constant"`` or ``"step_changes"``.
    initial_thickness : float
        The initial (reference) thickness of the layer, used to compute
        the sand thickness and for thickness-scaling ratios.
    clay_Sse : float
        Elastic skeletal specific storage of the clay interbeds (1/m).
    clay_Ssv : float
        Inelastic (virgin) skeletal specific storage of the clay (1/m).
    dz_clays : float
        Vertical discretisation (m) inside the clay interbeds.
    dt_master : float
        Master time-step (days, as matplotlib date numbers) for the clay
        interbed head solution.
    groundwater_solution_dates : dict
        ``{"{thickness:.2f} clays": np.ndarray}`` of date-number arrays
        corresponding to each interbed head solution.
    overburden_stress_compaction : bool
        Whether to include overburden stress changes in the compaction
        calculation.
    overburden_dates : np.ndarray or None
        Date-number array for the overburden time series.
    overburden_data : np.ndarray or None
        Overburden stress values (Pa) aligned with *overburden_dates*.
    rho_w : float
        Density of water (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).
    compaction_solver_debug_include_endnodes : bool
        If ``True``, the compaction solver uses the boundary nodes directly
        instead of midpoint integration.
    preset_precons : bool
        If ``True``, pre-set preconsolidation stresses are supplied.
    initial_condition_precons : dict
        ``{"{thickness:.2f} clays": np.ndarray}`` of initial
        preconsolidation head arrays for each interbed thickness.
    unconfined_aquifer_name : str or None
        Name of the unconfined aquifer (if any), used to set the
        ``unconfined`` flag in the solver.

    Returns
    -------
    dict
        Keys include:

        * ``"Interconnected matrix"`` -- 1-D array of sand deformation at
          each head-data time step.
        * ``"total_{thickness:.2f} clays"`` -- 1-D array of total clay
          deformation (already multiplied by the number of interbeds) for
          each interbed thickness.
        * ``"total"`` -- 2-D array ``[dates, total_deformation]``.
        * ``"inelastic_flags"`` -- dict of inelastic flag arrays keyed by
          ``"elastic_{thickness:.2f} clays"``.
    """
    logger.info("%s is an Aquifer. Solving for layer compaction.", layer_name)

    deformation: Dict[str, Any] = {}
    inelastic_flags: Dict[str, np.ndarray] = {}

    # ----- Compute sand thickness -----
    total_clay_in_layer = np.sum([
        thickness * count
        for thickness, count in interbeds_distributions.items()
    ])

    if layer_thickness_type == "constant":
        layer_sand_thickness = layer_thickness - total_clay_in_layer
        logger.info(
            "\tTotal sand thickness in aquifer is %.2f m.",
            layer_sand_thickness,
        )
    elif layer_thickness_type == "step_changes":
        layer_sand_thickness = initial_thickness - total_clay_in_layer
        logger.info(
            "\tInitial total sand thickness in aquifer is %.2f m.",
            layer_sand_thickness,
        )
    else:
        layer_sand_thickness = initial_thickness - total_clay_in_layer

    # ----- Interconnected matrix deformation -----
    # head_series["Interconnected matrix"] is an Nx2 array [date, head]
    head_ic = head_series["Interconnected matrix"]
    deformation["Interconnected matrix"] = np.array([
        layer_sand_thickness
        * (sand_Sse - compressibility_of_water)
        * (head_ic[i, 1] - head_ic[0, 1])
        for i in range(len(head_ic[:, 1]))
    ])

    # ----- Clay interbed deformation -----
    bed_thicknesses = list(interbeds_distributions.keys())
    logger.info(
        "\t\t%s is an aquifer with interbedded clays. "
        "Thicknesses of clays to solve compaction are %s",
        layer_name,
        bed_thicknesses,
    )

    for thickness in bed_thicknesses:
        logger.info("\t\t\tSolving for thickness %.2f.", thickness)
        clay_key = f"{thickness:.2f} clays"
        hmat_clay = head_series[clay_key]

        # ---- Prepare overburden stress (if needed) ----
        if overburden_stress_compaction:
            unconfined_tmp = (unconfined_aquifer_name == layer_name)
            gw_dates = groundwater_solution_dates[clay_key]

            if len(overburden_dates) != len(gw_dates):
                logger.info(
                    "\t\t\tOverburden series is %i long whereas head series "
                    "is %i long. Interpolating overburden stress.",
                    len(overburden_dates),
                    len(gw_dates),
                )
                f_interp = scipy.interpolate.interp1d(
                    overburden_dates, overburden_data,
                )
                overburden_data_tmp = f_interp(gw_dates)
            else:
                overburden_data_tmp = overburden_data

            b_tmp, inelastic_flag_tmp = solve_compaction_elastic_inelastic(
                hmat_clay,
                (clay_Sse - compressibility_of_water),
                (clay_Ssv - compressibility_of_water),
                dz_clays,
                unconfined=unconfined_tmp,
                overburden=overburden_stress_compaction,
                overburden_data=(1.0 / (rho_w * g)) * np.array(overburden_data_tmp),
                endnodes=compaction_solver_debug_include_endnodes,
                preset_precons=preset_precons,
                ic_precons=initial_condition_precons[clay_key],
            )
        else:
            b_tmp, inelastic_flag_tmp = solve_compaction_elastic_inelastic(
                hmat_clay,
                (clay_Sse - compressibility_of_water),
                (clay_Ssv - compressibility_of_water),
                dz_clays,
                endnodes=compaction_solver_debug_include_endnodes,
                preset_precons=preset_precons,
                ic_precons=initial_condition_precons[clay_key],
            )

        # Multiply single-interbed deformation by the number of interbeds
        deformation[f"total_{thickness:.2f} clays"] = (
            interbeds_distributions[thickness] * b_tmp
        )
        inelastic_flags[f"elastic_{thickness:.2f} clays"] = inelastic_flag_tmp

    # ----- Collect results at the output time-step -----
    # head_ic is the "Interconnected matrix" array [date, head]; column 0
    # holds the date numbers.
    head_dates = head_ic[:, 0]

    # Build t_total_tmp from head_dates directly (accounts for leap years)
    t_total_tmp_list = [head_dates[0]]
    for i in range(1, len(head_dates)):
        increment = head_dates[i] - head_dates[i - 1]
        t_total_tmp_list.append(t_total_tmp_list[-1] + increment)
    t_total_tmp = np.array(t_total_tmp_list)

    # Interconnected matrix at the output times
    row_indices = np.where(np.isin(head_ic[:, 0], t_total_tmp))[0]
    def_tot_tmp = np.zeros_like(t_total_tmp, dtype=float)
    def_tot_tmp += deformation["Interconnected matrix"][row_indices]

    # Add clay interbed deformation (resampled to the output grid)
    for thickness in bed_thicknesses:
        clay_times = 0.0001 * np.arange(
            10000 * np.min(head_ic[:, 0]),
            10000 * (np.max(head_ic[:, 0]) + 0.0001),
            10000 * dt_master,
        )
        in_output = np.isin(clay_times, t_total_tmp)
        clay_def = np.array(deformation[f"total_{thickness:.2f} clays"])[in_output]
        def_tot_tmp += clay_def

    deformation["total"] = np.array([t_total_tmp, def_tot_tmp])
    deformation["inelastic_flags"] = inelastic_flags

    return deformation


# ---------------------------------------------------------------------------
# 2. Aquitard-layer compaction
# ---------------------------------------------------------------------------

def solve_layer_compaction_aquitard(
    layer_name: str,
    head_series: np.ndarray,
    clay_Sse: float,
    clay_Ssv: float,
    compressibility_of_water: float,
    dz_clays: float,
    groundwater_solution_dates: np.ndarray,
    overburden_stress_compaction: bool,
    overburden_dates: Optional[np.ndarray],
    overburden_data: Optional[np.ndarray],
    rho_w: float,
    g: float,
    compaction_solver_debug_include_endnodes: bool,
    preset_precons: bool,
    initial_condition_precons: np.ndarray,
    unconfined_aquifer_name: Optional[str],
) -> Dict[str, Any]:
    """Compute compaction for a single aquitard layer.

    Parameters
    ----------
    layer_name : str
        Name of the aquitard layer (used for logging).
    head_series : np.ndarray
        Head matrix (nodes x time-steps) produced by the groundwater flow
        solver for this aquitard.
    clay_Sse : float
        Elastic skeletal specific storage (1/m).
    clay_Ssv : float
        Inelastic skeletal specific storage (1/m).
    compressibility_of_water : float
        Compressibility of water (1/m).
    dz_clays : float
        Vertical discretisation (m).
    groundwater_solution_dates : np.ndarray
        Date-number array for the head time-steps.
    overburden_stress_compaction : bool
        Whether to account for overburden stress changes.
    overburden_dates : np.ndarray or None
        Date-number array for the overburden time series.
    overburden_data : np.ndarray or None
        Overburden stress values (Pa).
    rho_w : float
        Density of water (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).
    compaction_solver_debug_include_endnodes : bool
        Pass through to the compaction solver.
    preset_precons : bool
        Whether pre-set preconsolidation stresses are provided.
    initial_condition_precons : np.ndarray
        Initial preconsolidation head array (may be empty).
    unconfined_aquifer_name : str or None
        Name of the unconfined aquifer (if any).

    Returns
    -------
    dict
        Contains ``"total"`` -- a 2-D array ``[dates, deformation]`` and
        ``"inelastic_flag"`` -- the inelastic flag matrix from the solver.
    """
    logger.info(
        "%s is an Aquitard. Solving for layer compaction.", layer_name,
    )

    deformation: Dict[str, Any] = {}

    if overburden_stress_compaction:
        unconfined_tmp = (unconfined_aquifer_name == layer_name)
        logger.info("UNCONFINED STATUS = %s", unconfined_tmp)

        if len(overburden_dates) != len(groundwater_solution_dates):
            logger.info(
                "\t\t\tOverburden series is %i long whereas head series is "
                "%i long. Interpolating overburden stress.",
                len(overburden_dates),
                len(groundwater_solution_dates),
            )
            f_interp = scipy.interpolate.interp1d(
                overburden_dates, overburden_data,
            )
            overburden_data_tmp = f_interp(groundwater_solution_dates)
        else:
            overburden_data_tmp = overburden_data

        totdeftmp, inelastic_flag_tmp = solve_compaction_elastic_inelastic(
            head_series,
            (clay_Sse - compressibility_of_water),
            (clay_Ssv - compressibility_of_water),
            dz_clays,
            unconfined=unconfined_tmp,
            overburden=overburden_stress_compaction,
            overburden_data=(1.0 / (rho_w * g)) * np.array(overburden_data_tmp),
            preset_precons=preset_precons,
            ic_precons=initial_condition_precons,
        )
    else:
        totdeftmp, inelastic_flag_tmp = solve_compaction_elastic_inelastic(
            head_series,
            (clay_Sse - compressibility_of_water),
            (clay_Ssv - compressibility_of_water),
            dz_clays,
            preset_precons=preset_precons,
            ic_precons=initial_condition_precons,
        )

    deformation["total"] = np.array([groundwater_solution_dates, totdeftmp])
    deformation["inelastic_flag"] = inelastic_flag_tmp

    return deformation


# ---------------------------------------------------------------------------
# 3. Top-level driver
# ---------------------------------------------------------------------------

def solve_all_compaction(
    config: Dict[str, Any],
    stratigraphy: Dict[str, Any],
    head_solutions: Dict[str, Any],
    head_data: Dict[str, Any],
    effective_stress: Dict[str, Any],
    groundwater_solution_dates: Dict[str, Any],
    inelastic_flag: Dict[str, Any],
    initial_condition_precons: Dict[str, Any],
    overburden_dates: Optional[np.ndarray] = None,
    overburden_data: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Iterate over all compactable layers and solve compaction.

    This is the top-level entry point that replaces the inline compaction
    solver block in model.py.

    Parameters
    ----------
    config : dict
        Model configuration with at least the following keys:

        * ``"layer_names"`` -- ordered list of layer names.
        * ``"layer_types"`` -- ``{name: "Aquifer"|"Aquitard"}``.
        * ``"layer_compaction_switch"`` -- ``{name: bool}``.
        * ``"layer_thickness_types"`` -- ``{name: str}``.
        * ``"layer_thicknesses"`` -- ``{name: float|dict}``.
        * ``"interbeds_distributions"`` -- ``{name: {thickness: count}}``.
        * ``"sand_Sse"`` -- ``{name: float}``.
        * ``"clay_Sse"`` -- ``{name: float}``.
        * ``"clay_Ssv"`` -- ``{name: float}``.
        * ``"compressibility_of_water"`` -- float.
        * ``"dz_clays"`` -- ``{name: float}``.
        * ``"dt_master"`` -- ``{name: float}``.
        * ``"rho_w"`` -- float.
        * ``"g"`` -- float.
        * ``"overburden_stress_compaction"`` -- bool.
        * ``"compaction_solver_debug_include_endnodes"`` -- bool.
        * ``"MODE"`` -- ``"Normal"`` or ``"resume"``.
        * ``"unconfined_aquifer_name"`` -- str or None.
    stratigraphy : dict
        May carry ``"initial_thicknesses"`` for variable-thickness layers.
    head_solutions : dict
        ``{layer_name: head_series_data}`` -- for aquifers this is a dict
        with ``"Interconnected matrix"`` and ``"{thickness:.2f} clays"``
        keys; for aquitards it is a head matrix (nodes x time).
    head_data : dict
        Raw head data ``{layer_name: np.ndarray}`` with columns
        ``[date, head]``.
    effective_stress : dict
        Effective stress arrays (not directly used in compaction maths but
        passed through for completeness).
    groundwater_solution_dates : dict
        Date-number arrays for each layer/sub-layer.
    inelastic_flag : dict
        Existing inelastic flag arrays (populated as a side effect).
    initial_condition_precons : dict
        Initial preconsolidation arrays for each layer/sub-layer.
    overburden_dates : np.ndarray, optional
        Date-number array for overburden time series.
    overburden_data : np.ndarray, optional
        Overburden stress values (Pa).

    Returns
    -------
    dict
        ``{layer_name: deformation_dict}`` where each ``deformation_dict``
        is the return value of :func:`solve_layer_compaction_aquifer` or
        :func:`solve_layer_compaction_aquitard`.
    """
    layer_names = config["layer_names"]
    layer_types = config["layer_types"]
    layer_compaction_switch = config["layer_compaction_switch"]
    layer_thickness_types = config["layer_thickness_types"]
    layer_thicknesses = config["layer_thicknesses"]
    interbeds_distributions = config["interbeds_distributions"]
    sand_Sse = config["sand_Sse"]
    clay_Sse = config["clay_Sse"]
    clay_Ssv = config["clay_Ssv"]
    compressibility_of_water = config["compressibility_of_water"]
    dz_clays = config["dz_clays"]
    dt_master = config["dt_master"]
    rho_w = config["rho_w"]
    g_val = config["g"]
    overburden_stress_compaction = config["overburden_stress_compaction"]
    debug_endnodes = config["compaction_solver_debug_include_endnodes"]
    mode = config.get("MODE", "Normal")
    unconfined_aquifer_name = config.get("unconfined_aquifer_name", None)

    initial_thicknesses = stratigraphy.get("initial_thicknesses", {})

    preset_precons = (mode == "resume")

    deformation: Dict[str, Any] = {}

    for layer in layer_names:
        if not layer_compaction_switch.get(layer, False):
            continue

        if layer_types[layer] == "Aquifer":
            logger.info("%s is an Aquifer. Solving for layer compaction.", layer)

            # Determine initial thickness for this layer
            if layer_thickness_types[layer] == "step_changes":
                init_thick = initial_thicknesses[layer]
            else:
                init_thick = layer_thicknesses[layer]

            deformation[layer] = solve_layer_compaction_aquifer(
                layer_name=layer,
                head_series=head_solutions[layer],
                interbeds_distributions=interbeds_distributions[layer],
                sand_Sse=sand_Sse[layer],
                compressibility_of_water=compressibility_of_water,
                layer_thickness=layer_thicknesses[layer],
                layer_thickness_type=layer_thickness_types[layer],
                initial_thickness=init_thick,
                clay_Sse=clay_Sse[layer],
                clay_Ssv=clay_Ssv[layer],
                dz_clays=dz_clays[layer],
                dt_master=dt_master[layer],
                groundwater_solution_dates=groundwater_solution_dates[layer],
                overburden_stress_compaction=overburden_stress_compaction,
                overburden_dates=overburden_dates,
                overburden_data=overburden_data,
                rho_w=rho_w,
                g=g_val,
                compaction_solver_debug_include_endnodes=debug_endnodes,
                preset_precons=preset_precons,
                initial_condition_precons=initial_condition_precons[layer],
                unconfined_aquifer_name=unconfined_aquifer_name,
            )

        elif layer_types[layer] == "Aquitard":
            logger.info("%s is an Aquitard. Solving for layer compaction.", layer)

            deformation[layer] = solve_layer_compaction_aquitard(
                layer_name=layer,
                head_series=head_solutions[layer],
                clay_Sse=clay_Sse[layer],
                clay_Ssv=clay_Ssv[layer],
                compressibility_of_water=compressibility_of_water,
                dz_clays=dz_clays[layer],
                groundwater_solution_dates=groundwater_solution_dates[layer],
                overburden_stress_compaction=overburden_stress_compaction,
                overburden_dates=overburden_dates,
                overburden_data=overburden_data,
                rho_w=rho_w,
                g=g_val,
                compaction_solver_debug_include_endnodes=debug_endnodes,
                preset_precons=preset_precons,
                initial_condition_precons=initial_condition_precons[layer],
                unconfined_aquifer_name=unconfined_aquifer_name,
            )

    return deformation


# ---------------------------------------------------------------------------
# 4. Thickness scaling for step-change layers
# ---------------------------------------------------------------------------

def scale_by_varying_thickness(
    deformation: np.ndarray,
    layer_thicknesses: Dict[str, float],
    initial_thickness: float,
    deformation_dates: np.ndarray,
) -> np.ndarray:
    """Scale a deformation time series for a layer with step-change thickness.

    The ``layer_thicknesses`` dict uses the following key conventions:

    * ``"pre-YEAR"`` -- thickness applicable from the start of the record up
      to September 1 of YEAR.
    * ``"YEAR1-YEAR2"`` -- thickness applicable from September 1 of YEAR1 to
      September 1 of YEAR2.
    * ``"YEAR-"`` -- thickness applicable from September 1 of YEAR to the
      end of the record.

    Within each period the deformation is scaled by
    ``thickness / initial_thickness``.  At period boundaries the scaled
    series is made continuous by adding the accumulated scaled deformation
    from the preceding period.

    Parameters
    ----------
    deformation : np.ndarray
        1-D array of deformation values (m), one per date in
        *deformation_dates*.
    layer_thicknesses : dict
        ``{period_key: thickness_m}`` as described above.
    initial_thickness : float
        Reference thickness used to normalise the scaling factor.
    deformation_dates : np.ndarray
        Matplotlib date-number array corresponding to *deformation*.

    Returns
    -------
    np.ndarray
        Scaled deformation array of the same length as *deformation*.
    """
    datetimedates = mdates.num2date(deformation_dates)

    # --- Find the "pre-YEAR" key ---
    prekeyname = None
    for key in layer_thicknesses:
        if "pre" in key:
            prekeyname = key
            break
    if prekeyname is None:
        raise ValueError(
            "layer_thicknesses must contain a 'pre-YEAR' key for "
            "step_changes thickness type."
        )

    pre_year = int(prekeyname.split("-")[1])
    pre_cutoff = datetime.datetime(pre_year, 9, 1, tzinfo=datetime.timezone.utc)

    # Scale the "pre" period
    scaling_factor = layer_thicknesses[prekeyname] / initial_thickness
    logical_pre = np.array([dt <= pre_cutoff for dt in datetimedates])
    deformation_scaled = deformation[logical_pre] * scaling_factor

    # --- Non-pre keys sorted (bounded ranges first, then trailing) ---
    nonpre_keys = sorted(
        [k for k in layer_thicknesses if "pre" not in k]
    )

    # First pass: bounded ranges ("YEAR1-YEAR2")
    for key in nonpre_keys:
        if key.endswith("-"):
            continue
        years = key.split("-")
        year_start = int(years[0])
        year_end = int(years[1])
        scaling_factor = layer_thicknesses[key] / initial_thickness
        logger.info(
            "\t\tScaling years %s by %s", years, scaling_factor,
        )
        dt_start = datetime.datetime(year_start, 9, 1, tzinfo=datetime.timezone.utc)
        dt_end = datetime.datetime(year_end, 9, 1, tzinfo=datetime.timezone.utc)
        logical_range = np.array([
            (dt > dt_start) and (dt <= dt_end) for dt in datetimedates
        ])
        if not np.any(logical_range):
            continue
        first_idx = np.where(logical_range)[0][0]
        deformation_scaled = np.append(
            deformation_scaled,
            (deformation[logical_range] - deformation[first_idx - 1])
            * scaling_factor
            + deformation_scaled[-1],
        )

    # Second pass: trailing ranges ("YEAR-")
    for key in nonpre_keys:
        if not key.endswith("-"):
            continue
        years = key.split("-")
        year_start = int(years[0])
        scaling_factor = layer_thicknesses[key] / initial_thickness
        logger.info(
            "\t\tScaling years %s by %s", years, scaling_factor,
        )
        dt_start = datetime.datetime(year_start, 9, 1, tzinfo=datetime.timezone.utc)
        logical_range = np.array([dt > dt_start for dt in datetimedates])
        if not np.any(logical_range):
            continue
        first_idx = np.where(logical_range)[0][0]
        deformation_scaled = np.append(
            deformation_scaled,
            (deformation[logical_range] - deformation[first_idx - 1])
            * scaling_factor
            + deformation_scaled[-1],
        )

    return deformation_scaled


# ---------------------------------------------------------------------------
# 5. Aggregate deformation across layers
# ---------------------------------------------------------------------------

def aggregate_deformation(
    deformation_by_layer: Dict[str, Dict[str, Any]],
    layer_compaction_switch: Dict[str, bool],
) -> pd.DataFrame:
    """Sum deformation across all compacting layers into a single DataFrame.

    Parameters
    ----------
    deformation_by_layer : dict
        ``{layer_name: deformation_dict}`` as returned by
        :func:`solve_all_compaction`.  Each ``deformation_dict`` must
        contain a ``"total"`` key holding a 2-D array
        ``[[dates], [deformation]]``.
    layer_compaction_switch : dict
        ``{layer_name: bool}`` indicating which layers should be included.

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``"dates"`` -- date strings (``"%d-%b-%Y"``).
        * One column per compacting layer with per-layer deformation.
        * ``"Total"`` -- the sum of all per-layer columns.

    Notes
    -----
    All layer deformation series are resampled onto the coarsest common
    date grid using ``np.isin`` so that they can be summed element-wise.
    """
    compacting_layers = [
        name for name, switch in layer_compaction_switch.items() if switch
    ]

    if not compacting_layers:
        logger.warning("No compacting layers found; returning empty DataFrame.")
        return pd.DataFrame()

    # Find the layer with the coarsest time-step (most widely spaced dates)
    # and use its date array as the common output grid.
    all_date_arrays = {
        layer: deformation_by_layer[layer]["total"][0, :]
        for layer in compacting_layers
    }

    # Use the first layer's dates as a starting common grid and
    # intersect with all others to get dates common to every layer.
    common_dates = all_date_arrays[compacting_layers[0]]
    for layer in compacting_layers[1:]:
        common_dates = common_dates[np.isin(common_dates, all_date_arrays[layer])]

    t_total_tmp = common_dates

    output: Dict[str, Any] = {}
    output["dates"] = [
        x.strftime("%d-%b-%Y") for x in mdates.num2date(t_total_tmp)
    ]

    t_overall = np.zeros_like(t_total_tmp, dtype=float)

    for layer in compacting_layers:
        layer_dates = deformation_by_layer[layer]["total"][0, :]
        layer_def = deformation_by_layer[layer]["total"][1, :]
        mask = np.isin(layer_dates, t_total_tmp)
        resampled = layer_def[mask]
        output[layer] = resampled
        t_overall += resampled

    output["Total"] = t_overall

    return pd.DataFrame(output)
