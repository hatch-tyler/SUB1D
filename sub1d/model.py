"""SUB1D model orchestrator.

This module provides the top-level ``run_model`` function that coordinates
the full simulation pipeline: configuration loading, head data I/O,
groundwater flow solution, compaction solving, and output generation.
"""
from __future__ import annotations

import os
import copy
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from sub1d.config import ModelConfig
from sub1d.layers import Stratigraphy
from sub1d.head_io import (
    read_head_data,
    clip_head_timeseries,
    compute_overburden_stress,
    interpolate_head_series,
    apply_resume_constant_head,
)
from sub1d.solver import (
    solve_head_equation_single,
    solve_head_equation_elastic_inelastic,
    solve_head_equation_crank_nicolson,
    solve_head_equation_cn_elastic_inelastic,
    solve_compaction_elastic_inelastic,
)
from sub1d.compaction import (
    solve_all_compaction,
    aggregate_deformation,
    scale_by_varying_thickness,
)
from sub1d.output import save_head_outputs, save_compaction_outputs, save_array
from sub1d.plotting import (
    plot_head_timeseries,
    plot_overburden_stress,
    plot_clay_distributions,
    plot_deformation,
    plot_total_deformation,
)
from sub1d.utils import make_output_folder

logger = logging.getLogger(__name__)


def run_model(config: ModelConfig,
              solver_override: Optional[str] = None,
              skip_compaction: bool = False) -> dict:
    """Run the full SUB1D subsidence model.

    Parameters
    ----------
    config : ModelConfig
        Complete model configuration.
    solver_override : str, optional
        Override solver type: "explicit" or "crank-nicolson".
    skip_compaction : bool
        If True, only solve the head equations (skip compaction).

    Returns
    -------
    dict
        Results dictionary with keys:
        - "head_solutions": head matrices per layer
        - "inelastic_flag": inelastic flags per layer
        - "effective_stress": effective stress per layer
        - "deformation": deformation per layer (if compaction solved)
    """
    t_total_start = time.time()

    # --- Setup output directory ---
    outdestination = os.path.join(config.admin.output_folder, config.admin.run_name)
    make_output_folder("config", outdestination, overwrite=config.admin.overwrite)

    # --- Build stratigraphy ---
    strat = Stratigraphy.from_config(config)
    logger.info("Hydrostratigraphy:")
    for name in strat.layer_names:
        logger.info("  %s (%s)", name, strat.layer_type(name))
    logger.info("Layers requiring solving: %s", strat.layers_requiring_solving)

    # --- Read head data ---
    logger.info("Reading input head data")
    t_read_start = time.time()

    aquifers_needing_heads = strat.all_aquifers_needing_head_data
    head_data = read_head_data(config.head_data_files, output_dir=outdestination)

    # --- Clip timeseries ---
    resume_start = None
    if config.mode == "resume" and config.resume_date:
        resume_start = pd.to_datetime(config.resume_date)

    head_data, starttime, endtime = clip_head_timeseries(
        head_data, start=resume_start,
    )

    # Apply resume constant head if needed
    if config.mode == "resume" and config.resume_head_value:
        head_data = apply_resume_constant_head(head_data, config.resume_head_value)

    # --- Compute overburden stress ---
    overburden_data = None
    overburden_dates = None
    unconfined_aquifer_name = None

    if config.overburden_stress_gwflow or config.overburden_stress_compaction:
        unconfined_aquifer_name = config.layer_names[0]
        logger.info("Computing overburden stress from %s", unconfined_aquifer_name)

        overburden_df = compute_overburden_stress(
            head_data, unconfined_aquifer_name,
            config.hydro.specific_yield or 0.2,
            config.hydro.rho_w, config.hydro.g,
        )
        overburden_dates = overburden_df.iloc[:, 0]
        overburden_data = overburden_df.iloc[:, 1].values

        input_copy = os.path.join(outdestination, "input_data")
        os.makedirs(input_copy, exist_ok=True)
        plot_overburden_stress(overburden_df, input_copy)
        overburden_df.to_csv(os.path.join(input_copy, "overburden_data.csv"), index=False)

    # --- Plot input head data ---
    input_copy = os.path.join(outdestination, "input_data")
    os.makedirs(input_copy, exist_ok=True)
    plot_head_timeseries(head_data, aquifers_needing_heads, input_copy)

    # Save clipped head data
    for aquifer in aquifers_needing_heads:
        safe_name = aquifer.replace(" ", "_")
        head_data[aquifer].to_csv(
            os.path.join(input_copy, f"input_time_series_{safe_name}.csv"),
            index=False,
        )

    t_read_elapsed = time.time() - t_read_start
    logger.info("Head data reading and processing: %.1fs", t_read_elapsed)

    # --- Plot clay distributions ---
    layer_thicknesses = {lc.name: lc.thickness for lc in config.layers}
    layer_thickness_types = {lc.name: lc.thickness_type for lc in config.layers}
    interbeds_distributions = {}
    for lc in config.layers:
        if lc.interbeds_distributions:
            interbeds_distributions[lc.name] = lc.interbeds_distributions

    plot_clay_distributions(
        strat.layers_requiring_solving, config.layer_types,
        layer_thickness_types, layer_thicknesses,
        interbeds_distributions, input_copy,
    )

    # --- Solve head equations in clay layers ---
    logger.info("Solving head equations in clay layers")
    t_solve_start = time.time()

    head_solutions = {}
    inelastic_flag = {}
    effective_stress = {}
    groundwater_solution_dates = {}
    Z = {}
    t_gwflow = {}
    initial_condition_precons = {}

    rho_w = config.hydro.rho_w
    g = config.hydro.g

    layers_to_solve = strat.layers_requiring_solving
    use_parallel = (
        config.solver.parallel_layers
        and len(layers_to_solve) > 1
    )

    def _solve_layer(layer):
        """Solve a single layer, return result dict."""
        logger.info("Solving for layer: %s", layer)
        # Each call mutates its own local dicts
        hs, ifl, es, gwd, z_d, tg, icp = {}, {}, {}, {}, {}, {}, {}
        if strat.is_aquitard(layer):
            _solve_aquitard_head(
                layer, config, strat, head_data, overburden_data,
                overburden_dates, unconfined_aquifer_name,
                hs, ifl, es, gwd, z_d, tg, icp, solver_override,
            )
        elif strat.is_aquifer(layer) and strat.has_interbeds(layer):
            _solve_aquifer_interbeds_head(
                layer, config, strat, head_data, overburden_data,
                overburden_dates, unconfined_aquifer_name,
                hs, ifl, es, gwd, z_d, tg, icp, solver_override,
            )
        return hs, ifl, es, gwd, z_d, tg, icp

    if use_parallel:
        logger.info("Solving %d layers in parallel (ThreadPoolExecutor).", len(layers_to_solve))
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_solve_layer, layer): layer for layer in layers_to_solve}
            for future in as_completed(futures):
                hs, ifl, es, gwd, z_d, tg, icp = future.result()
                head_solutions.update(hs)
                inelastic_flag.update(ifl)
                effective_stress.update(es)
                groundwater_solution_dates.update(gwd)
                Z.update(z_d)
                t_gwflow.update(tg)
                initial_condition_precons.update(icp)
    else:
        for layer in layers_to_solve:
            hs, ifl, es, gwd, z_d, tg, icp = _solve_layer(layer)
            head_solutions.update(hs)
            inelastic_flag.update(ifl)
            effective_stress.update(es)
            groundwater_solution_dates.update(gwd)
            Z.update(z_d)
            t_gwflow.update(tg)
            initial_condition_precons.update(icp)

    t_solve_elapsed = time.time() - t_solve_start
    logger.info("Head solving: %.1fs", t_solve_elapsed)

    # --- Save head outputs ---
    if config.output.save_output_head_timeseries or config.output.save_effective_stress:
        save_head_outputs(
            head_solutions, groundwater_solution_dates,
            inelastic_flag, effective_stress,
            config.layer_types, Z, t_gwflow,
            config.solver.dt_master, interbeds_distributions,
            outdestination, config,
        )

    results = {
        "head_solutions": head_solutions,
        "inelastic_flag": inelastic_flag,
        "effective_stress": effective_stress,
        "groundwater_solution_dates": groundwater_solution_dates,
    }

    # --- Solve compaction ---
    if not skip_compaction:
        logger.info("Solving compaction equations")
        t_compact_start = time.time()

        deformation = solve_all_compaction(
            config, strat, head_solutions, head_data,
            effective_stress, groundwater_solution_dates,
            inelastic_flag, initial_condition_precons,
            overburden_dates=overburden_dates,
            overburden_data=overburden_data,
        )

        results["deformation"] = deformation
        t_compact_elapsed = time.time() - t_compact_start
        logger.info("Compaction solving: %.1fs", t_compact_elapsed)

        # Save compaction outputs
        layer_compaction_switch = {lc.name: lc.compaction_switch for lc in config.layers}
        save_compaction_outputs(
            {}, deformation, config.layer_names, config.layer_types,
            layer_compaction_switch, outdestination, config.output.save_s,
        )

    t_total = time.time() - t_total_start
    logger.info("Total model runtime: %.1f seconds", t_total)

    return results


def _solve_aquitard_head(
    layer: str, config: ModelConfig, strat: Stratigraphy,
    head_data: dict, overburden_data, overburden_dates,
    unconfined_aquifer_name: str | None,
    head_solutions: dict, inelastic_flag_dict: dict,
    effective_stress: dict, gw_dates: dict, Z: dict, t_gwflow: dict,
    ic_precons: dict, solver_override: str | None,
) -> None:
    """Solve head equation for an aquitard layer."""
    top_boundary = strat.aquifer_above(layer)
    bot_boundary = strat.aquifer_below(layer)

    if not top_boundary or not bot_boundary:
        logger.warning("Aquitard %s missing bounding aquifers, skipping", layer)
        return

    logger.info("  Aquitard bounded by %s (above) and %s (below)",
                top_boundary, bot_boundary)

    dt = config.solver.dt_master[layer]
    dz = config.solver.dz_clays[layer]
    rho_w = config.hydro.rho_w
    g = config.hydro.g

    # Get boundary head timeseries
    t_top = mdates.date2num(head_data[top_boundary].iloc[:, 0].to_numpy())
    h_top = head_data[top_boundary].iloc[:, 1].to_numpy()
    t_bot = mdates.date2num(head_data[bot_boundary].iloc[:, 0].to_numpy())
    h_bot = head_data[bot_boundary].iloc[:, 1].to_numpy()

    # Interpolate to solver timestep
    top_interp = interpolate_head_series(t_top, h_top, dt)
    bot_interp = interpolate_head_series(t_bot, h_bot, dt)

    t_in = top_interp[:, 0]
    z = np.arange(0, _layer_thickness(config, layer) + 1e-5, dz)
    Z[layer] = z
    t_gwflow[layer] = t_in
    gw_dates[layer] = t_in

    # Prepare boundary conditions
    bc = np.vstack([top_interp[:, 1], bot_interp[:, 1]])

    # Initial condition
    ic = (top_interp[0, 1] + bot_interp[0, 1]) / 2.0
    ic_precons[layer] = np.array([])

    # Overburden
    ob_data = _get_overburden_for_layer(
        layer, t_in, overburden_data, overburden_dates,
        unconfined_aquifer_name, rho_w, g,
    )

    # Solve
    solver_type = config.solver.groundwater_flow_solver_type.get(layer, "elastic-inelastic")
    use_cn = (
        (solver_override == "crank-nicolson")
        if solver_override
        else (config.solver.default_solver == "crank-nicolson")
    )

    mb_check = config.solver.mass_balance_check

    t1 = time.time()
    if solver_type == "singlevalue":
        k = config.solver.vertical_conductivity[layer] / (
            config.hydro.clay_Ssk[layer] + config.hydro.compressibility_of_water
        )
        if use_cn:
            hmat = solve_head_equation_crank_nicolson(
                dt, t_in, dz, z, bc, ic, k,
                mass_balance_check=mb_check,
            )
        else:
            hmat = solve_head_equation_single(
                dt, t_in, dz, z, bc, ic, k,
                mass_balance_check=mb_check,
            )
        inelastic_flag_tmp = np.zeros_like(hmat)
    else:
        k_elastic = config.solver.vertical_conductivity[layer] / config.hydro.clay_Sse[layer]
        k_inelastic = config.solver.vertical_conductivity[layer] / config.hydro.clay_Ssv[layer]
        if use_cn:
            hmat, inelastic_flag_tmp = solve_head_equation_cn_elastic_inelastic(
                dt, t_in, dz, z, bc, ic, k_elastic, k_inelastic,
                overburden_stress=config.overburden_stress_gwflow,
                overburden_data=ob_data,
                mass_balance_check=mb_check,
            )
        else:
            hmat, inelastic_flag_tmp = solve_head_equation_elastic_inelastic(
                dt, t_in, dz, z, bc, ic, k_elastic, k_inelastic,
                overburden_stress=config.overburden_stress_gwflow,
                overburden_data=ob_data,
                mass_balance_check=mb_check,
            )

    logger.info("  Solved in %.1fs", time.time() - t1)

    head_solutions[layer] = hmat
    inelastic_flag_dict[layer] = inelastic_flag_tmp

    # Compute effective stress
    if config.overburden_stress_gwflow and ob_data is not None:
        effective_stress[layer] = (
            np.tile(ob_data, (hmat.shape[0], 1)) - rho_w * g * hmat
        )
    else:
        effective_stress[layer] = -rho_w * g * hmat


def _solve_aquifer_interbeds_head(
    layer: str, config: ModelConfig, strat: Stratigraphy,
    head_data: dict, overburden_data, overburden_dates,
    unconfined_aquifer_name: str | None,
    head_solutions: dict, inelastic_flag_dict: dict,
    effective_stress: dict, gw_dates: dict, Z: dict, t_gwflow: dict,
    ic_precons: dict, solver_override: str | None,
) -> None:
    """Solve head equation for interbeds within an aquifer layer."""
    dt = config.solver.dt_master[layer]
    dz = config.solver.dz_clays[layer]
    rho_w = config.hydro.rho_w
    g = config.hydro.g

    head_solutions[layer] = {"Interconnected matrix": head_data[layer]}
    inelastic_flag_dict[layer] = {}
    effective_stress[layer] = {}
    gw_dates[layer] = {}
    Z[layer] = {}
    t_gwflow[layer] = {}
    ic_precons[layer] = {}

    dist = strat.get_interbeds_distribution(layer)
    if not dist:
        return

    # Get aquifer head timeseries
    t_aquifer = mdates.date2num(head_data[layer].iloc[:, 0].to_numpy())
    h_aquifer = head_data[layer].iloc[:, 1].to_numpy()

    for thickness in dist.keys():
        logger.info("  Solving interbed thickness %.2f in %s", thickness, layer)

        z_tmp = np.arange(0, thickness + 1e-5, dz)

        # Interpolate aquifer head to solver dt
        interp = interpolate_head_series(t_aquifer, h_aquifer, dt)
        t_interp = interp[:, 0]
        h_interp = interp[:, 1]

        key = f"{thickness:.2f} clays"
        Z[layer][key] = z_tmp
        t_gwflow[layer][key] = t_interp
        gw_dates[layer][key] = t_interp
        ic_precons[layer][key] = np.array([])

        # BCs: same aquifer head on both sides for interbeds
        bc = np.vstack([h_interp, h_interp])
        ic = h_aquifer[0]

        # Overburden
        ob_data = _get_overburden_for_layer(
            layer, t_interp, overburden_data, overburden_dates,
            unconfined_aquifer_name, rho_w, g,
        )

        use_cn = (
            (solver_override == "crank-nicolson")
            if solver_override
            else (config.solver.default_solver == "crank-nicolson")
        )
        k_elastic = config.solver.vertical_conductivity[layer] / config.hydro.clay_Sse[layer]
        k_inelastic = config.solver.vertical_conductivity[layer] / config.hydro.clay_Ssv[layer]

        mb_check = config.solver.mass_balance_check

        t1 = time.time()
        if use_cn:
            hmat, ifl = solve_head_equation_cn_elastic_inelastic(
                dt, t_interp, dz, z_tmp, bc, ic, k_elastic, k_inelastic,
                overburden_stress=config.overburden_stress_gwflow,
                overburden_data=ob_data,
                mass_balance_check=mb_check,
            )
        else:
            hmat, ifl = solve_head_equation_elastic_inelastic(
                dt, t_interp, dz, z_tmp, bc, ic, k_elastic, k_inelastic,
                overburden_stress=config.overburden_stress_gwflow,
                overburden_data=ob_data,
                mass_balance_check=mb_check,
            )
        logger.info("    Solved in %.1fs", time.time() - t1)

        head_solutions[layer][key] = hmat
        inelastic_flag_dict[layer][key] = ifl

        if config.overburden_stress_gwflow and ob_data is not None:
            effective_stress[layer][key] = (
                np.tile(ob_data, (hmat.shape[0], 1)) - rho_w * g * hmat
            )
        else:
            effective_stress[layer][key] = -rho_w * g * hmat


def _layer_thickness(config: ModelConfig, layer: str) -> float:
    """Get the scalar thickness for a layer."""
    for lc in config.layers:
        if lc.name == layer:
            if isinstance(lc.thickness, dict):
                # For variable thickness, get the first (pre-) value
                for key, val in lc.thickness.items():
                    if "pre" in str(key):
                        return float(val)
                return float(list(lc.thickness.values())[0])
            return float(lc.thickness)
    return 0.0


def _get_overburden_for_layer(
    layer: str, t_solver: np.ndarray,
    overburden_data, overburden_dates,
    unconfined_aquifer_name: str | None,
    rho_w: float, g: float,
) -> np.ndarray | None:
    """Prepare overburden data array for a specific layer's solver grid."""
    if overburden_data is None:
        return None

    if layer == unconfined_aquifer_name:
        return np.zeros_like(t_solver)

    # Interpolate overburden to solver timestep
    if overburden_dates is not None:
        ob_dates_num = mdates.date2num(overburden_dates.to_numpy()) if hasattr(
            overburden_dates, 'to_numpy') else np.asarray(overburden_dates, dtype=float)
        ob_vals = np.asarray(overburden_data, dtype=float)
        return (1.0 / (rho_w * g)) * np.interp(t_solver, ob_dates_num, ob_vals)

    return None
