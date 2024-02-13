#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### This is the main model script. This is where all the "under the hood" operations occur. Do not edit unless you know what you are doing!! The correct call should be: python execute_model.py parameter_file.par
print("LOADING LIBRARIES -- MAY TAKE UP TO 15 MINUTES")

import os
import sys
import csv
import copy
import time
import shutil
import matplotlib
import datetime
import subprocess
import scipy
import operator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from netCDF4 import Dataset

print("LIBRARIES LOADED -- INITIALZING FUNCTIONS")

# =============================================================================
# ===============================MODEL FUNCTIONS==============================-

from utils import Logger
from solver import (
    solve_head_equation_singlevalue,
    solve_head_equation_elasticinelastic,
    subsidence_solver_aquitard_elasticinelastic,
)
from parameters import (
    read_parameters_admin,
    read_parameters_noadmin,
    read_parameter,
    read_parameter_layerthickness_multitype,
    parse_parameter_file,
)
from plotting import (
    create_head_video_elasticinelastic,
    plot_overburden_stress,
    plot_head_timeseries,
    plot_clay_distributions,
)

# =============================================================================
# ===============================RUN MODEL==============================-
print("FUNCTIONS LOADED -- RUNNING MODEL")

# get user input from command line
param_filename = input("Parameter File with Extension: ")
UserOverwriteYes = input(
    "If output folder already exists, and overwrite flag is specified as Y, Do you want to overwrite the old data? (Y/N): "
)

t_total_start = time.time()

# set this flag to True to save head outputs as netCDF grid files.
# NOTE: this requires gmt to be installed and uses the gmt xyz2grd command.
gmt = False

sns.set_context("talk")

# if len(sys.argv) != 2:   # Commented this out since we are now running from Spyder (PW)
#    print('Execute model error; terminal. Incorrect number of input arguments. Correct usage: python execute_model.py parameter_file.par')
#    sys.exit(1)

print("\n\n" + "".center(80, "*"))
print("  READING PARAMETERS  ".center(80, "*"))
print("".center(80, "*") + "\n")
param_read_start = time.time()
# time.sleep(0.5)

print(f"Reading parameters from file: {param_filename}")

# Open the parameter file and parse it
paramfilelines = parse_parameter_file(param_filename)

# Read in parameters
MODE = read_parameter("mode", str, 1, paramfilelines)

(
    internal_time_delay,
    overwrite,
    run_name,
    output_folder,
    outdestination,
) = read_parameters_admin(param_filename, paramfilelines, UserOverwriteYes)

if MODE == "resume":
    resume_directory = read_parameter("resume_directory", str, 1, paramfilelines)
    if not resume_directory:
        print("\t\tTerminal error: resume_directory not set.")
        sys.exit(1)
    if resume_directory == run_name:
        print("\t\tTerminal error: resume_directory same as output folder.")
        sys.exit(1)

    resume_date = read_parameter("resume_date", str, 1, paramfilelines)
    resume_date = pd.to_datetime(resume_date, format="%b-%d-%Y")
    print(f"\t\tResume date read in as {resume_date.stftime('%d-%b-%Y')}")

    no_layers = read_parameter("no_layers", int, 1, paramfilelines)
    layer_types = read_parameter("layer_types", str, no_layers, paramfilelines)
    no_aquifers = list(layer_types.values()).count("Aquifer")
    resume_head_value = read_parameter(
        "resume_head_value", str, no_aquifers, paramfilelines
    )
    resume_layer_thicknesses = read_parameter_layerthickness_multitype(
        "layer_thicknesses", paramfilelines
    )

    print(
        f"*** MODE is RESUME; reading all non-admin parameters from paramfile {os.path.join(resume_directory, 'paramfile.par')} ***"
    )
    shutil.copy2(
        f"{resume_directory}/paramfile.par",
        f"{outdestination}/resume_paramfile.par",
    )

    # parse parameters from resume parameter file
    paramfilelines = parse_parameter_file(f"{resume_directory}/paramfile.par")

(
    save_output_head_timeseries,
    save_effective_stress,
    save_internal_compaction,
    no_layers,
    layer_names,
    layer_types,
    no_aquifers,
    no_aquitards,
    layer_thickness_types,
    layer_thicknesses,
    layer_compaction_switch,
    interbeds_switch,
    interbeds_distributions,
    aquitards,
    interbedded_layers,
    no_layers_containing_clay,
    layers_requiring_solving,
    create_output_head_video,
    groundwater_flow_solver_type,
    overburden_stress_gwflow,
    compaction_solver_compressibility_type,
    compaction_solver_debug_include_endnodes,
    clay_Sse,
    clay_Ssv,
    clay_Ssk,
    sand_Sse,
    time_unit,
    sand_Ssk,
    compressibility_of_water,
    rho_w,
    g,
    dt_master,
    dz_clays,
    vertical_conductivity,
    overburden_stress_compaction,
    specific_yield,
    preconsolidation_head_type,
    preconsolidation_head_offset,
    save_s,
) = read_parameters_noadmin(paramfilelines)

# Check that the layer thicknesses were correctly imported
print("PARAMETER CHECK: checking layer_thickness_types and layer_thicknesses...")

layers_cst_thickness = [k for k, v in layer_thickness_types.items() if v == "constant"]
layers_var_thickness = [k for k, v in layer_thickness_types.items() if v != "constant"]
for layer in layers_cst_thickness:
    if isinstance(layer_thicknesses[layer], dict):
        print(
            f"ERROR, terminal, layer thicknesses for {layer} is {layer_thicknesses[layer]}. It's a constant thickness layer so shouldn't be a dictionary."
        )
        sys.exit(1)
    else:
        print(f"\t{layer} look good.")
for layer in layers_var_thickness:
    if not isinstance(layer_thicknesses[layer], dict):
        print(
            f"ERROR, terminal, layer thicknesses for {layer} is {layer_thicknesses[layer]}. It's a variable thickness layer so should be a dictionary."
        )
        sys.exit(1)
    else:
        print(f"\t{layer} look good.")

for layer in layers_var_thickness:
    if layer_thickness_types[layer] == "step_changes":
        tmp = layer_thicknesses[layer]
        pre = ["pre" in s for s in tmp]
        if sum(pre) == 1:
            print(f"\tExactly 1 'pre' entry for {layer}, looks good.")
        else:
            print(
                f"\tERROR:terminal. {layer} is variable thickness but doesn't have a pre entry. Needs fixing!."
            )
            sys.exit(1)

if len(layers_var_thickness) >= 1:
    initial_thicknesses = {}
    for layer in layers_var_thickness:
        prekeyname = np.array(list(layer_thicknesses[layer].keys()))[
            np.where(["pre" in key for key in list(layer_thicknesses[layer].keys())])[
                0
            ][0]
        ]
        initial_thicknesses[layer] = layer_thicknesses[layer][prekeyname]
    print(
        f"\tInitial thicknesses for varying aquifer thicknesses are {initial_thicknesses}."
    )

if MODE == "resume":
    print(
        "\t MODE is RESUME, therefore overriding original layer thicknesses with resume layer thicknesses."
    )
    layer_thicknesses = resume_layer_thicknesses
    layers_var_thickness = []

param_read_stop = time.time()
param_read_time = param_read_start - param_read_stop

# %% Next section is "READING INPUT DATA MODULE"
print("\n\n" + "".center(80, "*"))
print("  READING INPUT DATA  ".center(80, "*"))
print("".center(80, "*") + "\n")

reading_head_start = time.time()
# time.sleep(internal_time_delay)


aquifer_layer_names = [
    name for name, layer_type in layer_types.items() if layer_type == "Aquifer"
]
compactable_aquifers_names = [
    name
    for name, switch in layer_compaction_switch.items()
    if name in aquifer_layer_names and switch
]
aquitard_locations = [
    layer_names.index(name)
    for name, layer_type in layer_types.items()
    if layer_type == "Aquitard"
]
aquifers_above_aquitards = [layer_names[i - 1] for i in aquitard_locations]
aquifers_below_aquitards = [layer_names[i + 1] for i in aquitard_locations]

all_aquifers_needing_head_data = list(
    set(
        aquifers_below_aquitards + aquifers_above_aquitards + compactable_aquifers_names
    )
)

dt_headseries = {}

print(
    f"Preparing to read in head data. Aquifers for which head data is required are: {', '.join(map(str, all_aquifers_needing_head_data))}."
)

if len(all_aquifers_needing_head_data) >= 0:
    # make input_data directory in the outdestination folder if it does not exist
    input_copy = os.path.normpath(os.path.join(outdestination, "input_data"))
    os.makedirs(input_copy, exist_ok=True)

    head_data_files = read_parameter(
        "head_data_files", str, len(all_aquifers_needing_head_data), paramfilelines
    )

    # create a copy of the head_data_files dictionary. values will be replaced by
    # a numpy array of numeric dates and values from the csv file containing heads
    head_data = copy.deepcopy(head_data_files)

    for aquifer in all_aquifers_needing_head_data:
        fileloc = head_data_files[aquifer]

        print(f"File for {aquifer} specified as {fileloc}. Looking for file.")

        if not os.path.isfile(fileloc):
            raise FileNotFoundError(
                f"Error reading head data. File {fileloc} not found."
            )

        print(f"File {fileloc} exists. Storing copy in output folder.")
        print("Reading in head time series.")

        # copy csv files to input_data folder in output directory
        shutil.copy2(fileloc, os.path.join(input_copy, os.path.basename(fileloc)))

        # read groundwater level data from csv file
        data = pd.read_csv(fileloc, parse_dates=[0])

        # TODO: need to understand why to convert dates to a number here...
        # try:
        #    dates = mdates.date2num(data.iloc[:, 0].values)
        # except Exception:
        #    print(
        #        "Pandas couldn't parse the date head. Going to try treating the date as a float. If it's not, things may fail from hereon."
        #    )
        #    dates = np.array([float(ting) for ting in data.iloc[:, 0].values])
        #    if time_unit == "years":
        #        dates = 365 * dates

        # data = data.iloc[:, 1].values
        # head_data[aquifer] = np.array([dates, data]).T
        head_data[aquifer] = data
        print(f"Successfully read in. Head data for {aquifer} printing now.")
        print(head_data[aquifer])
        dt_headseries[aquifer] = np.diff(head_data[aquifer].iloc[:, 0])[0]

    print("Reading head data complete.")
else:
    print("No aquifers requiring head data; skipping reading head data.")

# get the starting times for each aquifer layer
# starttimes = [
#    np.min(head_data[aquifer][:, 0]) for aquifer in all_aquifers_needing_head_data
# ]
starttimes = [
    head_data[aquifer].iloc[:, 0].min() for aquifer in all_aquifers_needing_head_data
]
print([t.strftime("%m-%d-%Y") for t in starttimes])

# start time is the maximum for all aquifer layers
starttime = np.max(starttimes)

# get ending times for each aquifer layer
# endtimes = [
#    np.max(head_data[aquifer][:, 0]) for aquifer in all_aquifers_needing_head_data
# ]
endtimes = [
    head_data[aquifer].iloc[:, 0].max() for aquifer in all_aquifers_needing_head_data
]

# end time is the minimum for all aquifer layers
endtime = np.min(endtimes)

print("CLIPPING HEAD TIMESERIES TO HAVE CONSISTENT START/END DATES ACROSS AQUIFERS.")
# print(
#    f"Latest startdate found is {mdates.num2date(starttime).strftime('%d-%b-%Y')} and earliest end date is {mdates.num2date(endtime).strftime('%d-%b-%Y')}. These will be used as model start/end times."
# )
print(
    f"Latest startdate found is {starttime.strftime('%d-%b-%Y')} and earliest end date is {endtime.strftime('%d-%b-%Y')}. These will be used as model start/end times."
)
# Clip to have common starting date
print("Clipping input series to model starttime.")
if MODE == "resume":
    starttime = resume_date
    print(
        f"\tHead startdate is resume date; clipping series accordingly to start at {resume_date.strftime('%d-%b-%Y')}."
    )

for aquifer in all_aquifers_needing_head_data:
    # create filter criteria for dates between the starttime and endtime
    idx_to_keep = (head_data[aquifer].iloc[:, 0] >= starttime) & (
        head_data[aquifer].iloc[:, 0] <= endtime
    )

    # filter head data between start time and endtime
    head_data[aquifer] = head_data[aquifer][idx_to_keep]

    print(f"Clipping for {aquifer} completed.")

    if MODE == "resume":
        if resume_head_value[aquifer] == "cst":
            print(
                f"Resume head value=cst. Setting head in aquifer layers to be constant at the value it was in {resume_date.strftime('%d-%b-%Y')}."
            )

            print(f"Setting head in {aquifer} to be {head_data[aquifer].iloc[:, 1][0]:.2f}.")
            head_data[aquifer].iloc[:, 1] = head_data[aquifer].iloc[:, 1][0]
            print(head_data[aquifer])

    # write head data to csv for each aquifer layer
    head_data[aquifer].to_csv(
        os.path.join(input_copy, f"input_time_series_{aquifer.replace(' ', '_')}.csv"),
        index=False,
    )

if overburden_stress_gwflow or overburden_stress_compaction:
    for aquifer in all_aquifers_needing_head_data:
        if aquifer == layer_names[0]:
            print(
                f"{aquifer} is the uppermost aquifer; therefore it is unconfined." 
                "Calculating overburden stress from changing water levels in this aquifer."
            )
            unconfined_aquifer_name = aquifer
            overburden_dates = head_data[aquifer].iloc[:, 0]
            
            # calculate overburden 
            overburden_data = (
                specific_yield
                * rho_w
                * g
                * (head_data[aquifer].iloc[:, 1] - head_data[aquifer].iloc[0, 1])
            )
            overburden_data.index = overburden_dates
            overburden_data = overburden_data.to_frame(name="Overburden Stress").reset_index()

    print("Clipping overburden stress.")
    idx_to_keep = (overburden_dates >= starttime) & (overburden_dates <= endtime)
    overburden_data = overburden_data[idx_to_keep]
    print("Clipping done.")

    # plot overburden data
    plot_overburden_stress(overburden_data, input_copy)

    # write overburden stress to csv
    overburden_data.to_csv(os.path.join(input_copy, "overburden_data.csv"), index=False)

    print("Overburden stress calculated and saved in input_data.")

effective_stress = {}

# plot input head timeseries
plot_head_timeseries(head_data, all_aquifers_needing_head_data, input_copy)

# get time at end of reading and plotting head timeseries
reading_head_stop = time.time()
reading_head_time = reading_head_stop - reading_head_start

if len(layers_requiring_solving) >= 0:
    print("Making input clay distribution plot.")
    plot_clay_distributions(
        layers_requiring_solving,
        layer_types,
        layer_thickness_types,
        layer_thicknesses,
        interbeds_distributions,
        input_copy,
    )

# %% New section, head solver.
print("\n\n" + "".center(80, "*"))
print("  SOLVING FOR HEAD TIME SERIES IN CLAY LAYERS  ".center(80, "*"))
print("".center(80, "*") + "\n")

# get start time for solving for heads in clay layers
solving_head_start = time.time()
# time.sleep(internal_time_delay)
#
print("Hydrostratigraphy:")
print("".center(60, "-"))
for layer in layer_names:
    text = f"{layer} ({layer_types[layer]})"
    print(text.center(60, " "))
    print("".center(60, "-"))

print(
    f"\n\nHead time series to be solved within the following layers: {layers_requiring_solving}"
)

inelastic_flag = {}
inelastic_flag_compaction = {}
Z = {}
t_gwflow = {}

head_series = copy.deepcopy(head_data)

initial_condition_precons = {}
# if save_output_head_timeseries:
#    os.mkdir(os.path.join(outdestination, "head_outputs"))

if len(all_aquifers_needing_head_data) >= 0:
    # make input_data directory in the outdestination folder if it does not exist
    input_copy = os.path.normpath(os.path.join(outdestination, "input_data"))
    os.makedirs(input_copy, exist_ok=True)

    head_data_files = read_parameter(
        "head_data_files", str, len(all_aquifers_needing_head_data), paramfilelines
    )

    # create a copy of the head_data_files dictionary. values will be replaced by
    # a numpy array of numeric dates and values from the csv file containing heads
    head_data = copy.deepcopy(head_data_files)

    for aquifer in all_aquifers_needing_head_data:
        fileloc = head_data_files[aquifer]

        print(f"File for {aquifer} specified as {fileloc}. Looking for file.")

        if not os.path.isfile(fileloc):
            raise FileNotFoundError(
                f"Error reading head data. File {fileloc} not found."
            )

        print(f"File {fileloc} exists. Storing copy in output folder.")
        print("Reading in head time series.")

        # copy csv files to input_data folder in output directory
        shutil.copy2(fileloc, os.path.join(input_copy, os.path.basename(fileloc)))

        # read groundwater level data from csv file
        data = pd.read_csv(fileloc, parse_dates=[0])

        # TODO: need to understand why to convert dates to a number here...
        # try:
        #    dates = mdates.date2num(data.iloc[:, 0].values)
        # except Exception:
        #    print(
        #        "Pandas couldn't parse the date head. Going to try treating the date as a float. If it's not, things may fail from hereon."
        #    )
        #    dates = np.array([float(ting) for ting in data.iloc[:, 0].values])
        #    if time_unit == "years":
        #        dates = 365 * dates

        # data = data.iloc[:, 1].values
        # head_data[aquifer] = np.array([dates, data]).T
        head_data[aquifer] = data
        print(f"Successfully read in. Head data for {aquifer} printing now.")
        print(head_data[aquifer])
        dt_headseries[aquifer] = np.diff(head_data[aquifer].iloc[:, 0])[0]

    print("Reading head data complete.")
else:
    print("No aquifers requiring head data; skipping reading head data.")

# get the starting times for each aquifer layer
# starttimes = [
#    np.min(head_data[aquifer][:, 0]) for aquifer in all_aquifers_needing_head_data
# ]
starttimes = [
    head_data[aquifer].iloc[:, 0].min() for aquifer in all_aquifers_needing_head_data
]
print([t.strftime("%m-%d-%Y") for t in starttimes])

# start time is the maximum for all aquifer layers
starttime = np.max(starttimes)

# get ending times for each aquifer layer
# endtimes = [
#    np.max(head_data[aquifer][:, 0]) for aquifer in all_aquifers_needing_head_data
# ]
endtimes = [
    head_data[aquifer].iloc[:, 0].max() for aquifer in all_aquifers_needing_head_data
]

# end time is the minimum for all aquifer layers
endtime = np.min(endtimes)

print("CLIPPING HEAD TIMESERIES TO HAVE CONSISTENT START/END DATES ACROSS AQUIFERS.")
# print(
#    f"Latest startdate found is {mdates.num2date(starttime).strftime('%d-%b-%Y')} and earliest end date is {mdates.num2date(endtime).strftime('%d-%b-%Y')}. These will be used as model start/end times."
# )
print(
    f"Latest startdate found is {starttime.strftime('%d-%b-%Y')} and earliest end date is {endtime.strftime('%d-%b-%Y')}. These will be used as model start/end times."
)
# Clip to have common starting date
print("Clipping input series to model starttime.")
if MODE == "resume":
    starttime = resume_date
    print(
        f"\tHead startdate is resume date; clipping series accordingly to start at {resume_date.strftime('%d-%b-%Y')}."
    )

for aquifer in all_aquifers_needing_head_data:
    # create filter criteria for dates between the starttime and endtime
    idx_to_keep = (head_data[aquifer].iloc[:, 0] >= starttime) & (
        head_data[aquifer].iloc[:, 0] <= endtime
    )

    # filter head data between start time and endtime
    head_data[aquifer] = head_data[aquifer][idx_to_keep]

    print(f"Clipping for {aquifer} completed.")

    if MODE == "resume":
        if resume_head_value[aquifer] == "cst":
            print(
                f"Resume head value=cst. Setting head in aquifer layers to be constant at the value it was in {resume_date.strftime('%d-%b-%Y')}."
            )

            print(f"Setting head in {aquifer} to be {head_data[aquifer].iloc[:, 1][0]:.2f}.")
            head_data[aquifer].iloc[:, 1] = head_data[aquifer].iloc[:, 1][0]
            print(head_data[aquifer])

    # write head data to csv for each aquifer layer
    head_data[aquifer].to_csv(
        os.path.join(input_copy, f"input_time_series_{aquifer.replace(' ', '_')}.csv"),
        index=False,
    )

if overburden_stress_gwflow or overburden_stress_compaction:
    for aquifer in all_aquifers_needing_head_data:
        if aquifer == layer_names[0]:
            print(
                f"{aquifer} is the uppermost aquifer; therefore it is unconfined." 
                "Calculating overburden stress from changing water levels in this aquifer."
            )
            unconfined_aquifer_name = aquifer
            overburden_dates = head_data[aquifer].iloc[:, 0]
            
            # calculate overburden 
            overburden_data = (
                specific_yield
                * rho_w
                * g
                * (head_data[aquifer].iloc[:, 1] - head_data[aquifer].iloc[0, 1])
            )
            overburden_data.index = overburden_dates
            overburden_data = overburden_data.to_frame(name="Overburden Stress").reset_index()

    print("Clipping overburden stress.")
    idx_to_keep = (overburden_dates >= starttime) & (overburden_dates <= endtime)
    overburden_data = overburden_data[idx_to_keep]
    print("Clipping done.")

    # plot overburden data
    plot_overburden_stress(overburden_data, input_copy)

    # write overburden stress to csv
    overburden_data.to_csv(os.path.join(input_copy, "overburden_data.csv"), index=False)

    print("Overburden stress calculated and saved in input_data.")

effective_stress = {}

# plot input head timeseries
plot_head_timeseries(head_data, all_aquifers_needing_head_data, input_copy)

# get time at end of reading and plotting head timeseries
reading_head_stop = time.time()
reading_head_time = reading_head_stop - reading_head_start

if len(layers_requiring_solving) >= 0:
    print("Making input clay distribution plot.")
    plot_clay_distributions(
        layers_requiring_solving,
        layer_types,
        layer_thickness_types,
        layer_thicknesses,
        interbeds_distributions,
        input_copy,
    )

# %% New section, head solver.
print("\n\n" + "".center(80, "*"))
print("  SOLVING FOR HEAD TIME SERIES IN CLAY LAYERS  ".center(80, "*"))
print("".center(80, "*") + "\n")

# get start time for solving for heads in clay layers
solving_head_start = time.time()
# time.sleep(internal_time_delay)
#
print("Hydrostratigraphy:")
print("".center(60, "-"))
for layer in layer_names:
    text = f"{layer} ({layer_types[layer]})"
    print(text.center(60, " "))
    print("".center(60, "-"))

print(
    f"\n\nHead time series to be solved within the following layers: {layers_requiring_solving}"
)

inelastic_flag = {}
inelastic_flag_compaction = {}
Z = {}
t_gwflow = {}

head_series = copy.deepcopy(head_data)

initial_condition_precons = {}
# if save_output_head_timeseries:
#    os.mkdir(os.path.join(outdestination, "head_outputs"))

if len(layers_requiring_solving) >= 0:
    groundwater_solution_dates = {}
    for layer in layers_requiring_solving:
        print(f"Beginning solving process for layer {layer}.")

        if layer_types[layer] == "Aquitard":
            print(f"{layer} is an aquitard.")

            # get the index of the aquitard in the layer_names list
            aquitard_position = layer_names.index(layer)

            # get the layer names of the top and bottom boundaries using the position of the aquitard in the layer_names list
            top_boundary = layer_names[aquitard_position - 1]
            bot_boundary = layer_names[aquitard_position + 1]

            # initialize the initial condition for preconsolidation head as an empty numpy array
            initial_condition_precons[layer] = np.array([])
            print(
                f"Head time series required for overlying layer {top_boundary} and lower layer {bot_boundary}."
            )
            if top_boundary in head_data.keys() and bot_boundary in head_data.keys():
                print("Head time series found.")

            # check if dt_master is specified
            if layer not in dt_master.keys():
                raise ValueError(
                    f"Error solving head series. dt_master not specified for layer {layer}."
                )

            # get times for the top boundary
            t_top = head_data[top_boundary].iloc[:, 0].to_numpy()
            
            # calculate the timestep in days for all dates in the head timeseries data
            dt_tmp = np.diff(mdates.date2num(t_top))
            print(dt_tmp)
            
            # check compatibility of dt_master and dt_tmp for aquitard
            # dt_master is a multiple of all dt_tmp for the layer
            test1 = [n.is_integer() for n in dt_master[layer] / dt_tmp]
            #test1 = np.all(np.mod(dt_master[layer], dt_tmp) == 0)
            print(test1)
            # dt_tmps are a multiple of dt_master for the layer
            test2 = [n.is_integer() for n in dt_tmp / dt_master[layer]]
            #test2 = np.all(np.mod(dt_tmp, dt_master[layer]) == 0)
            print(test2)
            # dt in the time series must be constant...if dt_tmp is monthly with units of day this doesn't appear to work
            test3 = list(dt_tmp).count(dt_tmp[0]) != len(dt_tmp)
            #test3 = np.all(dt_tmp == dt_tmp[0])
            print(test3)

            #if (not np.all(test1) and not np.all(test2)) or (
            #    not list(dt_tmp).count(dt_tmp[0])
            #) == len(dt_tmp):
            if (not np.all(test1) and not np.all(test2)) or test3:
                raise ValueError(
                    f"Error solving head series. dt_master not compatible with dt in the {top_boundary} input series. "
                    "dt_master must be an integer multiple of dt in the time series, and dt in the time series must be constant."
                )

            if dt_master[layer] >= dt_tmp[0]:
                spacing_top = int(dt_master[layer] / dt_tmp[0])
                top_head_tmp = head_data[top_boundary]

            else:
                print(
                    "\t\t\tNOTE: dt_master < dt_data. Linear resampling of input head series occuring."
                )
                t_interp_new = 0.0001 * np.arange(
                    10000 * min(t_top), 10000 * max(t_top) + 1, 10000 * dt_master[layer]
                )
                h_interp_new = np.interp(t_interp_new, t_top, head_data[top_boundary].iloc[:, 1].to_numpy())
                top_head_tmp = np.array([t_interp_new, h_interp_new]).T
                spacing_top = 1

            t_bot = head_data[bot_boundary][:, 0]
            dt_tmp = np.diff(t_bot)
            test1 = [n.is_integer() for n in dt_master[layer] / dt_tmp]
            test2 = [n.is_integer() for n in dt_tmp / dt_master[layer]]
            if (not np.all(test1) and not np.all(test2)) or (
                not list(dt_tmp).count(dt_tmp[0])
            ) == len(dt_tmp):
                print(
                    f"\t\t\tSolving head series error: TERMINAL. dt_master not compatible with dt in the {top_boundary} "
                    "input series. dt_master must be an integer multiple of dt in the time series. EXITING."
                )
                sys.exit(1)
            if dt_master[layer] >= dt_tmp[0]:
                spacing_bot = int(dt_master[layer] / dt_tmp[0])
                bot_head_tmp = head_data[bot_boundary]
            else:
                print(
                    "\t\t\tNOTE: dt_master < dt_data. Linear resampling of input head series occuring."
                )
                t_interp_new = 0.0001 * np.arange(
                    10000 * min(t_bot), 10000 * max(t_bot) + 1, 10000 * dt_master[layer]
                )
                h_interp_new = np.interp(t_interp_new, t_top, head_data[bot_boundary].iloc[:, 1].to_numpy())
                bot_head_tmp = np.array([t_interp_new, h_interp_new]).T
                spacing_bot = 1

            if not all(t_top == t_bot):
                print(
                    f"\t\t\tSolving head series error: TERMINAL. Time series in {top_boundary} and {bot_boundary} aquifers have different dates."
                )
                print(t_top)
                print(t_bot)
                sys.exit(1)
            else:
                print("\t\t\tTime series found with correct dt and over same timespan.")

            z = np.arange(
                0, layer_thicknesses[layer] + 0.00001, dz_clays[layer]
            )  # 0.000001 to include the stop value.
            Z[layer] = z

            t_in = 0.0001 * np.arange(
                10000 * np.min(t_top),
                10000 * np.max(t_top) + 1,
                10000 * dt_master[layer],
            )  # 0.000001 to include the stop value. note that t_top and t_bot were clipped to the same yearrange when importing head data.
            print(t_in)
            t_gwflow[layer] = t_in

            if overburden_stress_gwflow:
                if layer != unconfined_aquifer_name:
                    print("\t\tPreparing overburden stress.")
                    if len(overburden_dates) != len(t_in):
                        print("\t\t\tInterpolating overburden stress.")
                        overburden_data_tmp = np.interp(t_in, overburden_dates, overburden_data)
                        overburden_dates_tmp = t_in
                    else:
                        overburden_data_tmp = overburden_data
                else:
                    overburden_data_tmp = np.zeros_like(t_in)
            else:
                overburden_data_tmp = [0]

            if preconsolidation_head_type == "initial_plus_offset":
                initial_precons = False
                print(
                    "\t\tHead initial condition is initial_plus_offset, so a constant head initial condition "
                    "will be applied. Constant value is mean of surrounding aquifers."
                )

                initial_condition_precons[layer] = np.array([])
                initial_condition_tmp = (
                    (top_head_tmp[0, 1] + bot_head_tmp[0, 1]) / 2
                ) * np.ones_like(z) + (
                    preconsolidation_head_offset[top_boundary]
                    + preconsolidation_head_offset[bot_boundary]
                ) / 2
                print(f"\t\t\tinitial head value is {initial_condition_tmp[0]:.2f}")

            else:
                initial_precons = False
                initial_condition_tmp = (top_head_tmp[0, 1] + top_head_tmp[0, 1]) / 2

            if MODE == "resume":
                initial_precons = True
                print(
                    f"\t\tMode is resume. Looking for initial condition in directory {resume_directory}/head_outputs."
                )
                nc_file_path = f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_head_data.nc"
                csv_file_path = f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_head_data.csv"

                if os.path.isfile(nc_file_path):
                    print("Head found as .nc file. Reading.")
                    Dat = Dataset(nc_file_path, "r", format="CF-1.7")
                    time_bc_tmp = Dat.variables["time"][:]
                    head_bc_tmp = Dat.variables["z"][:]
                    idx_bc_tmp = np.argmin(np.abs(mdates.date2num(resume_date) - time_bc_tmp))
                    if np.min(np.abs(mdates.date2num(resume_date) - time_bc_tmp)) >= 1:
                        print(
                            f"\tNote that you are resuming with the initial head condition from {time_bc_tmp[idx_bc_tmp]}, but the specified resume date was {resume_date}."
                        )
                elif os.path.isfile(csv_file_path):
                    print("Head found as .csv file. Reading.")
                    head_bc_tmp = np.genfromtxt(csv_file_path, delimiter=",")
                    time_bc_tmp1 = np.char.rstrip(
                        np.genfromtxt(
                            f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_groundwater_solution_dates.csv",
                            dtype=str,
                            delimiter=",",
                        )
                    )
                    if time_bc_tmp1[0][-1] == "M":
                        time_bc_tmp = mdates.date2num(
                            [
                                datetime.datetime.strptime(string, "%a %d %b %Y %I:%M:%S %p") for string in time_bc_tmp1
                            ]
                        )
                    else:
                        time_bc_tmp = mdates.date2num(
                            [datetime.datetime.strptime(string, "%c") for string in time_bc_tmp1]
                        )
                    idx_bc_tmp = np.argmin(np.abs(mdates.date2num(resume_date) - time_bc_tmp))
                else:
                    print("\tUnable to find head file as .nc or .csv. Something has gone wrong; aborting.")
                    sys.exit(1)

                print(
                    f"\t\tNow looking for effective stress initial condition in directory {resume_directory}."
                )
                if os.path.isfile(
                    f"{resume_directory}/{layer.replace(' ', '_')}effective_stress.nc"
                ):
                    print("t_eff found as .nc file. Reading.")
                    Dat = Dataset(
                        f"{resume_directory}/{layer.replace(' ', '_')}effective_stress.nc",
                        "r",
                        format="CF-1.7",
                    )
                    time_teff_tmp = Dat.variables["time"][:]
                    teff_bc_tmp = Dat.variables["z"][:] / (
                        rho_w * g
                    )  # Get it into units of head
                    idx_teff_bc_tmp = np.argmin(
                        np.abs(mdates.date2num(resume_date) - time_teff_tmp)
                    )
                    if (np.min(np.abs(mdates.date2num(resume_date) - time_teff_tmp)) >= 1):
                        print(
                            "\tNote that you are resuming with the initial t_eff condition from "
                            f"{time_bc_tmp[idx_bc_tmp]}, but the specified resume date was {resume_date}."
                        )
                elif os.path.isfile(
                    f"{resume_directory}/{layer.replace(' ', '_')}clayeffective_stress.csv"
                ):
                    print("t_eff found as .csv file. Reading.")
                    teff_bc_tmp = np.genfromtxt(
                        f"{resume_directory}/{layer.replace(' ', '_')}effective_stress.csv",
                        delimiter=",",
                    )
                    teff_bc_tmp = teff_bc_tmp / (rho_w * g)
                    time_teff_tmp1 = np.char.rstrip(
                        np.genfromtxt(
                            f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_groundwater_solution_dates.csv",
                            dtype=str,
                            delimiter=",",
                        )
                    )
                    if (
                        time_bc_tmp1[0][-1] == "M"
                    ):  # This means it was done on Bletchley in xterm, so %c won't work as it whacks AM or PM on the end.
                        time_teff_tmp = mdates.date2num(
                            [
                                datetime.datetime.strptime(
                                    string, "%a %d %b %Y %I:%M:%S %p"
                                )
                                for string in time_teff_tmp1
                            ]
                        )
                    else:
                        time_teff_tmp = mdates.date2num(
                            [
                                datetime.datetime.strptime(string, "%c")
                                for string in time_teff_tmp1
                            ]
                        )
                    idx_teff_bc_tmp = np.argmin(
                        np.abs(mdates.date2num(resume_date) - teff_bc_tmp)
                    )

                    print("T_eff resume not yet coded for a csv file - abort.")
                    sys.exit(1)
                else:
                    print(
                        "\tUnable to find t_eff file as .nc or .csv. Something has gone wrong; aborting."
                    )
                    sys.exit(1)

                initial_condition_tmp = head_bc_tmp[:, idx_bc_tmp][
                    ::-1
                ]  # The [::-1] is because aquitard head is saved upside down
                initial_condition_precons[layer] = np.max(
                    teff_bc_tmp[:, : idx_teff_bc_tmp + 1], axis=1
                )[::-1]

            if groundwater_flow_solver_type[layer] == "singlevalue":
                hmat_tmp = solve_head_equation_singlevalue(
                    dt_master[layer],
                    t_in,
                    dz_clays[layer],
                    z,
                    np.vstack(
                        (top_head_tmp[::spacing_top, 1], top_head_tmp[::spacing_bot, 1])
                    ),
                    initial_condition_tmp,
                    vertical_conductivity[layer]
                    / (clay_Ssk[layer] + compressibility_of_water),
                )
            elif groundwater_flow_solver_type[layer] == "elastic-inelastic":
                t1_start = time.time()
                # solve the elasticinelastic head equation
                hmat_tmp, inelastic_flag_tmp = solve_head_equation_elasticinelastic(
                    dt_master[layer],
                    t_in,
                    dz_clays[layer],
                    z,
                    np.vstack(
                        (top_head_tmp[::spacing_top, 1], bot_head_tmp[::spacing_bot, 1])
                    ),
                    initial_condition_tmp,
                    vertical_conductivity[layer] / clay_Sse[layer],
                    vertical_conductivity[layer] / clay_Ssv[layer],
                    overburdenstress=overburden_stress_gwflow,
                    overburden_data=1 / (rho_w * g) * np.array(overburden_data_tmp),
                    initial_precons=initial_precons,
                    initial_condition_precons=-initial_condition_precons[layer],
                )
                t1_stop = time.time()
                print("\t\t\tElapsed time in seconds:", t1_stop - t1_start)

            head_series[layer] = hmat_tmp
            inelastic_flag[layer] = inelastic_flag_tmp
            effective_stress[layer] = (
                np.tile(overburden_data_tmp, (np.shape(hmat_tmp)[0], 1))
                - rho_w * g * hmat_tmp
            )

            if save_output_head_timeseries:
                if np.size(inelastic_flag_tmp) >= 3e6:
                    if gmt:
                        print(
                            "\t\t\tInelastic flag gwflow has more than 3 million entries; saving as signed char."
                        )
                        inelastic_flag_tmp.astype(np.byte).tofile(
                            f"{outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW"
                        )
                        print("\t\t\t\tConverting to netCDF format. Command is:")
                        cmd_tmp = (
                            f"gmt xyz2grd {outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW "
                            f"-G{outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW.nb "
                            f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer])[0]:.5f} "
                            f"-R{np.min(t_gwflow[layer]):.3f}/{np.max(t_gwflow[layer]):.3f}/"
                            f"{np.min(Z[layer]):.3f}/{np.max(Z[layer]):.3f}"
                            "-ZTLc"
                        )

                        print(cmd_tmp)
                        subprocess.call(cmd_tmp, shell=True)
                        os.remove(
                            f"{outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW"
                        )

                    else:
                        print(
                            "\t\t\tInelastic flag gwflow has more than 3 million entries; saving as signed char."
                        )
                        inelastic_flag_tmp.astype(np.byte).tofile(
                            f"{outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW"
                        )

                else:
                    with open(
                        f"{outdestination}/head_outputs/{layer.replace(' ', '_')}inelastic_flag_GWFLOW.csv",
                        "w+",
                    ) as myCsv:
                        csvWriter = csv.writer(myCsv, delimiter=",")
                        csvWriter.writerows(inelastic_flag_tmp)

            # dateslist = [x.strftime('%d-%b-%Y') for x in num2date(t_in)]
            groundwater_solution_dates[layer] = t_in

            if overburden_stress_gwflow:
                effective_stress[layer] = (
                    np.tile(overburden_data_tmp, (np.shape(hmat_tmp)[0], 1))
                    - rho_w * g * hmat_tmp
                )
            else:
                effective_stress[layer] = np.zeros_like(hmat_tmp) - rho_w * g * hmat_tmp

            if save_effective_stress:
                print("\t\tSaving effective stress and overburden stress outputs.")
                if overburden_stress_gwflow:
                    if len(overburden_data_tmp) * len(z_tmp) >= 1e6:
                        print(
                            "\t\t\tOverburden stress has more than 1 million entries; saving as 32 bit floats."
                        )
                        overburden_tmp_tosave = np.tile(
                            overburden_data_tmp, (len(z_tmp), 1)
                        )
                        overburden_tmp_tosave.astype(np.single).tofile(
                            f"{outdestination}/{layer.replace(' ', '_')}_overburden_stress"
                        )
                        if gmt:
                            print("\t\t\t\tConverting to netCDF format. Command is:")
                            cmd_tmp = (
                                f"gmt xyz2grd {outdestination}/{layer.replace(' ', '_')}_overburden_stress "
                                f"-G{outdestination}/{layer.replace(' ', '_')}_overburden_stress.nc "
                                f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer])[0]:.5f} "
                                f"-R{np.min(t_gwflow[layer]):.3f}/{np.max(t_gwflow[layer]):.3f}/"
                                f"{np.min(Z[layer]):.3f}/{np.max(Z[layer]):.3f}"
                                "-ZTLf"
                            )

                            print(cmd_tmp)
                            subprocess.call(cmd_tmp, shell=True)
                            os.remove(
                                f"{outdestination}/{layer.replace(' ', '_')}_overburden_stress"
                            )
                    else:
                        with open(
                            f"{outdestination}/{layer.replace(' ', '_')}_overburden_stress.csv",
                            "w+",
                        ) as myCsv:
                            csvWriter = csv.writer(myCsv, delimiter=",")
                            csvWriter.writerows(
                                np.tile(overburden_data_tmp, (len(z_tmp), 1))
                            )
                else:
                    with open(
                        f"{outdestination}/{layer.replace(' ', '_')}_overburden_stress.csv",
                        "w+",
                    ) as myCsv:
                        csvWriter = csv.writer(myCsv, delimiter=",")
                        csvWriter.writerows(np.zeros_like(hmat_tmp))

                if np.size(effective_stress[layer]) >= 1e6:
                    print(
                        "\t\t\tEffective stress has more than 1 million entries; saving as 32 bit floats."
                    )
                    effective_stress[layer].astype(np.single).tofile(
                        f"{outdestination}/{layer.replace(' ', '_')}effective_stress"
                    )
                    if gmt:
                        print("\t\t\t\tConverting to netCDF format. Command is:")
                        cmd_tmp = (
                            f"gmt xyz2grd {outdestination}/{layer.replace(' ', '_')}effective_stress "
                            f"-G{outdestination}/{layer.replace(' ', '_')}effective_stress.nc "
                            f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer])[0]:.5f} "
                            f"-R{np.min(t_gwflow[layer]):.3f}/{np.max(t_gwflow[layer]):.3f}/"
                            f"{np.min(Z[layer]):.2f}/{np.max(Z[layer]):.2f}"
                            "-ZTLf"
                        )
                        print(cmd_tmp)
                        subprocess.call(cmd_tmp, shell=True)
                        os.remove(
                            f"{outdestination}/{layer.replace(' ', '_')}effective_stress"
                        )
                else:
                    with open(
                        f"{outdestination}/{layer.replace(' ', '_')}effective_stress.csv",
                        "w+",
                    ) as myCsv:
                        csvWriter = csv.writer(myCsv, delimiter=",")
                        csvWriter.writerows(effective_stress[layer])

        elif layer_types[layer] == "Aquifer":
            head_series[layer] = dict([("Interconnected matrix", head_data[layer])])
            inelastic_flag[layer] = {}
            groundwater_solution_dates[layer] = {}
            Z[layer] = {}
            t_gwflow[layer] = {}
            effective_stress[layer] = {}
            initial_condition_precons[layer] = {}
            print(f"\t\t{layer} is an aquifer.")
            if interbeds_switch[layer]:
                interbeds_tmp = interbeds_distributions[layer]
                bed_thicknesses_tmp = list(interbeds_tmp.keys())
                print(
                    f"\t\t{layer} is an aquifer with interbedded clays. "
                    f"Thicknesses of clays to be solved are {bed_thicknesses_tmp}"
                )
                for thickness in bed_thicknesses_tmp:
                    print("")
                    print(f"\t\tSolving for thickness {thickness:.2f}.")

                    t_aquifer_tmp = mdates.date2num(head_data[layer].iloc[:, 0].to_numpy())
                    h_aquifer_tmp = head_data[layer].iloc[:, 1].to_numpy()
                    z_tmp = np.arange(
                        0, thickness + 0.00001, dz_clays[layer]
                    )  # 0.000001 to include the stop value.
                    t_interp_new = 0.0001 * np.arange(
                        10000 * min(t_aquifer_tmp),
                        10000 * max(t_aquifer_tmp) + 1,
                        10000 * dt_master[layer],
                    )  # The ridiculous factor of 1/10000 is to ensure the np.arange function takes integer steps. Else, floating point precision issues mean that things go wrong for timesteps less than <0.1 seconds.
                    h_aquifer_tmp_interpolated = np.array([
                        t_interp_new, 
                        np.interp(t_interp_new, t_aquifer_tmp, h_aquifer_tmp)
                    ]).T


                    if preconsolidation_head_type == "initial_plus_offset":
                        print(
                            "\t\tHead initial condition is initial_plus_offset, so a constant head "
                            "initial condition will be applied."
                        )
                        initial_precons = False
                        initial_condition_tmp = (
                            h_aquifer_tmp[0] * np.ones_like(z_tmp)
                            + preconsolidation_head_offset[layer]
                        )
                        initial_condition_precons[layer][
                            f"{thickness:.2f} clays"
                        ] = np.array([])
                        print(
                            f"\t\tinitial head value is {initial_condition_tmp[0]:.2f}"
                        )

                    else:
                        initial_precons = False
                        initial_condition_precons[layer][
                            f"{thickness:.2f} clays"
                        ] = np.array([])
                        initial_condition_tmp = h_aquifer_tmp[0]

                    if MODE == "resume":
                        initial_precons = True
                        print(f"\t\tMode is resume. Looking for initial condition in directory {resume_directory}/head_outputs.")
                        
                        if os.path.isfile(f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clay_head_data.nc"):
                            print("Head found as .nc file. Reading.")
                            Dat = Dataset(
                                f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clay_head_data.nc",
                                "r",
                                format="CF-1.7",
                            )
                            time_bc_tmp = Dat.variables["time"][:]
                            head_bc_tmp = Dat.variables["z"][:]
                            idx_bc_tmp = np.argmin(np.abs(mdates.date2num(resume_date) - time_bc_tmp))
                            if (np.min(np.abs(mdates.date2num(resume_date) - time_bc_tmp)) >= 1):
                                print(
                                    f"\tNote that you are are resuming with the initial head condition from {time_bc_tmp[idx_bc_tmp]}"
                                    f", but the specified resume date was {resume_date}."
                                )
                        
                        elif os.path.isfile(f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clay_head_data.csv"):
                            print("Head found as .csv file. Reading.")
                            head_bc_tmp = np.genfromtxt(
                                f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clay_head_data.csv",
                                delimiter=",",
                            )
                            time_bc_tmp1 = np.char.rstrip(
                                np.genfromtxt(
                                    f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_groundwater_solution_dates.csv",
                                    dtype=str,
                                    delimiter=",",
                                )
                            )
                            if (time_bc_tmp1[0][-1] == "M"):  # This means it was done on Bletchley in xterm, so %c won't work as it whacks AM or PM on the end.
                                time_bc_tmp = mdates.date2num(
                                    [
                                        datetime.datetime.strptime(
                                            string, "%a %d %b %Y %I:%M:%S %p"
                                        )
                                        for string in time_bc_tmp1
                                    ]
                                )
                            else:
                                time_bc_tmp = mdates.date2num(
                                    [
                                        datetime.datetime.strptime(string, "%c")
                                        for string in time_bc_tmp1
                                    ]
                                )
                            idx_bc_tmp = np.argmin(np.abs(mdates.date2num(resume_date) - time_bc_tmp))
                        
                        else:
                            print("\tUnable to find head file as .nc or .csv. Something has gone wrong; aborting.")
                            sys.exit(1)

                        print(f"\t\tNow looking for effective stress initial condition in directory {resume_directory}.")
                        
                        nc_file_path = f"{resume_directory}/{layer.replace(' ', '_')}_{thickness:.2f}clayeffective_stress.nc"
                        csv_file_path = f"{resume_directory}/{layer.replace(' ', '_')}_{thickness:.1f}clayeffective_stress.csv"
                        
                        if os.path.isfile(nc_file_path):
                            print("t_eff found as .nc file. Reading.")
                            Dat = Dataset(nc_file_path, "r", format="CF-1.7")
                            time_teff_tmp = Dat.variables["time"][:]
                            teff_bc_tmp = Dat.variables["z"][:] / (rho_w * g)  # Get it into units of head
                        elif os.path.isfile(csv_file_path):
                            print("t_eff found as .csv file. Reading.")
                            teff_bc_tmp = np.genfromtxt(csv_file_path, delimiter=",") / (rho_w * g)
                            time_teff_tmp1 = np.char.rstrip(
                                np.genfromtxt(
                                    f"{resume_directory}/head_outputs/{layer.replace(' ', '_')}_groundwater_solution_dates.csv",
                                    dtype=str,
                                    delimiter=",",
                                )
                            )
                            if time_teff_tmp1[0][-1] == "M":
                                time_teff_tmp = mdates.date2num(
                                    [datetime.datetime.strptime(string, "%a %d %b %Y %I:%M:%S %p") for string in time_teff_tmp1]
                                )
                            else:
                                time_teff_tmp = mdates.date2num(
                                    [datetime.datetime.strptime(string, "%c") for string in time_teff_tmp1]
                                )
                        else:
                            print("\tUnable to find t_eff file as .nc or .csv. Something has gone wrong; aborting.")
                            sys.exit(1)
                        
                        idx_teff_bc_tmp = np.argmin(np.abs(mdates.date2num(resume_date) - time_teff_tmp))
                        if np.min(np.abs(mdates.date2num(resume_date) - time_teff_tmp)) >= 1:
                            print(
                                f"\tNote that you are resuming with the initial t_eff condition from {time_bc_tmp[idx_bc_tmp]}, but the specified resume date was {resume_date}."
                            )


                        initial_condition_tmp = head_bc_tmp[:, idx_bc_tmp]
                        initial_condition_precons[layer][
                            f"{thickness:.2f} clays"
                        ] = np.max(teff_bc_tmp[:, : idx_teff_bc_tmp + 1], axis=1)

                    if overburden_stress_gwflow:
                        if layer != unconfined_aquifer_name:
                            print("\t\t\tInterpolating overburden stress.")
                        else:
                            print("\t\t\tThis is the unconfined aquifer; overburden still being included.")
                            
                        if len(overburden_dates) != len(t_interp_new):
                            overburden_data_tmp = np.interp(
                                t_interp_new,
                                mdates.date2num(overburden_dates.to_numpy()),
                                overburden_data.iloc[:, 1].to_numpy()
                            )

                            overburden_dates_tmp = t_interp_new
                        else:
                            overburden_data_tmp = overburden_data
                    else:
                        overburden_data_tmp = [0]

                    t1_start = time.time()
                    (
                        hmat_tmp,
                        inelastic_flag_tmp,
                    ) = solve_head_equation_elasticinelastic(
                        dt_master[layer],
                        t_interp_new,
                        dz_clays[layer],
                        z_tmp,
                        np.vstack(
                            (
                                h_aquifer_tmp_interpolated[:, 1],
                                h_aquifer_tmp_interpolated[:, 1],
                            )
                        ),
                        initial_condition_tmp,
                        vertical_conductivity[layer] / clay_Sse[layer],
                        vertical_conductivity[layer] / clay_Ssv[layer],
                        overburdenstress=overburden_stress_gwflow,
                        overburden_data=1
                        / (rho_w * g)
                        * np.array(overburden_data_tmp),
                        initial_precons=initial_precons,
                        initial_condition_precons=-initial_condition_precons[layer][
                            f"{thickness:.2f} clays"
                        ],
                    )
                    t1_stop = time.time()
                    print("\t\t\tElapsed time in seconds:", t1_stop - t1_start)
                    head_series[layer][f"{thickness:.2f} clays"] = hmat_tmp
                    inelastic_flag[layer][
                        f"{thickness:.2f} clays"
                    ] = inelastic_flag_tmp
                    t_gwflow[layer][f"{thickness:.2f} clays"] = t_interp_new
                    Z[layer][f"{thickness:.2f} clays"] = z_tmp

                    if save_output_head_timeseries:
                        if np.size(inelastic_flag_tmp) >= 3e6:
                            if gmt:
                                print(
                                    "\t\t\tInelastic flag gwflow has more than 3 million entries; saving as signed char."
                                )
                                inelastic_flag_tmp.astype(np.byte).tofile(
                                    f"{outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW"
                                )
                                print(
                                    "\t\t\t\tConverting to netCDF format. Command is:"
                                )
                                cmd_tmp = (
                                    f"gmt xyz2grd {outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW "
                                    f"-G{outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW.nb "
                                    f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer][f'{thickness:.2f} clays'])[0]:.5f} "
                                    f"-R{np.min(t_gwflow[layer][f'{thickness:.2f} clays']):.3f}/{np.max(t_gwflow[layer][f'{thickness:.2f} clays']):.3f}/"
                                    f"{np.min(Z[layer][f'{thickness:.2f} clays']):.3f}/{np.max(Z[layer][f'{thickness:.2f} clays']):.3f} "
                                    "-ZTLc"
                                )

                                print(cmd_tmp)
                                subprocess.call(cmd_tmp, shell=True)
                                os.remove(
                                    f"{outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW"
                                )

                            else:
                                print(
                                    "\t\t\tInelastic flag gwflow has more than 3 million entries; saving as signed char."
                                )
                                inelastic_flag_tmp.astype(np.byte).tofile(
                                    f"{outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW"
                                )

                        else:
                            with open(
                                f"{outdestination}/head_outputs/{layer.replace(' ', '_')}_{thickness:.2f}clayinelastic_flag_GWFLOW.csv",
                                "w+",
                            ) as myCsv:
                                csvWriter = csv.writer(myCsv, delimiter=",")
                                csvWriter.writerows(inelastic_flag_tmp)

                    groundwater_solution_dates[layer][
                        f"{thickness:.2f} clays"
                    ] = t_interp_new

                    if overburden_stress_gwflow:
                        effective_stress[layer][f"{thickness:.2f} clays"] = (
                            np.tile(overburden_data_tmp, (np.shape(hmat_tmp)[0], 1))
                            - rho_w * g * hmat_tmp
                        )
                    else:
                        effective_stress[layer][f"{thickness:.2f} clays"] = (
                            np.zeros_like(hmat_tmp) - rho_w * g * hmat_tmp
                        )

                    if save_effective_stress:
                        print(
                            "\t\tSaving effective stress and overburden stress outputs."
                        )
                        if (np.size(effective_stress[layer][f"{thickness:.2f} clays"]) >= 1e6):
                            print(
                                "\t\t\tEffective stress has more than 1 million entries; saving as 32 bit floats."
                            )
                            effective_stress[layer][f"{thickness:.2f} clays"].astype(np.single).tofile(
                                f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clayeffective_stress"
                            )
                            if gmt:
                                print(
                                    "\t\t\t\tConverting to netCDF format. Command is:"
                                )
                                cmd_tmp = (
                                    f"gmt xyz2grd {outdestination}/{layer.replace(' ', '_')}_{thickness}clayeffective_stress "
                                    f"-G{outdestination}/{layer.replace(' ', '_')}_{thickness}clayeffective_stress.nc "
                                    f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer][f'{thickness} clays'])[0]:.5f} "
                                    f"-R{np.min(t_gwflow[layer][f'{thickness} clays']):.3f}/{np.max(t_gwflow[layer][f'{thickness} clays']):.3f}/"
                                    f"{np.min(Z[layer][f'{thickness} clays']):.2f}/{np.max(Z[layer][f'{thickness} clays']):.2f} "
                                    "-ZTLf"
                                )

                                print(cmd_tmp)
                                subprocess.call(cmd_tmp, shell=True)
                                os.remove(f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clayeffective_stress")
                                
                        else:
                            with open(
                                f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clayeffective_stress.csv",
                                "w+",
                            ) as myCsv:
                                csvWriter = csv.writer(myCsv, delimiter=",")
                                csvWriter.writerows(
                                    effective_stress[layer][
                                        f"{thickness:.2f} clays"
                                    ]
                                )

                        if overburden_stress_gwflow:
                            if len(overburden_data_tmp) * len(z_tmp) >= 1e6:
                                print(
                                    "\t\t\tOverburden stress has more than 1 million entries; saving as 32 bit floats."
                                )
                                overburden_tmp_tosave = np.tile(
                                    overburden_data_tmp, (len(z_tmp), 1)
                                )
                                overburden_tmp_tosave.astype(np.single).tofile(
                                    f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clay_overburden_stress"
                                )
                                if gmt:
                                    print(
                                        "\t\t\t\tConverting to netCDF format. Command is:"
                                    )
                                    cmd_tmp = (
                                        f"gmt xyz2grd {outdestination}/{layer.replace(' ', '_')}_{thickness}clay_overburden_stress "
                                        f"-G{outdestination}/{layer.replace(' ', '_')}_{thickness}clay_overburden_stress.nc "
                                        f"-I{dt_master[layer]:.3f}/{np.diff(Z[layer][f'{thickness:.2f} clays'])[0]:.5f} "
                                        f"-R{np.min(t_gwflow[layer][f'{thickness:.2f} clays']):.3f}/{np.max(t_gwflow[layer][f'{thickness:.2f} clays']):.3f}/"
                                        f"{np.min(Z[layer][f'{thickness:.2f} clays']):.3f}/{np.max(Z[layer][f'{thickness:.2f} clays']):.3f} "
                                        "-ZTLf"
                                    )

                                    print(cmd_tmp)
                                    subprocess.call(cmd_tmp, shell=True)
                                    os.remove(f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clay_overburden_stress")
                            
                            else:
                                with open(f"{outdestination}/{layer.replace(' ', '_')}_{thickness}clay_overburden_stress.csv", "w+") as myCsv:
                                    csvWriter = csv.writer(myCsv, delimiter=",")
                                    csvWriter.writerows(
                                        np.tile(
                                            overburden_data_tmp, (len(z_tmp), 1)
                                        )
                                    )

            else:
                print(
                    f"\t{aquifer} is an aquifer with no interbedded clays. No solution required."
                )

else:
    print(
        "No layers require head time series solutions; skipping solving for head time series in clay layers."
    )

solving_head_stop = time.time()
solving_head_time = solving_head_stop - solving_head_start

# %% New section, saving head outputs.
# print()
# print()
# print("".center(80, "*"))
# print("  SAVING HEAD TIMESERIES OUTPUTS  ".center(80, "*"))
# print("".center(80, "*"))
# print()
# saving_head_start = time.time()
# time.sleep(internal_time_delay)
#
#
# if save_output_head_timeseries:
#    print("save_output_head_timeseries = True. Saving head timeseries for all layers.")
#    for layer in layers_requiring_solving:
#        print("\tSaving head timeseries for %s." % layer)
#        if layer_types[layer] == "Aquifer":
#            dates_str = [
#                x.strftime("%d-%b-%Y")
#                for x in mdates.num2date(head_data[layer][:, 0])
#            ]
#            np.savetxt(
#                "%s/head_outputs/%s_head_data.csv"
#                % (outdestination, layer.replace(" ", "_")),
#                np.column_stack((dates_str, head_data[layer][:, 1])),
#                fmt="%s",
#            )
#            if interbeds_switch[layer]:
#                interbeds_tmp = interbeds_distributions[layer]
#                for thickness in list(interbeds_tmp.keys()):
#                    if np.size(head_series[layer][f"{thickness:.2f} clays"]) >= 1e6:
#                        if gmt:
#                            print(
#                                "\t\t\tHead has more than 1 million entries; saving as 32 bit floats."
#                            )
#                            head_series[layer][f"{thickness:.2f} clays"].astype(
#                                np.single
#                            ).tofile(
#                                "%s/head_outputs/%s_%sclay_head_data"
#                                % (outdestination, layer.replace(" ", "_"), thickness)
#                            )
#                            print("\t\t\t\tConverting to netCDF format. Command is:")
#                            cmd_tmp = (
#                                "gmt xyz2grd %s/head_outputs/%s_%sclay_head_data -G%s/head_outputs/%s_%sclay_head_data.nc -I%.3f/%.5f -R%.3ft/%.3ft/%.3f/%.3f -ZTLf"
#                                % (
#                                    outdestination,
#                                    layer.replace(" ", "_"),
#                                    thickness,
#                                    outdestination,
#                                    layer.replace(" ", "_"),
#                                    f"{thickness:.2f}",
#                                    dt_master[layer],
#                                    np.diff(Z[layer][f"{thickness:.2f} clays"])[0],
#                                    np.min(t_gwflow[layer][f"{thickness:.2f} clays"]),
#                                    np.max(t_gwflow[layer][f"{thickness:.2f} clays"]),
#                                    np.min(Z[layer][f"{thickness:.2f} clays"]),
#                                    np.max(Z[layer][f"{thickness:.2f} clays"]),
#                                )
#                            )
#
#                            print(cmd_tmp)
#                            subprocess.call(cmd_tmp, shell=True)
#                            os.remove(
#                                "%s/head_outputs/%s_%sclay_head_data"
#                                % (outdestination, layer.replace(" ", "_"), thickness)
#                            )
#                        else:
#                            print(
#                                "\t\t\tHead has more than 1 million entries; saving as 16 bit floats."
#                            )
#                            head_series[layer][f"{thickness:.2f} clays"].astype(
#                                np.half
#                            ).tofile(
#                                "%s/head_outputs/%s_%sclay_head_data"
#                                % (outdestination, layer.replace(" ", "_"), thickness)
#                            )
#
#                    else:
#                        with open(
#                            "%s/head_outputs/%s_%sclay_head_data.csv"
#                            % (
#                                outdestination,
#                                layer.replace(" ", "_"),
#                                f"{thickness:.2f}",
#                            ),
#                            "w+",
#                        ) as myCsv:
#                            csvWriter = csv.writer(myCsv, delimiter=",")
#                            csvWriter.writerows(
#                                head_series[layer][f"{thickness:.2f} clays"]
#                            )
#            with open(
#                "%s/head_outputs/%s_groundwater_solution_dates.csv"
#                % (outdestination, layer.replace(" ", "_")),
#                "w",
#            ) as myfile:
#                wr = csv.writer(myfile)
#                res = list(groundwater_solution_dates[layer].keys())[0]
#                wr.writerow(
#                    [
#                        x.strftime("%c")
#                        for x in mdates.num2date(
#                            groundwater_solution_dates[layer][res]
#                        )
#                    ]
#                )
#
#        if layer_types[layer] == "Aquitard":
#            if np.size(head_series[layer]) >= 1e6:
#                if gmt:
#                    print(
#                        "\t\t\tHead has more than 1 million entries; saving as 32 bit floats."
#                    )
#                    head_series[layer].astype(np.single).tofile(
#                        "%s/head_outputs/%s_head_data"
#                        % (outdestination, layer.replace(" ", "_"))
#                    )
#                    print("\t\t\t\tConverting to netCDF format. Command is:")
#                    cmd_tmp = (
#                        "gmt xyz2grd %s/head_outputs/%s_head_data -G%s/head_outputs/%s_head_data.nc -I%.3f/%.5f -R%.3ft/%.3ft/%.3f/%.3f -ZTLf"
#                        % (
#                            outdestination,
#                            layer.replace(" ", "_"),
#                            outdestination,
#                            layer.replace(" ", "_"),
#                            dt_master[layer],
#                            np.diff(Z[layer])[0],
#                            np.min(t_gwflow[layer]),
#                            np.max(t_gwflow[layer]),
#                            np.min(Z[layer]),
#                            np.max(Z[layer]),
#                        )
#                    )
#
#                    print(cmd_tmp)
#                    subprocess.call(cmd_tmp, shell=True)
#                    os.remove(
#                        "%s/head_outputs/%s_head_data"
#                        % (outdestination, layer.replace(" ", "_"))
#                    )
#                else:
#                    print(
#                        "\t\t\tHead has more than 1 million entries; saving as 16 bit floats."
#                    )
#                    head_series[layer].astype(np.half).tofile(
#                        "%s/head_outputs/%s_head_data"
#                        % (outdestination, layer.replace(" ", "_"))
#                    )
#
#            else:
#                with open(
#                    "%s/head_outputs/%s_head_data.csv"
#                    % (outdestination, layer.replace(" ", "_")),
#                    "w+",
#                ) as myCsv:
#                    csvWriter = csv.writer(myCsv, delimiter=",")
#                    csvWriter.writerows(head_series[layer])
#            with open(
#                "%s/head_outputs/%s_groundwater_solution_dates.csv"
#                % (outdestination, layer.replace(" ", "_")),
#                "w",
#            ) as myfile:
#                wr = csv.writer(myfile)
#                wr.writerow(
#                    [
#                        x.strftime("%c")
#                        for x in mdates.num2date(
#                            groundwater_solution_dates[layer]
#                        )
#                    ]
#                )
#
# for layer in layers_requiring_solving:
#    if create_output_head_video[layer]:
#        print(
#            "create_output_head_video = True. Creating head timeseries video for specified layers. Note: requires ffmpeg installed."
#        )
#        if layer_types[layer] == "Aquitard":
#            print("\tCreating video for %s." % layer)
#            hmat_tmp = head_series[layer]
#            inelastic_flag_vid = inelastic_flag[layer]
#            inelastic_flag_vid = inelastic_flag_vid == 1
#            dates_str = [
#                x.strftime("%d-%b-%Y")
#                for x in mdates.num2date(groundwater_solution_dates[layer])
#            ]
#            create_head_video_elasticinelastic(
#                hmat_tmp,
#                Z[layer],
#                inelastic_flag_vid,
#                dates_str,
#                outdestination + "/figures",
#                layer,
#            )
#
#        if layer_types[layer] == "Aquifer":
#            print("\tCreating video for %s." % layer)
#            interbeds_tmp = interbeds_distributions[layer]
#            bed_thicknesses_tmp = list(interbeds_tmp.keys())
#            print(
#                "\t\t%s is an aquifer with interbedded clays. Thicknesses of clays to make videos are %s"
#                % (layer, bed_thicknesses_tmp)
#            )
#            for thickness in bed_thicknesses_tmp:
#                print("\t\tCreating video for %s_%.2f clays" % (layer, thickness))
#                hmat_tmp = head_series[layer][f"{thickness:.2f} clays"]
#                inelastic_flag_vid = inelastic_flag[layer][f"{thickness:.2f} clays"]
#                inelastic_flag_vid = inelastic_flag_vid == 1
#                dates_str = [
#                    x.strftime("%d-%b-%Y")
#                    for x in mdates.num2date(
#                        groundwater_solution_dates[layer][f"{thickness:.2f} clays"]
#                    )
#                ]
#                create_head_video_elasticinelastic(
#                    hmat_tmp,
#                    Z[layer][f"{thickness:.2f} clays"],
#                    inelastic_flag_vid,
#                    dates_str,
#                    outdestination + "/figures",
#                    "%s_%.2f_clays" % (layer, thickness),
#                    delt=30,
#                )
#
# saving_head_stop = time.time()
# saving_head_time = saving_head_stop - saving_head_start
#
## %% New section, compaction solver.
# print()
# print()
# print("".center(80, "*"))
# print("  SOLVING COMPACTION EQUATION  ".center(80, "*"))
# print("".center(80, "*"))
# print()
# solving_compaction_start = time.time()
# time.sleep(internal_time_delay)
#
## deformation_series={}
## deformation_series_elastic={}
## deformation_series_inelastic={}
## deformation_series_sand={}
# deformation = {}
# db = {}
# deformation_OUTPUT = {}
# compacting_layers = [
#    name for name, value in layer_compaction_switch.items() if value == True
# ]
# if MODE == "resume":
#    preset_precons = True
# else:
#    preset_precons = False
#
#
# for layer in layer_names:
#    if layer_types[layer] == "Aquifer":
#        if layer_compaction_switch[layer]:
#            print()
#            print("%s is an Aquifer. Solving for layer compaction." % layer)
#            deformation[layer] = {}
#            db[layer] = {}
#            inelastic_flag_compaction[layer] = {}
#            if layer_thickness_types[layer] == "constant":
#                layer_sand_thickness_tmp = layer_thicknesses[layer] - np.sum(
#                    [
#                        list(interbeds_distributions[layer].keys())[i]
#                        * list(interbeds_distributions[layer].values())[i]
#                        for i in range(len(interbeds_distributions[layer]))
#                    ]
#                )
#                print(
#                    "\tTotal sand thickness in aquifer is %.2f m."
#                    % layer_sand_thickness_tmp
#                )
#            elif layer_thickness_types[layer] == "step_changes":
#                layer_sand_thickness_tmp = initial_thicknesses[layer] - np.sum(
#                    [
#                        list(interbeds_distributions[layer].keys())[i]
#                        * list(interbeds_distributions[layer].values())[i]
#                        for i in range(len(interbeds_distributions[layer]))
#                    ]
#                )
#                print(
#                    "\tInitial total sand thickness in aquifer is %.2f m."
#                    % layer_sand_thickness_tmp
#                )
#            deformation[layer]["Interconnected matrix"] = [
#                layer_sand_thickness_tmp
#                * (sand_Sse[layer] - compressibility_of_water)
#                * (
#                    head_series[layer]["Interconnected matrix"][i, 1]
#                    - head_series[layer]["Interconnected matrix"][0, 1]
#                )
#                for i in range(len(head_series[layer]["Interconnected matrix"][:, 1]))
#            ]
#            interbeds_tmp = interbeds_distributions[layer]
#            bed_thicknesses_tmp = list(interbeds_tmp.keys())
#            print(
#                "\t\t%s is an aquifer with interbedded clays. Thicknesses of clays to solve compaction are %s"
#                % (layer, bed_thicknesses_tmp)
#            )
#            for thickness in bed_thicknesses_tmp:
#                print("\t\t\tSolving for thickness %.2f." % thickness)
#
#                if overburden_stress_compaction:
#                    unconfined_tmp = unconfined_aquifer_name == layer
#                    if len(overburden_dates) != len(
#                        groundwater_solution_dates[layer][f"{thickness:.2f} clays"]
#                    ):
#                        print(
#                            "\t\t\tOverburden series is %i long whereas head series is %i long. Interpolating overburden stress."
#                            % (
#                                len(overburden_dates),
#                                len(
#                                    groundwater_solution_dates[layer][
#                                        f"{thickness:.2f} clays"
#                                    ]
#                                ),
#                            )
#                        )
#                        f_tmp = scipy.interpolate.interp1d(
#                            overburden_dates, overburden_data
#                        )
#                        overburden_data_tmp = f_tmp(
#                            groundwater_solution_dates[layer][f"{thickness:.2f} clays"]
#                        )
#                        overburden_dates_tmp = groundwater_solution_dates[layer][
#                            f"{thickness:.2f} clays"
#                        ]
#                    else:
#                        overburden_data_tmp = overburden_data
#
#                    (
#                        deformation[layer]["total_%.2f clays" % thickness],
#                        inelastic_flag_compaction[layer][
#                            "elastic_%.2f clays" % thickness
#                        ],
#                    ) = subsidence_solver_aquitard_elasticinelastic(
#                        head_series[layer][f"{thickness:.2f} clays"],
#                        (clay_Sse[layer] - compressibility_of_water),
#                        (clay_Ssv[layer] - compressibility_of_water),
#                        dz_clays[layer],
#                        unconfined=unconfined_tmp,
#                        overburden=overburden_stress_compaction,
#                        overburden_data=1 / (rho_w * g) * np.array(overburden_data_tmp),
#                        endnodes=compaction_solver_debug_include_endnodes,
#                        preset_precons=preset_precons,
#                        ic_precons=initial_condition_precons[layer][
#                            f"{thickness:.2f} clays"
#                        ],
#                    )
#                else:
#                    (
#                        deformation[layer]["total_%.2f clays" % thickness],
#                        inelastic_flag_compaction[layer][
#                            "elastic_%.2f clays" % thickness
#                        ],
#                    ) = subsidence_solver_aquitard_elasticinelastic(
#                        head_series[layer][f"{thickness:.2f} clays"],
#                        (clay_Sse[layer] - compressibility_of_water),
#                        (clay_Ssv[layer] - compressibility_of_water),
#                        dz_clays[layer],
#                        endnodes=compaction_solver_debug_include_endnodes,
#                        preset_precons=preset_precons,
#                        ic_precons=initial_condition_precons[layer][
#                            f"{thickness:.2f} clays"
#                        ],
#                    )
#                deformation[layer]["total_%.2f clays" % thickness] = (
#                    interbeds_distributions[layer][thickness]
#                    * deformation[layer]["total_%.2f clays" % thickness]
#                )
#                # deformation[layer]['elastic_%.2f clays' % thickness] = interbeds_distributions[layer][thickness] * deformation[layer]['elastic_%.2f clays' % thickness]
#                # deformation[layer]['inelastic_%.2f clays' % thickness]= interbeds_distributions[layer][thickness] * deformation[layer]['elastic_%.2f clays' % thickness]
#            # Now collect the results at the max timestep
#            dt_sand_tmp = np.diff(head_data[layer][:, 0])[0]
#            print(
#                "\tSumming deformation for layer %s. dts are %.2f and %.2f."
#                % (layer, dt_sand_tmp, dt_master[layer])
#            )
#            dt_max_tmp = np.max([dt_sand_tmp, dt_master[layer]])
#
#            # t_total_tmp = 0.001 * np.arange(1000*np.min(head_data[layer][:,0]),1000*np.max(head_data[layer][:,0]+0.000001),1000*dt_max_tmp) # this is the master dt which will apply for all the sublayers within layer
#            # actual_increment = np.mean(np.diff(head_data[layer][:,0])) # these 2 lines replace the code below. it wasn't calculating the actual increment for some reason.
#            # t_total_tmp = 0.001 * np.arange(1000*np.min(head_data[layer][:,0]), 1000*np.max(head_data[layer][:,0] + 0.000001), 1000*actual_increment)
#            head_dates = head_data[layer][
#                :, 0
#            ]  # this is code I used to account for leap years.
#            t_total_tmp_list = [head_dates[0]]  # starting from the first date
#
#            for i in range(1, len(head_dates)):
#                increment = head_dates[i] - head_dates[i - 1]
#                t_total_tmp_list.append(t_total_tmp_list[-1] + increment)
#
#            t_total_tmp = np.array(
#                t_total_tmp_list
#            )  # * 0.001 # not sure why this .001 was there, but it was messing up the dates. Removed.
#
#            deformation_OUTPUT_tmp = {}
#            deformation_OUTPUT_tmp["dates"] = [
#                x.strftime("%d-%b-%Y") for x in mdates.num2date(t_total_tmp)
#            ]
#
#            deformation_OUTPUT_tmp["Interconnected Matrix"] = np.array(
#                deformation[layer]["Interconnected matrix"]
#            )[np.where(np.isin(head_data[layer][:, 0], t_total_tmp))]
#            def_tot_tmp = np.zeros_like(t_total_tmp, dtype="float")
#
#            # def_tot_tmp += np.array(deformation[layer]['Interconnected matrix'])[np.where(np.isin(head_data[layer][:,0],t_total_tmp))]
#            row_indices = np.where(np.isin(head_data[layer][:, 0], t_total_tmp))[
#                0
#            ]  # these 3 linesare a workaround for the line above, which was throwing an error
#            isin_result = np.isin(head_data[layer][:, 0], t_total_tmp)  # for debugging
#            # print(head_data[layer].shape)
#            # print(head_data[layer][:,0].shape)
#            # print(t_total_tmp.shape)
#            # print(isin_result)
#            # print(np.where(isin_result))
#            indexed_array = np.array(deformation[layer]["Interconnected matrix"])[
#                row_indices
#            ]
#            def_tot_tmp += indexed_array
#
#            for thickness in bed_thicknesses_tmp:
#                def_tot_tmp += np.array(
#                    deformation[layer]["total_%.2f clays" % thickness]
#                )[
#                    np.isin(
#                        0.0001
#                        * np.arange(
#                            10000 * np.min(head_data[layer][:, 0]),
#                            10000 * (np.max(head_data[layer][:, 0]) + 0.0001),
#                            10000 * dt_master[layer],
#                        ),
#                        t_total_tmp,
#                    )
#                ]
#                ting = np.array(deformation[layer]["total_%.2f clays" % thickness])[
#                    np.isin(
#                        0.0001
#                        * np.arange(
#                            10000 * np.min(head_data[layer][:, 0]),
#                            10000 * (np.max(head_data[layer][:, 0]) + 0.0001),
#                            10000 * dt_master[layer],
#                        ),
#                        t_total_tmp,
#                    )
#                ]
#                deformation_OUTPUT_tmp["total_%.2f clays" % thickness] = ting
#
#            deformation[layer]["total"] = np.array([t_total_tmp, def_tot_tmp])
#            deformation_OUTPUT_tmp["total"] = def_tot_tmp
#            deformation_OUTPUT[layer] = pd.DataFrame(deformation_OUTPUT_tmp)
#
#    if layer_types[layer] == "Aquitard":
#        print()
#        print("%s is an Aquitard. Solving for layer compaction." % layer)
#        if layer_compaction_switch[layer]:
#            if compaction_solver_compressibility_type[layer] == "elastic-inelastic":
#                deformation[layer] = {}
#
#                if overburden_stress_compaction:
#                    unconfined_tmp = unconfined_aquifer_name == layer
#                    print("UNCONFINED STATUS = %s" % unconfined_tmp)
#                    if len(overburden_dates) != len(groundwater_solution_dates[layer]):
#                        print(
#                            "\t\t\tOverburden series is %i long whereas head series is %i long. Interpolating overburden stress."
#                            % (
#                                len(overburden_dates),
#                                len(groundwater_solution_dates[layer]),
#                            )
#                        )
#                        f_tmp = scipy.interpolate.interp1d(
#                            overburden_dates, overburden_data
#                        )
#                        overburden_data_tmp = f_tmp(groundwater_solution_dates[layer])
#                        overburden_dates_tmp = groundwater_solution_dates[layer]
#                    else:
#                        overburden_data_tmp = overburden_data
#
#                    (
#                        totdeftmp,
#                        inelastic_flag_compaction[layer],
#                    ) = subsidence_solver_aquitard_elasticinelastic(
#                        head_series[layer],
#                        (clay_Sse[layer] - compressibility_of_water),
#                        (clay_Ssv[layer] - compressibility_of_water),
#                        dz_clays[layer],
#                        unconfined=unconfined_tmp,
#                        overburden=overburden_stress_compaction,
#                        overburden_data=1 / (rho_w * g) * np.array(overburden_data_tmp),
#                        preset_precons=preset_precons,
#                        ic_precons=initial_condition_precons[layer],
#                    )
#                    deformation[layer]["total"] = np.array(
#                        [groundwater_solution_dates[layer], totdeftmp]
#                    )
#
#                else:
#                    (
#                        totdeftmp,
#                        inelastic_flag_compaction[layer],
#                    ) = subsidence_solver_aquitard_elasticinelastic(
#                        head_series[layer],
#                        (clay_Sse[layer] - compressibility_of_water),
#                        (clay_Ssv[layer] - compressibility_of_water),
#                        dz_clays[layer],
#                        preset_precons=preset_precons,
#                        ic_precons=initial_condition_precons[layer],
#                    )
#                    deformation[layer]["total"] = np.array(
#                        [groundwater_solution_dates[layer], totdeftmp]
#                    )
#
# if (
#    MODE == "Normal"
# ):  # If we are resuming, we do not scale layer thicknesses by default.
#    if len(layers_var_thickness) >= 1:
#        print("")
#        print("Scaling layer outputs by temporally varying layer thicknesses.")
#        for layer in layers_var_thickness:
#            print("\tScaling TOTAL outputs for %s." % layer)
#            prekeyname = np.array(list(layer_thicknesses[layer].keys()))[
#                np.where(
#                    ["pre" in key for key in list(layer_thicknesses[layer].keys())]
#                )[0][0]
#            ]
#            datetimedates = mdates.num2date(deformation[layer]["total"][0, :])
#            #        datetimedates = [dt.strptime(d,'%d-%b-%Y') for d in deformation_OUTPUT[layer]['dates'].values]
#            logicaltmp = [
#                datetimedate
#                <= datetime.datetime(
#                    int("%s" % prekeyname.split("-")[1]),
#                    9,
#                    1,
#                    tzinfo=datetime.timezone.utc,
#                )
#                for datetimedate in datetimedates
#            ]
#            scaling_factor_tmp = (
#                layer_thicknesses[layer][prekeyname] / initial_thicknesses[layer]
#            )
#            deformation_scaled_tmp = (
#                deformation[layer]["total"][1, :][logicaltmp] * scaling_factor_tmp
#            )
#
#            nonprekeynames = np.array(list(layer_thicknesses[layer].keys()))[
#                np.where(
#                    ["pre" not in key for key in list(layer_thicknesses[layer].keys())]
#                )[0]
#            ]
#            nonprekeynames.sort()
#            for key in nonprekeynames:
#                if not key.endswith("-"):
#                    scaling_factor_tmp = (
#                        layer_thicknesses[layer][key] / initial_thicknesses[layer]
#                    )
#                    years_tmp = key.split("-")
#                    print("\t\tScaling years", years_tmp, "by ", scaling_factor_tmp)
#                    logicaltmp = [
#                        (
#                            datetimedate
#                            <= datetime.datetime(
#                                int("%s" % years_tmp[1]),
#                                9,
#                                1,
#                                tzinfo=datetime.timezone.utc,
#                            )
#                        )
#                        and (
#                            datetimedate
#                            > datetime.datetime(
#                                int("%s" % years_tmp[0]),
#                                9,
#                                1,
#                                tzinfo=datetime.timezone.utc,
#                            )
#                        )
#                        for datetimedate in datetimedates
#                    ]
#                    deformation_scaled_tmp = np.append(
#                        deformation_scaled_tmp,
#                        (
#                            deformation[layer]["total"][1, :][logicaltmp]
#                            - deformation[layer]["total"][1, :][
#                                np.where(logicaltmp)[0][0] - 1
#                            ]
#                        )
#                        * scaling_factor_tmp
#                        + deformation_scaled_tmp[-1],
#                    )
#            for key in nonprekeynames:
#                if key.endswith("-"):
#                    scaling_factor_tmp = (
#                        layer_thicknesses[layer][key] / initial_thicknesses[layer]
#                    )
#                    years_tmp = key.split("-")
#                    print("\t\tScaling years", years_tmp, "by ", scaling_factor_tmp)
#                    logicaltmp = [
#                        datetimedate
#                        > datetime.datetime(
#                            int("%s" % years_tmp[0]), 9, 1, tzinfo=datetime.timezone.utc
#                        )
#                        for datetimedate in datetimedates
#                    ]
#                    deformation_scaled_tmp = np.append(
#                        deformation_scaled_tmp,
#                        (
#                            deformation[layer]["total"][1, :][logicaltmp]
#                            - deformation[layer]["total"][1, :][
#                                np.where(logicaltmp)[0][0] - 1
#                            ]
#                        )
#                        * scaling_factor_tmp
#                        + deformation_scaled_tmp[-1],
#                    )
#            deformation[layer]["total"][1, :] = deformation_scaled_tmp
#            print("\tScaling SUBOUTPUTS for %s." % layer)
#
#            scaling_factor_tmp = (
#                layer_thicknesses[layer][prekeyname] / initial_thicknesses[layer]
#            )
#            deformation_scaled_tmp = (
#                deformation[layer]["total"][1, :][logicaltmp] * scaling_factor_tmp
#            )
#
# solving_compaction_stop = time.time()
# solving_compaction_time = solving_compaction_stop - solving_compaction_start
#
## %% New section, saving compaction outputs.
# print()
# print()
# print("".center(80, "*"))
# print("  SAVING COMPACTION SOLVER OUTPUTS  ".center(80, "*"))
# print("".center(80, "*"))
# print()
# saving_compaction_start = time.time()
# time.sleep(internal_time_delay)
#
#
# for layer in layer_names:
#    if layer_types[layer] == "Aquitard":
#        if layer_compaction_switch[layer]:
#            print("Saving figures and data for aquitard layer %s." % layer)
#            print("\tMaking deformation figure")
#            sns.set_style("darkgrid")
#            sns.set_context("talk")
#            plt.figure(figsize=(18, 12))
#            t = groundwater_solution_dates[layer]
#
#            plt.plot_date(t, deformation[layer]["total"][1, :])
#            # plt.plot_date(t,deformation[layer]['elastic'],label='elastic')
#            # plt.plot_date(t,deformation[layer]['inelastic'],label='inelastic')
#            plt.legend()
#            plt.savefig(
#                "%s/figures/compaction_%s.png"
#                % (outdestination, layer.replace(" ", "_")),
#                bbox_inches="tight",
#            )
#            plt.xlim(
#                mdates.date2num(
#                    [datetime.date(2015, 1, 1), datetime.date(2020, 1, 1)]
#                )
#            )
#            plt.savefig(
#                "%s/figures/compaction_%s_20152020.png"
#                % (outdestination, layer.replace(" ", "_")),
#                bbox_inches="tight",
#            )
#            plt.close()
#
#            # if save_internal_compaction:
#            #     print('\tSaving db')
#
#            #     if np.size(db[layer]) >= 1e6:
#            #         if gmt:
#            #             print('\t\t\tdb has more than 1 million entries; saving as 32 bit floats.')
#            #             db[layer].astype(np.single).tofile('%s/s_outputs/%s_db' % (outdestination, layer.replace(' ','_')))
#            #             print('\t\t\t\tConverting to netCDF format. Command is:')
#            #             cmd_tmp="gmt xyz2grd %s/s_outputs/%s_%sclay_db -G%s/s_outputs/%s_db.nc -I%.2f/%.5f -R%.2ft/%.2ft/%.2f/%.2f -ZTLf" % (outdestination, layer.replace(' ','_'),thickness,outdestination, layer.replace(' ','_'),dt_master[layer],np.diff(Z[layer][0],np.min(t_gwflow[layer]),np.max(t_gwflow[layer]),np.min(Z[layer]),np.max(Z[layer])))
#
#            #             print(cmd_tmp)
#            #             subprocess.call(cmd_tmp,shell=True)
#            #             os.remove('%s/s_outputs/%s_db' % (outdestination, layer.replace(' ','_')))
#            #         else:
#            #             print('\t\t\tdb has more than 1 million entries; saving as 16 bit floats.')
#            #             db[layer].astype(np.half).tofile('%s/s_outputs/%s_db' % (outdestination, layer.replace(' ','_')))
#            #     else:
#            #         with open('%s/%s_db.csv' % (outdestination, layer.replace(' ','_')), "w+") as myCsv:
#            #             csvWriter = csv.writer(myCsv, delimiter=',')
#            #             csvWriter.writerows(np.array(db[layer]).T)
#
#            # print('\tSaving s_elastic timeseries')
#            # np.savetxt('%s/%s_s_elastic.csv' % (outdestination, layer.replace(' ','_')),deformation[layer]['elastic'])
#
#            if save_s:
#                print("\tSaving s timeseries")
#                np.savetxt(
#                    "%s/%s_s.csv" % (outdestination, layer.replace(" ", "_")),
#                    deformation[layer]["total"],
#                )
#
#    #
#    if layer_types[layer] == "Aquifer":
#        if layer_compaction_switch[layer]:
#            if save_internal_compaction:
#                #     print('\tSaving db')
#                #     interbeds_tmp=interbeds_distributions[layer]
#                #     bed_thicknesses_tmp=list(interbeds_tmp.keys())
#
#                #     for thickness in bed_thicknesses_tmp:
#
#                #         if np.size(db[layer]['total_%.2f clays' % thickness]) >= 1e6:
#                #             if gmt:
#                #                 print('\t\t\tdb has more than 1 million entries; saving as 32 bit floats.')
#                #                 np.array(db[layer]['total_%.2f clays' % thickness]).astype(np.single).tofile('%s/s_outputs/%s_%sclay_db' % (outdestination, layer.replace(' ','_'),thickness))
#                #                 print('\t\t\t\tConverting to netCDF format. Command is:')
#                #                 Z_midpoints_tmp = [(Z[layer]['%.2f clays' % thickness][i] + Z[layer]['%.2f clays' % thickness][i+1])/2 for i in range(len(Z[layer]['%.2f clays' % thickness])-1)]
#                #                 cmd_tmp="gmt xyz2grd %s/s_outputs/%s_%sclay_db -G%s/s_outputs/%s_%sclay_db.nc -I%.2f/%.5f -R%.2ft/%.2ft/%.2f/%.2f -ZLTf" % (outdestination, layer.replace(' ','_'),thickness, outdestination, layer.replace(' ','_'),'%.2f' % thickness,dt_master[layer],np.diff(Z[layer]['%.2f clays' % thickness])[0],np.min(t_gwflow[layer]['%.2f clays' % thickness]),np.max(t_gwflow[layer]['%.2f clays' % thickness]) - np.diff(t_gwflow[layer]['%.2f clays' % thickness])[0] ,np.min(Z_midpoints_tmp),np.max(Z_midpoints_tmp))
#
#                #                 print(cmd_tmp)
#                #                 subprocess.call(cmd_tmp,shell=True)
#                #                 if os.path.isfile('%s/s_outputs/%s_%sclay_db.nc' % (outdestination, layer.replace(' ','_'),'%.2f' % thickness)):
#                #                     os.remove('%s/s_outputs/%s_%sclay_db' % (outdestination, layer.replace(' ','_'),thickness))
#                #                 else:
#                #                     print('\t\t\tSomething went wrong, .nc file not found, keeping 32 bit floats file.')
#                #             else:
#                #                 print('\t\t\tdb has more than 1 million entries; saving as 16 bit floats.')
#                #                 np.array(db[layer]['total_%.2f clays' % thickness]).astype(np.half).tofile('%s/s_outputs/%s_%sclay_db' % (outdestination, layer.replace(' ','_'),thickness))
#
#                if (
#                    np.size(
#                        inelastic_flag_compaction[layer][
#                            "elastic_%.2f clays" % thickness
#                        ]
#                    )
#                    >= 3e6
#                ):
#                    if gmt:
#                        print(
#                            "\t\t\tInelastic flag has more than 3 million entries; saving as signed char."
#                        )
#                        inelastic_flag_compaction[layer][
#                            "elastic_%.2f clays" % thickness
#                        ].astype(np.byte).tofile(
#                            "%s/s_outputs/%s_%sclayinelastic_flag_COMPACTION"
#                            % (outdestination, layer.replace(" ", "_"), thickness)
#                        )
#                        print("\t\t\t\tConverting to netCDF format. Command is:")
#                        cmd_tmp = (
#                            "gmt xyz2grd %s/s_outputs/%s_%sclayinelastic_flag_COMPACTION -G%s/s_outputs/%s_%sclayinelastic_flag_COMPACTION.nb -I%.3f/%.5f -R%.3ft/%.3ft/%.3f/%.3f -ZTLc"
#                            % (
#                                outdestination,
#                                layer.replace(" ", "_"),
#                                thickness,
#                                outdestination,
#                                layer.replace(" ", "_"),
#                                f"{thickness:.2f}",
#                                dt_master[layer],
#                                np.diff(Z[layer][f"{thickness:.2f} clays"])[0],
#                                np.min(t_gwflow[layer][f"{thickness:.2f} clays"]),
#                                np.max(t_gwflow[layer][f"{thickness:.2f} clays"]),
#                                np.min(Z[layer][f"{thickness:.2f} clays"])
#                                + np.diff(Z[layer][f"{thickness:.2f} clays"])[0] / 2,
#                                np.max(Z[layer][f"{thickness:.2f} clays"])
#                                - np.diff(Z[layer][f"{thickness:.2f} clays"])[0] / 2,
#                            )
#                        )
#
#                        print(cmd_tmp)
#                        subprocess.call(cmd_tmp, shell=True)
#                        os.remove(
#                            "%s/s_outputs/%s_%sclayinelastic_flag_COMPACTION"
#                            % (outdestination, layer.replace(" ", "_"), thickness)
#                        )
#
#                    else:
#                        print(
#                            "\t\t\tInelastic flag gwflow has more than 3 million entries; saving as signed char."
#                        )
#                        inelastic_flag_tmp.astype(np.byte).tofile(
#                            "%s/s_outputs/%s_%sclayinelastic_flag_COMPACTION"
#                            % (outdestination, layer.replace(" ", "_"), thickness)
#                        )
#
#                else:
#                    with open(
#                        "%s/s_outputs/%s_%sclayinelastic_flag_COMPACTION.csv"
#                        % (outdestination, layer.replace(" ", "_"), thickness),
#                        "w+",
#                    ) as myCsv:
#                        csvWriter = csv.writer(myCsv, delimiter=",")
#                        csvWriter.writerows(inelastic_flag_tmp)
#
#            print("Saving figures and data for aquifer layer %s." % layer)
#            if not os.path.isdir("%s/figures/%s" % (outdestination, layer)):
#                os.mkdir("%s/figures/%s" % (outdestination, layer))
#
#            print("\tSaving layer data.")
#            print("")
#            print("printing def output thing")
#            print(deformation_OUTPUT[layer])
#            if len(layers_var_thickness) >= 1:
#                if layer in layers_var_thickness:
#                    print("")
#                    print(
#                        "\tScaling sublayer outputs by temporally varying layer thicknesses."
#                    )
#                    interbeds_tmp = interbeds_distributions[layer]
#                    bed_thicknesses_tmp = list(interbeds_tmp.keys())
#                    print(
#                        "\t\t%s is an aquifer with interbedded clays. Scaling thickness of clays: %s"
#                        % (layer, bed_thicknesses_tmp)
#                    )
#                    print("\t\tFirst scaling the interbeds....")
#                    for thickness in bed_thicknesses_tmp:
#                        prekeyname = np.array(list(layer_thicknesses[layer].keys()))[
#                            np.where(
#                                [
#                                    "pre" in key
#                                    for key in list(layer_thicknesses[layer].keys())
#                                ]
#                            )[0][0]
#                        ]
#                        datetimedates = [
#                            datetime.datetime.strptime(d, "%d-%b-%Y")
#                            for d in deformation_OUTPUT[layer]["dates"].values
#                        ]
#                        logicaltmp = [
#                            datetimedate
#                            <= datetime.datetime(
#                                int("%s" % prekeyname.split("-")[1]), 9, 1
#                            )
#                            for datetimedate in datetimedates
#                        ]
#
#                        scaling_factor_tmp = (
#                            layer_thicknesses[layer][prekeyname]
#                            / initial_thicknesses[layer]
#                        )
#                        deformation_scaled_tmp = (
#                            deformation_OUTPUT[layer]["total_%.2f clays" % thickness][
#                                logicaltmp
#                            ].values
#                            * scaling_factor_tmp
#                        )
#                        print(deformation_scaled_tmp)
#                        nonprekeynames = np.array(
#                            list(layer_thicknesses[layer].keys())
#                        )[
#                            np.where(
#                                [
#                                    "pre" not in key
#                                    for key in list(layer_thicknesses[layer].keys())
#                                ]
#                            )[0]
#                        ]
#                        nonprekeynames.sort()
#                        for key in nonprekeynames:
#                            if not key.endswith("-"):
#                                scaling_factor_tmp = (
#                                    layer_thicknesses[layer][key]
#                                    / initial_thicknesses[layer]
#                                )
#                                years_tmp = key.split("-")
#                                print(
#                                    "\t\tScaling years",
#                                    years_tmp,
#                                    "by ",
#                                    scaling_factor_tmp,
#                                )
#                                logicaltmp = [
#                                    (
#                                        datetimedate
#                                        <= datetime.datetime(
#                                            int("%s" % years_tmp[1]),
#                                            9,
#                                            1,
#                                        )
#                                    )
#                                    and (
#                                        datetimedate
#                                        > datetime.datetime(
#                                            int("%s" % years_tmp[0]), 9, 1
#                                        )
#                                    )
#                                    for datetimedate in datetimedates
#                                ]
#                                deformation_scaled_tmp = np.append(
#                                    deformation_scaled_tmp,
#                                    (
#                                        deformation_OUTPUT[layer][
#                                            "total_%.2f clays" % thickness
#                                        ][logicaltmp]
#                                        - deformation_OUTPUT[layer][
#                                            "total_%.2f clays" % thickness
#                                        ][np.where(logicaltmp)[0][0] - 1]
#                                    )
#                                    * scaling_factor_tmp
#                                    + deformation_scaled_tmp[-1],
#                                )
#                        for key in nonprekeynames:
#                            if key.endswith("-"):
#                                scaling_factor_tmp = (
#                                    layer_thicknesses[layer][key]
#                                    / initial_thicknesses[layer]
#                                )
#                                years_tmp = key.split("-")
#                                print(
#                                    "\t\tScaling years",
#                                    years_tmp,
#                                    "by ",
#                                    scaling_factor_tmp,
#                                )
#                                logicaltmp = [
#                                    datetimedate
#                                    > datetime.datetime(int("%s" % years_tmp[0]), 9, 1)
#                                    for datetimedate in datetimedates
#                                ]
#                                deformation_scaled_tmp = np.append(
#                                    deformation_scaled_tmp,
#                                    (
#                                        deformation_OUTPUT[layer][
#                                            "total_%.2f clays" % thickness
#                                        ][logicaltmp]
#                                        - deformation_OUTPUT[layer][
#                                            "total_%.2f clays" % thickness
#                                        ][np.where(logicaltmp)[0][0] - 1]
#                                    )
#                                    * scaling_factor_tmp
#                                    + deformation_scaled_tmp[-1],
#                                )
#                        deformation_OUTPUT[layer][
#                            "total_%.2f clays" % thickness
#                        ] = deformation_scaled_tmp
#
#                    print("\t\tNow scaling the elastic, and recomputing the total")
#                    scaling_factor_tmp = (
#                        layer_thicknesses[layer][prekeyname]
#                        / initial_thicknesses[layer]
#                    )
#                    logicaltmp = [
#                        datetimedate
#                        <= datetime.datetime(int("%s" % prekeyname.split("-")[1]), 9, 1)
#                        for datetimedate in datetimedates
#                    ]
#                    deformation_scaled_tmp = (
#                        deformation_OUTPUT[layer]["Interconnected Matrix"][
#                            logicaltmp
#                        ].values
#                        * scaling_factor_tmp
#                    )
#                    print(deformation_scaled_tmp)
#                    nonprekeynames = np.array(list(layer_thicknesses[layer].keys()))[
#                        np.where(
#                            [
#                                "pre" not in key
#                                for key in list(layer_thicknesses[layer].keys())
#                            ]
#                        )[0]
#                    ]
#                    nonprekeynames.sort()
#                    for key in nonprekeynames:
#                        if not key.endswith("-"):
#                            scaling_factor_tmp = (
#                                layer_thicknesses[layer][key]
#                                / initial_thicknesses[layer]
#                            )
#                            years_tmp = key.split("-")
#                            print(
#                                "\t\tScaling years",
#                                years_tmp,
#                                "by ",
#                                scaling_factor_tmp,
#                            )
#                            logicaltmp = [
#                                (
#                                    datetimedate
#                                    <= datetime.datetime(
#                                        int("%s" % years_tmp[1]),
#                                        9,
#                                        1,
#                                    )
#                                )
#                                and (
#                                    datetimedate
#                                    > datetime.datetime(int("%s" % years_tmp[0]), 9, 1)
#                                )
#                                for datetimedate in datetimedates
#                            ]
#                            deformation_scaled_tmp = np.append(
#                                deformation_scaled_tmp,
#                                (
#                                    deformation_OUTPUT[layer]["Interconnected Matrix"][
#                                        logicaltmp
#                                    ]
#                                    - deformation_OUTPUT[layer][
#                                        "Interconnected Matrix"
#                                    ][np.where(logicaltmp)[0][0] - 1]
#                                )
#                                * scaling_factor_tmp
#                                + deformation_scaled_tmp[-1],
#                            )
#                    for key in nonprekeynames:
#                        if key.endswith("-"):
#                            scaling_factor_tmp = (
#                                layer_thicknesses[layer][key]
#                                / initial_thicknesses[layer]
#                            )
#                            years_tmp = key.split("-")
#                            print(
#                                "\t\tScaling years",
#                                years_tmp,
#                                "by ",
#                                scaling_factor_tmp,
#                            )
#                            logicaltmp = [
#                                datetimedate
#                                > datetime.datetime(int("%s" % years_tmp[0]), 9, 1)
#                                for datetimedate in datetimedates
#                            ]
#                            deformation_scaled_tmp = np.append(
#                                deformation_scaled_tmp,
#                                (
#                                    deformation_OUTPUT[layer]["Interconnected Matrix"][
#                                        logicaltmp
#                                    ]
#                                    - deformation_OUTPUT[layer][
#                                        "Interconnected Matrix"
#                                    ][np.where(logicaltmp)[0][0] - 1]
#                                )
#                                * scaling_factor_tmp
#                                + deformation_scaled_tmp[-1],
#                            )
#                    deformation_OUTPUT[layer][
#                        "Interconnected Matrix"
#                    ] = deformation_scaled_tmp
#
#                    def_tot_tmp = np.zeros_like(deformation_scaled_tmp, dtype="float")
#
#                    def_tot_tmp += np.array(
#                        deformation_OUTPUT[layer]["Interconnected Matrix"]
#                    )
#                    for thickness in bed_thicknesses_tmp:
#                        def_tot_tmp += np.array(
#                            deformation_OUTPUT[layer]["total_%.2f clays" % thickness]
#                        )
#
#                    deformation_OUTPUT[layer]["total"] = def_tot_tmp
#
#            deformation_OUTPUT[layer].to_csv(
#                "%s/%s_Total_Deformation_Out.csv" % (outdestination, layer), index=False
#            )
#
#            print("\tSaving layer compaction figure")
#            sns.set_style("whitegrid")
#            sns.set_context("talk")
#            l_aqt = []
#
#            plt.figure(figsize=(18, 12))
#            (line_tmp,) = plt.plot_date(
#                head_series[layer]["Interconnected matrix"][:, 0],
#                deformation_OUTPUT[layer]["Interconnected Matrix"],
#                "-",
#                label="Interconnected matrix",
#            )
#            l_aqt.append(line_tmp)
#
#            interbeds_tmp = interbeds_distributions[layer]
#            bed_thicknesses_tmp = list(interbeds_tmp.keys())
#
#            for thickness in bed_thicknesses_tmp:
#                (line_tmp,) = plt.plot_date(
#                    [
#                        datetime.datetime.strptime(d, "%d-%b-%Y")
#                        for d in deformation_OUTPUT[layer]["dates"].values
#                    ],
#                    deformation_OUTPUT[layer]["total_%.2f clays" % thickness],
#                    "-",
#                    label="%s_%ix%.2f clays"
#                    % (layer, interbeds_distributions[layer][thickness], thickness),
#                )
#                l_aqt.append(line_tmp)
#
#            (line_tmp,) = plt.plot_date(
#                deformation[layer]["total"][0, :],
#                deformation_OUTPUT[layer]["total"],
#                "-",
#                label="total",
#            )
#            l_aqt.append(line_tmp)
#            plt.title("%s" % layer)
#            plt.ylabel("Deformation (m)")
#            plt.legend()
#            plt.savefig(
#                os.makedirs(
#                    os.path.join("Output", "test2", "figures", layer), exist_ok=True
#                )
#                or os.path.join(
#                    "Output",
#                    "test2",
#                    "figures",
#                    layer,
#                    f"overall_compaction_{layer}.png",
#                ),
#                bbox_inches="tight",
#            )
#            #            plt.savefig(outdestination+"/figures/"+layer+"/"+"overall_compaction_"+layer+".png",bbox_inches='tight')
#            #               '%s/figures/%s/overall_compaction_%s.png' % (outdestination,layer,layer),bbox_inches='tight')
#            #            if np.min(line_tmp.get_xdata()) <= date2num(date(2015,1,1)):
#            #                for line in l_aqt:
#            #                    line.set_ydata(np.array(line.get_ydata()) - np.array(line.get_ydata())[np.array(line.get_xdata())==date2num(date(2015,1,1))])
#            #
#            #                plt.xlim(date2num([date(2015,1,1),date(2020,1,1)]))
#            #
#            #                plt.savefig('%s/figures/%s/overall_compaction_%s_201520.png' % (outdestination,layer,layer),bbox_inches='tight')
#            plt.close()
#
#            if save_s:
#                print("\tSaving s (interconnected matrix)")
#
#                if np.size(deformation[layer]["Interconnected matrix"]) >= 3e6:
#                    print(
#                        "\t\t\ts (interconnected matrix) has more than 1 million entries; saving as 32 bit floats."
#                    )
#                    deformation[layer]["Interconnected matrix"].astype(
#                        np.single
#                    ).tofile(
#                        "%s/s_outputs/%s_s_matrix"
#                        % (outdestination, layer.replace(" ", "_"))
#                    )
#
#                else:
#                    np.savetxt(
#                        "%s/s_outputs/%s_s_matrix.csv"
#                        % (outdestination, layer.replace(" ", "_")),
#                        deformation[layer]["Interconnected matrix"],
#                    )
#
#                print("\tSaving s (clay layers)")
#                for thickness in bed_thicknesses_tmp:
#                    print("\t\t%.2f" % thickness)
#                    if (
#                        np.size(deformation[layer]["total_%.2f clays" % thickness])
#                        >= 1e6
#                    ):
#                        print(
#                            "\t\t\ts (clay) has more than 1 million entries; saving as 32 bit floats."
#                        )
#                        deformation[layer]["total_%.2f clays" % thickness].astype(
#                            np.single
#                        ).tofile(
#                            "%s/s_outputs/%s_s_%.2fclays"
#                            % (outdestination, layer.replace(" ", "_"), thickness)
#                        )
#                    else:
#                        np.savetxt(
#                            "%s/s_outputs/%s_s_%.2fclays.csv"
#                            % (outdestination, layer.replace(" ", "_"), thickness),
#                            deformation[layer]["total_%.2f clays" % thickness],
#                        )
#
#
# print("Creating overall compaction plot and saving deformation series")
# sns.set_style("whitegrid")
# plt.figure(figsize=(18, 12))
# dt_master_compacting_layers = {key: dt_master[key] for key in compacting_layers}
# maxdt = max(dt_master_compacting_layers.values())
# maxdtlayer = max(dt_master_compacting_layers.items(), key=operator.itemgetter(1))[0]
# dt_interconnecteds = [
#    dt_headseries[layer]
#    for layer in layer_names
#    if (layer in compacting_layers) and (layer_types[layer] == "Aquifer")
# ]
# if np.max(dt_interconnecteds) > maxdt:
#    maxdt = np.max(dt_interconnecteds)
# print("\tmax dt (output dt) = %.2f" % maxdt)
# deformation_OUTPUT = {}
## t_total_tmp = 0.0001*np.arange(10000*np.min(deformation[maxdtlayer]['total'][0,:]),10000*(np.max(deformation[maxdtlayer]['total'][0,:]+0.001)),10000*maxdt) <<I think this line is wrong again. Removing and seeing if it works
# print(t_total_tmp)
# deformation_OUTPUT["dates"] = [
#    x.strftime("%d-%b-%Y") for x in mdates.num2date(t_total_tmp)
# ]
# t_overall = np.zeros_like(t_total_tmp, dtype=float)
# l_aqt = []
# l_aqf = []
# for layer in layer_names:
#    if layer_compaction_switch[layer]:
#        if layer_types[layer] == "Aquitard":
#            # dt_tmp = int(maxdt/dt_master[layer])
#            #            t_overall = t_overall + deformation[layer]['total'][1,::dt_tmp]
#            (l_tmp,) = plt.plot_date(
#                deformation[layer]["total"][0, :],
#                deformation[layer]["total"][1, :],
#                "-",
#                label="%s" % layer,
#            )
#            l_aqt.append(l_tmp)
#            deformation_OUTPUT[layer] = deformation[layer]["total"][1, :][
#                np.isin(deformation[layer]["total"][0, :], t_total_tmp)
#            ]
#
#        if layer_types[layer] == "Aquifer":
#            (l_tmp,) = plt.plot_date(
#                deformation[layer]["total"][0, :],
#                deformation[layer]["total"][1, :],
#                "-",
#                label="%s" % layer,
#            )
#            l_aqf.append(l_tmp)
#            # dt_tmp = int(maxdt/dt_master[layer])
#            deformation_OUTPUT[layer] = deformation[layer]["total"][1, :][
#                np.isin(deformation[layer]["total"][0, :], t_total_tmp)
#            ]
#
## Add up all the deformations from each layer
#
# def_tot_tmp = np.zeros_like(t_total_tmp, dtype="float")
# for layer in layer_names:
#    if layer_compaction_switch[layer]:
#        newtot = np.array(deformation[layer]["total"][1, :])[
#            np.isin(deformation[layer]["total"][0, :], t_total_tmp)
#        ]
#        t_overall = t_overall + newtot
#
# deformation_OUTPUT["Total"] = t_overall
# def_out = pd.DataFrame(deformation_OUTPUT)
# def_out.to_csv("%s/Total_Deformation_Out.csv" % outdestination, index=False)
#
#
# (l3,) = plt.plot_date(t_total_tmp, t_overall, label="TOTAL def")
#
# plt.ylabel("Z (m)")
# plt.legend()
# plt.savefig(
#    "%s/figures/total_deformation_figure.png" % outdestination, bbox_inches="tight"
# )
## Rezero on jan 2015
# plt.xlim(
#    mdates.date2num([datetime.date(2015, 1, 1), datetime.date(2020, 1, 1)])
# )
# if np.min(l_tmp.get_xdata()) <= mdates.date2num(datetime.date(2015, 1, 1)):
#    for line in l_aqt:
#        line.set_ydata(
#            np.array(line.get_ydata())
#            - np.array(line.get_ydata())[
#                np.array(line.get_xdata())
#                == mdates.date2num(datetime.date(2015, 1, 1))
#            ]
#        )
#
#    # rescale axis
#    ax = plt.gca()
#    # recompute the ax.dataLim
#    ax.relim()
#    # update ax.viewLim using the new dataLim
#    ax.autoscale()
#    plt.draw()
#    plt.savefig(
#        "%s/figures/total_deformation_figure_20152020.png" % outdestination,
#        bbox_inches="tight",
#    )
# plt.close()
#
# saving_compaction_stop = time.time()
# saving_compaction_time = saving_compaction_stop - saving_compaction_start
#
# t_total_stop = time.time()
# t_total = t_total_stop - t_total_start
#
# plt.figure(figsize=(18, 18))
# plt.pie(
#    np.abs(
#        [
#            param_read_time,
#            solving_compaction_time,
#            saving_head_time,
#            reading_head_time,
#            solving_head_time,
#            saving_compaction_time,
#            t_total
#            - np.sum(
#                [
#                    param_read_time,
#                    solving_compaction_time,
#                    saving_head_time,
#                    reading_head_time,
#                    solving_head_time,
#                    saving_compaction_time,
#                ]
#            ),
#        ]
#    ),
#    labels=[
#        "param_read_time",
#        "solving_compaction_time",
#        "saving_head_time",
#        "reading_head_time",
#        "solving_head_time",
#        "saving_compaction_time",
#        "misc",
#    ],
#    autopct=lambda p: "{:.2f}%  ({:,.0f})".format(p, p * t_total / 100),
# )
# plt.title("Total runtime = %i seconds" % t_total)
# plt.savefig("%s/figures/runtime_breakdown.png" % outdestination, bbox_inches="tight")
# plt.close()
#
#
# print("Model Run Complete")
#
