import sys
import re
import distutils
import numpy as np

from utils import make_output_folder

# Define parameter default values. We will then read and overwrite any from the parameter file.
DEFAULTS_DICTIONARY = {
    "internal_time_delay": True,
    "overwrite": True,
    "run_name": False,
    "output_folder": False,
    "no_layers": True,
    "layer_names": True,
    "layer_types": True,
    "layer_thicknesses": True,
    "layer_compaction_switch": True,
    "interbeds_switch": True,
    "interbeds_type": False,
    "clay_Ssk_type": False,
    "clay_Ssk": False,
    "sand_Ssk": True,
    "compressibility_of_water": True,
    "dz_clays": True,
    "dt_gwaterflow": True,
    "create_output_head_video": True,
    "overburden_stress_gwflow": True,
    "overburden_stress_compaction": True,
    "rho_w": True,
    "g": True,
    "specific_yield": True,
    "save_effective_stress": True,
    "time_unit": True,
    "compaction_solver_debug_include_endnodes": True,
    "save_internal_compaction": True,
    "mode": True,
    "resume_directory": False,
    "resume_date": False,
    "layer_thickness_types": True,
    "compaction_solver_compressibility_type": False,
    "preconsolidation_head_type": True,
    "save_s": True,
}  # Define which variables have pre-defined defaults

DEFAULT_VALUES = {
    "internal_time_delay": 0.5,
    "overwrite": False,
    "no_layers": 2,
    "layer_names": ["Upper Aquifer", "Lower Aquifer"],
    "layer_types": {"Upper Aquifer": "Aquifer", "Lower Aquifer": "Aquifer"},
    "layer_thicknesses": {"Upper Aquifer": 100.0, "Lower Aquifer": 100.0},
    "layer_compaction_switch": {"Upper Aquifer": True, "Lower Aquifer": True},
    "interbeds_switch": {"Upper Aquifer": False, "Lower Aquifer": False},
    "sand_Ssk": 1,
    "compressibility_of_water": 4.4e-10,
    "dz_clays": 0.3,
    "dt_gwaterflow": 1,
    "create_output_head_video": False,
    "overburden_stress_gwflow": False,
    "overburden_stress_compaction": False,
    "rho_w": 1000,
    "g": 9.81,
    "specific_yield": 0.2,
    "save_effective_stress": False,
    "time_unit": "days",
    "compaction_solver_debug_include_endnodes": False,
    "save_internal_compaction": False,
    "mode": "Normal",
    "layer_thickness_types": "constant",
    "preconsolidation_head_type": "initial",
    "save_s": False,
}

def parse_parameter_file(param_filename):
    with open(param_filename, "r") as f:
        paramfilelines = f.readlines()

    paramfilelines = [line.strip() for line in paramfilelines]
    paramfilelines = [x for x in paramfilelines if not x.startswith("#")]
    
    return [x for x in paramfilelines if x]


def read_parameter(name, typ, length, paramfilelines, printlots=True):
    """Reads in a parameter from paramfilelines.
    name=parameter name
    typ=parameter type (str,int,float). if length>1 this is the type that each entry in the list/dictionary will be.
    length=how many comma separated entries are there for the parameter"""

    par_paramfile = [
        x for x in paramfilelines if x.replace(" ", "").startswith("%s=" % name)
    ]
    if len(par_paramfile) == 0:
        if not DEFAULTS_DICTIONARY[name]:
            print(
                "\tReading parameters: NOTICE. No '%s' found in parameter file and no default set."
                % name
            )
            return

        else:
            print(
                "\t\tReading parameters error: WARNING. No '%s' found in parameter file. Using default value."
                % name
            )
            par = DEFAULT_VALUES[name]
            if printlots:
                print("\t%s=%s" % (name, par))
            return par

    par = par_paramfile[0].split("#")[0].split("=")[1]

    if length == 1:
        par = par.strip()
    else:
        if ":" not in par:
            par = par.split(",")
            if len(par) != length:
                print(
                    "\t\tReading parameters error: terminal. %s should have %i entries but only has %i."
                    % (name, length, len(par))
                )
                sys.exit(1)
            par = [x.strip() for x in par]

    if ":" in par:
        if typ != dict:
            par = re.split(":|,", par)
            if len(par) != length * 2:
                print("\tERROR: %s=%s" % (name, par))
                print(
                    "\t\tReading parameters error: terminal. %s should have %i entries but has %i."
                    % (name, length, 0.5 * len(par))
                )
                sys.exit(1)
        else:
            par = re.split("},", par)
            b = [re.split(":{", p.replace("}", "").strip()) for p in par]
            return b
        if typ != bool:
            par = dict(
                [
                    (par[2 * i].strip(), typ(par[2 * i + 1].strip()))
                    for i in range(length)
                ]
            )
        else:
            par = dict(
                [
                    (
                        par[2 * i].strip(),
                        bool(distutils.util.strtobool(par[2 * i + 1].strip())),
                    )
                    for i in range(length)
                ]
            )

    if length == 1:
        if type(par) != dict:
            if typ == bool:
                par = bool(distutils.util.strtobool(par))
            else:
                par = typ(par)

    if printlots:
        print("\t%s=%s" % (name, par))

    if length == 1:
        if type(par) != dict:
            if type(par) != typ:
                print(
                    "\t\tReading parameters error: WARNING. %s is of class %s but may need to be %s. This may lead to errors later."
                    % (name, type(par), typ)
                )
        else:
            if type(par) != dict:
                iscorrect = [type(x) == typ for x in par]
                if False in iscorrect:
                    print(
                        "\t\tReading parameters error: WARNING. elements of %s should be %s but 1 or more is not. This may lead to errors later."
                        % (name, typ)
                    )

    if len(par_paramfile) > 1:
        print(
            "\t\tReading parameters error: WARNING. Multiple '%s's found in parameter file. using the first."
            % name
        )

    if type(par) == dict:
        iscorrect = [type(x) == typ for x in par.values()]
        if False in iscorrect:
            print(
                "\t\tReading parameters error: WARNING. values of %s should be %s but 1 or more is not. This may lead to errors later."
                % (name, typ)
            )

    return par


def read_parameter_layerthickness_multitype(name, paramfilelines, printlots=True):
    par_out = {}
    par_paramfile = [
        x for x in paramfilelines if x.replace(" ", "").startswith("%s=" % name)
    ]
    par = par_paramfile[0].split("#")[0].split("=")[1]
    par = re.split(":|,", par)
    # Find if there are any dictionaries
    contains_dict = ["{" in s or "}" in s for s in par]
    if sum(contains_dict) % 2 != 0:
        print("\tERROR: odd number of { or } brackets found." % (name, par))
        print(
            "\t\tReading parameters error: terminal. %s should have even number of { } brackets."
            % name
        )
    elif sum(contains_dict) > 0:
        for i_tmp in range(int(len(np.where(contains_dict)[0]) / 2)):
            layername_tmp = par[np.where(contains_dict)[0][2 * i_tmp] - 1]
            par_out[layername_tmp] = {}
            dic_length_tmp = int(
                (
                    np.where(contains_dict)[0][2 * i_tmp + 1]
                    - np.where(contains_dict)[0][2 * i_tmp]
                    + 1
                )
                / 2
            )
            for j_tmp in range(dic_length_tmp):
                keytmp = par[np.where(contains_dict)[0][2 * i_tmp] + j_tmp * 2].split(
                    "{"
                )[-1]
                val_tmp = par[
                    np.where(contains_dict)[0][2 * i_tmp] + 1 + j_tmp * 2
                ].split("}")[0]
                par_out[layername_tmp][keytmp] = float(val_tmp)
    dict_idxs_full = []
    for i_tmp in range(int(len(np.where(contains_dict)[0]) / 2)):
        dict_idxs_full = np.append(
            dict_idxs_full,
            np.arange(
                np.where(contains_dict)[0][2 * i_tmp] - 1,
                np.where(contains_dict)[0][2 * i_tmp + 1] + 1,
            ),
        )
    all_idxs = np.arange(len(par))
    nondict_idxs = [idx for idx in all_idxs if idx not in dict_idxs_full]
    nondict_par = np.array(par)[nondict_idxs]
    for i_tmp in range(int(len(nondict_par) / 2)):
        par_out[nondict_par[2 * i_tmp]] = float(nondict_par[2 * i_tmp + 1])
    if printlots:
        print("\t%s=%s" % (name, par_out))
    return par_out


def read_parameters_admin(param_filename, paramfilelines, UserOverwriteYes):
    internal_time_delay = read_parameter(
        "internal_time_delay", float, 1, paramfilelines
    )
    overwrite = read_parameter("overwrite", bool, 1, paramfilelines)
    run_name = read_parameter("run_name", str, 1, paramfilelines)
    output_folder = read_parameter("output_folder", str, 1, paramfilelines)
    outdestination = "%s/%s" % (output_folder, run_name)
    make_output_folder(param_filename, outdestination, overwrite, UserOverwriteYes)
    return internal_time_delay, overwrite, run_name, output_folder, outdestination


def read_parameters_noadmin(paramfilelines):
    # Function to read all parameters; defined as a function again for neatness.

    save_output_head_timeseries = read_parameter(
        "save_output_head_timeseries", bool, 1, paramfilelines
    )
    save_effective_stress = read_parameter(
        "save_effective_stress", bool, 1, paramfilelines
    )
    save_internal_compaction = read_parameter(
        "save_internal_compaction", bool, 1, paramfilelines
    )
    no_layers = read_parameter("no_layers", int, 1, paramfilelines)
    layer_names = read_parameter("layer_names", str, no_layers, paramfilelines)
    if no_layers == 1:
        layer_names = np.array([layer_names])
    layer_types = read_parameter("layer_types", str, no_layers, paramfilelines)
    no_aquifers = list(layer_types.values()).count("Aquifer")
    no_aquitards = list(layer_types.values()).count("Aquitard")
    print("\t\tNumber of aquifer layers calculated to be %i." % no_aquifers)
    print("\t\tNumber of aquitard layers calculated to be %i." % no_aquitards)
    layer_thickness_types = read_parameter(
        "layer_thickness_types", str, no_layers, paramfilelines
    )
    layer_thicknesses = read_parameter_layerthickness_multitype(
        "layer_thicknesses", paramfilelines
    )
    layer_compaction_switch = read_parameter(
        "layer_compaction_switch", bool, no_layers, paramfilelines
    )
    interbeds_switch = read_parameter(
        "interbeds_switch",
        bool,
        list(layer_types.values()).count("Aquifer"),
        paramfilelines,
    )
    preconsolidation_head_type = read_parameter(
        "preconsolidation_head_type", str, 1, paramfilelines
    )
    if preconsolidation_head_type == "initial_plus_offset":
        preconsolidation_head_offset = read_parameter(
            "preconsolidation_head_offset", float, no_aquifers, paramfilelines
        )
    else:
        preconsolidation_head_offset = False
    # interbeds_type=read_parameter('interbeds_type',str,list(layer_types.values()).count('Aquifer'),paramfilelines)
    # Import interbeds_distributions -- an awkward parameter as its a dictionary of dictionaries!
    interbeds_distributions1 = read_parameter(
        "interbeds_distributions", dict, sum(interbeds_switch.values()), paramfilelines
    )
    interbeds_distributions1 = np.array(interbeds_distributions1)
    if np.shape(interbeds_distributions1)[0] == 1:
        interbeds_distributions1 = interbeds_distributions1[0]
        minidics = [
            dict(
                [
                    (
                        float(
                            re.split(",|:", interbeds_distributions1[2 * i + 1])[2 * j]
                        ),
                        float(
                            re.split(",|:", interbeds_distributions1[2 * i + 1])[
                                2 * j + 1
                            ]
                        ),
                    )
                    for j in range(
                        int(
                            len(re.split(",|:", interbeds_distributions1[2 * i + 1]))
                            / 2
                        )
                    )
                ]
            )
            for i in range(sum(interbeds_switch.values()))
        ]
        interbeds_distributions = dict(
            [
                (interbeds_distributions1[2 * i], minidics[i])
                for i in range(sum(interbeds_switch.values()))
            ]
        )
        print("\tinterbeds_distributions=%s" % interbeds_distributions)
    else:
        interbeds_distributions = {}
        for abc in interbeds_distributions1:
            interbeds_distributions[abc[0]] = dict(
                [
                    (
                        float(re.split(":|,", abc[1])[2 * i]),
                        float(re.split(":|,", abc[1])[2 * i + 1]),
                    )
                    for i in range(len(re.split(",", abc[1])))
                ]
            )
        print("\tinterbeds_distributions=%s" % interbeds_distributions)

    aquitards = [name for name, value in layer_types.items() if value == "Aquitard"]
    interbedded_layers = [
        name for name, value in interbeds_switch.items() if value == True
    ]
    no_layers_containing_clay = len(aquitards) + len(interbedded_layers)
    # no_layers_containing_clay = list(layer_types.values()).count('Aquitard') + sum(list(interbeds_switch.values()))
    print(
        "\t\tNumber of layers containing clay calculated to be %i."
        % no_layers_containing_clay
    )

    layers_requiring_solving = interbedded_layers + aquitards
    create_output_head_video = read_parameter(
        "create_output_head_video", bool, no_layers_containing_clay, paramfilelines
    )
    groundwater_flow_solver_type = read_parameter(
        "groundwater_flow_solver_type",
        str,
        len(layers_requiring_solving),
        paramfilelines,
    )
    if False in [
        x == "singlevalue" or x == "elastic-inelastic"
        for x in groundwater_flow_solver_type.values()
    ]:
        print(
            "\t\tReading parameters error: terminal. Only groundwater_flow_solver_type of 'singlevalue' or 'elastic-inelastic' currently supported."
        )
        sys.exit(1)
    overburden_stress_gwflow = read_parameter(
        "overburden_stress_gwflow", bool, 1, paramfilelines
    )
    compaction_solver_compressibility_type = read_parameter(
        "compaction_solver_compressibility_type", str, 1, paramfilelines
    )
    compaction_solver_debug_include_endnodes = read_parameter(
        "compaction_solver_debug_include_endnodes", bool, 1, paramfilelines
    )

    clay_Ssk = read_parameter(
        "clay_Ssk",
        float,
        sum(value == "singlevalue" for value in groundwater_flow_solver_type.values()),
        paramfilelines,
    )
    clay_Sse = read_parameter(
        "clay_Sse",
        float,
        sum(
            groundwater_flow_solver_type[layer] == "elastic-inelastic"
            or compaction_solver_compressibility_type[layer] == "elastic-inelastic"
            for layer in layer_names
        ),
        paramfilelines,
    )
    clay_Ssv = read_parameter(
        "clay_Ssv",
        float,
        sum(
            groundwater_flow_solver_type[layer] == "elastic-inelastic"
            or compaction_solver_compressibility_type[layer] == "elastic-inelastic"
            for layer in layer_names
        ),
        paramfilelines,
    )
    sand_Sse = read_parameter("sand_Sse", float, no_aquifers, paramfilelines)

    time_unit = read_parameter("time_unit", str, 1, paramfilelines)

    # clay_porosity = read_parameter('clay_porosity',float,no_layers_containing_clay,paramfilelines)
    sand_Ssk = read_parameter("sand_Ssk", float, no_aquifers, paramfilelines)
    compressibility_of_water = read_parameter(
        "compressibility_of_water", float, 1, paramfilelines
    )
    rho_w = read_parameter("rho_w", float, 1, paramfilelines)
    g = read_parameter("g", float, 1, paramfilelines)
    dt_master = read_parameter(
        "dt_master", float, no_layers_containing_clay, paramfilelines
    )
    dz_clays = read_parameter(
        "dz_clays", float, no_layers_containing_clay, paramfilelines
    )
    vertical_conductivity = read_parameter(
        "vertical_conductivity", float, len(layers_requiring_solving), paramfilelines
    )
    overburden_stress_compaction = read_parameter(
        "overburden_stress_compaction", bool, 1, paramfilelines
    )
    # overburden_compaction = read_parameter('overburden_compaction',bool,1,paramfilelines)
    save_s = read_parameter("save_s", bool, 1, paramfilelines)
    if (
        overburden_stress_gwflow or overburden_stress_compaction
    ):  # Only used if we're doing overburden anywhere
        specific_yield = read_parameter("specific_yield", float, 1, paramfilelines)
    else:
        specific_yield = None

    return (
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
    )
