"""Shared fixtures for SUB1D tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simple_grid():
    """A simple 1D grid for testing solvers."""
    n_nodes = 21
    L = 1.0
    dx = L / (n_nodes - 1)
    x = np.linspace(0, L, n_nodes)
    return x, dx, L


@pytest.fixture
def diffusion_params():
    """Standard diffusion parameters."""
    return {
        "k": 1.0,       # diffusivity
        "dt": 0.0001,   # timestep (small for stability)
        "n_steps": 1000,
    }


@pytest.fixture
def sample_par_content():
    """Sample .par file content for config testing."""
    return """### MODEL INPUT PARAMETER FILE ###
# Test configuration

## Admin
run_name=test_run
output_folder=Output/
overwrite=False

## Hydrostratigraphy
no_layers=3
layer_names=Upper Aquifer,Clay Layer,Lower Aquifer
layer_types=Upper Aquifer:Aquifer,Clay Layer:Aquitard,Lower Aquifer:Aquifer
layer_thickness_types=Upper Aquifer:constant,Clay Layer:constant,Lower Aquifer:constant
layer_thicknesses=Upper Aquifer:50,Clay Layer:10,Lower Aquifer:100
layer_compaction_switch=Upper Aquifer:1,Clay Layer:1,Lower Aquifer:1
interbeds_switch=Upper Aquifer:1,Lower Aquifer:0
interbeds_distributions=Upper Aquifer:{5:2,10:1}
initial_stress_type=Upper Aquifer:initial_equilibrium,Clay Layer:initial_equilibrium,Lower Aquifer:initial_equilibrium

## Input head
head_data_files=Upper Aquifer:upper_heads.csv,Lower Aquifer:lower_heads.csv

## Solver
dt_master=Upper Aquifer:0.5,Clay Layer:1,Lower Aquifer:0.5
dz_clays=Upper Aquifer:0.2,Clay Layer:0.5,Lower Aquifer:0.2
groundwater_flow_solver_type=Upper Aquifer:elastic-inelastic,Clay Layer:elastic-inelastic,Lower Aquifer:elastic-inelastic
overburden_stress_gwflow=True

## Compaction
compaction_solver_compressibility_type=Clay Layer:elastic-inelastic
overburden_stress_compaction=True

## Hydrology
clay_Sse=Upper Aquifer:1e-4,Clay Layer:1e-4,Lower Aquifer:1e-4
clay_Ssv=Upper Aquifer:1e-3,Clay Layer:1e-3,Lower Aquifer:1e-3
sand_Sse=Upper Aquifer:3.3e-5,Lower Aquifer:3.3e-5
compressibility_of_water=1.5e-6
vertical_conductivity=Upper Aquifer:1e-6,Clay Layer:1e-6,Lower Aquifer:1e-6
rho_w=1000
g=9.81
specific_yield=0.2
time_unit=days

## Output
save_effective_stress=False
save_internal_compaction=False
save_output_head_timeseries=False
create_output_head_video=Upper Aquifer:False,Clay Layer:False,Lower Aquifer:False
save_s=False
preconsolidation_head_type=initial
compaction_solver_debug_include_endnodes=False
mode=Normal
"""
