# SUB1D

**1D Land Subsidence / Compaction Model** (v2.0.0)

A modular Python package for simulating groundwater-flow-driven land subsidence in layered aquifer-aquitard systems. SUB1D solves one-dimensional vertical diffusion equations for hydraulic head through clay layers, then computes elastic and inelastic compaction to predict land-surface deformation over time.

Originally based on the model described in Lees et al. (2022), this version has been restructured into a tested, modular package with vectorized solvers, implicit time-stepping options, and modern configuration support.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Physical Assumptions](#physical-assumptions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Input Files](#input-files)
  - [Configuration File (YAML)](#configuration-file-yaml)
  - [Configuration File (Legacy .par)](#configuration-file-legacy-par)
  - [Head Data CSV Files](#head-data-csv-files)
- [Configuration Reference](#configuration-reference)
  - [admin](#admin)
  - [layers](#layers)
  - [solver](#solver)
  - [hydrology](#hydrology)
  - [overburden](#overburden)
  - [compaction](#compaction)
  - [initial_stress](#initial_stress)
  - [output](#output)
  - [Resume Mode](#resume-mode)
- [Solvers](#solvers)
- [Output Files](#output-files)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [License](#license)

---

## What It Does

SUB1D models land subsidence caused by groundwater extraction in multi-layered aquifer systems. The simulation proceeds in two coupled stages:

1. **Head Equation Solving** -- For each clay layer (aquitards and aquifer interbeds), SUB1D solves the 1D vertical diffusion equation for hydraulic head:

   ```
   dh/dt = K_v / S_s * d²h/dz²
   ```

   where `h` is hydraulic head, `K_v` is vertical hydraulic conductivity, and `S_s` is specific storage. Aquifer head timeseries measured at wells serve as boundary conditions at the top and bottom of each clay unit.

2. **Compaction Calculation** -- From the solved head field, effective stress is computed at each depth and timestep. Compaction is accumulated using elastic (recoverable) and inelastic (permanent, virgin compression) specific storage coefficients, tracking the preconsolidation stress history at every node:

   - **Elastic regime**: effective stress below historical maximum (preconsolidation) -- uses `S_ske`
   - **Inelastic regime**: effective stress exceeds preconsolidation -- uses `S_skv` (typically 10x larger)

   Total deformation is the depth-integrated sum across all clay layers and interbeds.

## Physical Assumptions

- **1D vertical flow**: Lateral flow within clay layers is negligible; head changes propagate vertically through clays driven by boundary conditions from adjacent aquifers.
- **Terzaghi effective stress**: Effective stress equals total stress minus pore-water pressure. The skeletal (grain-to-grain) stress drives compaction.
- **Linear storage**: Elastic and inelastic specific storage coefficients are constant within each regime.
- **Instantaneous aquifer response**: Aquifer heads are prescribed from field measurements (no flow equation solved in the aquifers themselves).
- **Aquitard boundaries**: The top and bottom of each aquitard are pinned to the head in the adjacent aquifer. Interbeds within an aquifer use the aquifer head on both faces.
- **Small-strain**: Compaction is small relative to layer thickness (no geometry update, though time-varying thickness can be specified via step changes).
- **Optional overburden stress**: Water-level changes in the uppermost (unconfined) aquifer can drive overburden stress changes via `S_y * rho_w * g * delta_h`.

---

## Installation

**Requirements**: Python 3.10+

### From source (recommended)

```bash
cd SUB1D
pip install -e ".[all]"
```

This installs the package in editable mode with all optional dependencies (PyYAML, netCDF4, Seaborn, Numba).

### Minimal install

```bash
pip install -e .
```

Core dependencies (installed automatically): NumPy >= 1.22, SciPy >= 1.8, Pandas >= 1.4, Matplotlib >= 3.5.

### Optional dependency groups

| Group    | Install command          | Provides                               |
|----------|--------------------------|----------------------------------------|
| `yaml`   | `pip install -e ".[yaml]"`   | YAML configuration file support (PyYAML) |
| `netcdf` | `pip install -e ".[netcdf]"` | NetCDF output via GMT conversion       |
| `numba`  | `pip install -e ".[numba]"`  | JIT-compiled solvers (50-400x speedup) |
| `dev`    | `pip install -e ".[dev]"`    | Testing tools (pytest, pytest-cov)     |
| `all`    | `pip install -e ".[all]"`    | All of the above                       |

### Verify installation

```bash
sub1d --help
python -m pytest tests/ -v
```

---

## Quick Start

1. **Prepare head data** -- Create a CSV file for each aquifer with columns for date and head elevation (see [Head Data CSV Files](#head-data-csv-files)).

2. **Create a configuration file** -- Copy an example from `examples/` and edit to match your site:

   ```bash
   cp examples/confined_hanford.yaml my_site.yaml
   # Edit my_site.yaml with your layer geometry, parameters, and file paths
   ```

3. **Run the model**:

   ```bash
   sub1d my_site.yaml
   ```

4. **View results** -- Output files are written to the directory specified by `admin.output_folder` in your config. The primary result is a CSV of total deformation vs. time for each layer, plus an aggregate total.

---

## Command-Line Interface

```
usage: sub1d [-h] [--overwrite] [--solver {explicit,crank-nicolson}]
             [-v] [--no-compaction] config
```

**Positional argument:**

| Argument | Description |
|----------|-------------|
| `config` | Path to configuration file (`.yaml`, `.yml`, or `.par`) |

**Options:**

| Flag                    | Description                                              |
|-------------------------|----------------------------------------------------------|
| `--overwrite`           | Overwrite existing output directory if it exists         |
| `--solver {explicit,crank-nicolson}` | Override solver type from command line       |
| `-v`, `--verbose`       | Enable DEBUG-level logging                               |
| `--no-compaction`       | Skip compaction -- solve head equations only             |

**Examples:**

```bash
# Basic run
sub1d config.yaml

# Use Crank-Nicolson solver with verbose logging
sub1d config.yaml --solver crank-nicolson -v

# Solve only head equations, overwrite previous output
sub1d config.yaml --no-compaction --overwrite

# Run with a legacy parameter file
sub1d paramfile.par
```

You can also invoke the model from Python:

```python
from sub1d import run_model
from sub1d.config import load_yaml_config

config = load_yaml_config("my_site.yaml")
results = run_model(config)
```

---

## Input Files

### Configuration File (YAML)

The primary configuration format is YAML. Two annotated examples are provided in `examples/`:

- `confined_hanford.yaml` -- Confined aquifer system (South Hanford, 68 clay layers)
- `unconfined_lindsay.yaml` -- Unconfined aquifer system (Lindsay, 5 clay layers)

Minimal example:

```yaml
admin:
  run_name: my_run
  output_folder: Output/

mode: Normal

layers:
  - name: Upper Aquifer
    type: Aquifer
    thickness: 50
    compaction_switch: true
    interbeds_switch: true
    interbeds_distributions:
      3: 2    # 2 interbeds, each 3 m thick

  - name: Clay Layer
    type: Aquitard
    thickness: 25
    compaction_switch: true

  - name: Lower Aquifer
    type: Aquifer
    thickness: 100
    compaction_switch: false

head_data_files:
  Upper Aquifer: upper_aquifer_heads.csv
  Lower Aquifer: lower_aquifer_heads.csv

solver:
  dt_master:
    Upper Aquifer: 0.5
    Clay Layer: 1.0
  dz_clays:
    Upper Aquifer: 0.2
    Clay Layer: 0.5
  groundwater_flow_solver_type:
    Upper Aquifer: elastic-inelastic
    Clay Layer: elastic-inelastic
  vertical_conductivity:
    Upper Aquifer: 1.0e-6
    Clay Layer: 1.0e-6

hydrology:
  clay_Sse:
    Upper Aquifer: 1.2e-4
    Clay Layer: 1.2e-4
  clay_Ssv:
    Upper Aquifer: 1.0e-3
    Clay Layer: 1.0e-3
  sand_Sse:
    Upper Aquifer: 3.3e-5
  compressibility_of_water: 1.5e-6
  rho_w: 1000
  g: 9.81

time_unit: days
```

### Configuration File (Legacy .par)

SUB1D also reads the legacy parameter-file format (`.par`) for backward compatibility. These are plain-text key-value files:

```
run_name = my_run
output_folder = Output/
layer_names = Upper Aquifer,Clay Layer,Lower Aquifer
layer_types = Upper Aquifer:Aquifer,Clay Layer:Aquitard,Lower Aquifer:Aquifer
layer_thicknesses = Upper Aquifer:50,Clay Layer:25,Lower Aquifer:100
dt_master = Upper Aquifer:0.5,Clay Layer:1.0
dz_clays = Upper Aquifer:0.2,Clay Layer:0.5
```

Lines beginning with `#` are comments. See the YAML examples for the full set of parameters.

### Head Data CSV Files

Each aquifer that provides a boundary condition for adjacent clay layers needs a head timeseries CSV file. The path to each file is specified in the configuration under `head_data_files`.

**Format requirements:**

| Column | Content | Notes |
|--------|---------|-------|
| 1      | Date    | Any format parseable by `pandas.read_csv(parse_dates=[0])`. Recommended: `YYYY-MM-DD` |
| 2      | Head    | Hydraulic head elevation in meters. Numeric values. |

**Example** (`upper_aquifer_heads.csv`):

```csv
date,head
1920-01-01,50.0
1920-02-01,49.8
1920-03-01,49.5
1920-04-01,49.1
...
2020-12-01,35.2
```

**Validation**: On loading, each CSV is automatically checked for:

- NaN or Inf values (interpolated with a warning)
- Duplicate dates (first occurrence kept, duplicates dropped)
- Non-monotonic dates (sorted with a warning)
- Large temporal gaps > 30 days (warning only)
- Extreme head values |head| > 1000 m (warning only)

**Multiple aquifers**: When multiple aquifer timeseries have different date ranges, SUB1D automatically clips to the overlapping window (latest start date to earliest end date).

---

## Configuration Reference

### admin

| Field               | Type   | Default | Description                           |
|---------------------|--------|---------|---------------------------------------|
| `run_name`          | string | --      | **Required.** Name for this simulation run |
| `output_folder`     | string | --      | **Required.** Root directory for output files |
| `overwrite`         | bool   | `false` | Overwrite output directory if it exists |

### layers

A list of layer definitions, ordered from top (shallowest) to bottom (deepest). The geological model alternates between aquifers and aquitards.

| Field                     | Type        | Default      | Description |
|---------------------------|-------------|--------------|-------------|
| `name`                    | string      | --           | **Required.** Unique layer name |
| `type`                    | string      | --           | **Required.** `"Aquifer"` or `"Aquitard"` |
| `thickness`               | float/dict  | --           | **Required.** Constant value (meters) or dict of step changes |
| `thickness_type`          | string      | `"constant"` | `"constant"` or `"step_changes"` |
| `compaction_switch`       | bool        | `true`       | Enable compaction calculation for this layer |
| `interbeds_switch`        | bool        | `false`      | Layer contains clay interbeds (aquifers only) |
| `interbeds_distributions` | dict        | --           | Mapping of interbed thickness (m) to count |

**Step-change thickness example:**

```yaml
thickness_type: step_changes
thickness:
  pre-1977: 121        # Before 1977
  1977-1984: 148       # 1977 through 1984
  1984-1994: 319       # 1984 through 1994
  1994-: 340           # 1994 onward
```

**Interbeds example:**

```yaml
interbeds_switch: true
interbeds_distributions:
  3: 3     # Three 3-meter-thick interbeds
  5: 1     # One 5-meter-thick interbed
  9: 1     # One 9-meter-thick interbed
```

### solver

Per-layer solver parameters. Every layer that contains clay (aquitards + interbedded aquifers) must have entries in these dictionaries.

| Field                          | Type       | Default       | Description |
|--------------------------------|------------|---------------|-------------|
| `dt_master`                    | dict       | --            | **Required.** Timestep size in days, per layer |
| `dz_clays`                     | dict       | --            | **Required.** Vertical grid spacing in meters, per layer |
| `groundwater_flow_solver_type` | dict       | --            | **Required.** `"singlevalue"` or `"elastic-inelastic"`, per layer |
| `vertical_conductivity`        | dict       | --            | **Required.** Vertical K in m/day, per layer |
| `default_solver`               | string     | `"explicit"`  | `"explicit"` (FTCS) or `"crank-nicolson"` (implicit) |
| `parallel_layers`              | bool       | `true`        | Solve independent layers in parallel |
| `smoothing_width`              | float      | `0.0`         | Sigmoid blend width for elastic/inelastic transition (0 = sharp) |
| `mass_balance_check`           | bool       | `false`       | Periodically verify solver mass balance |
| `mass_balance_threshold`       | float      | `1e-6`        | Mass balance warning threshold |
| `adaptive_timestepping`        | bool       | `false`       | Enable adaptive timestep control (experimental) |

**Solver type guidance:**

- `"singlevalue"` -- Constant diffusivity. Appropriate when elastic and inelastic storage are similar, or for quick runs.
- `"elastic-inelastic"` -- Per-node diffusivity switching based on preconsolidation stress history. Required to capture irreversible compaction behavior.

**Explicit vs. Crank-Nicolson:**

- `"explicit"` (FTCS) -- Fast per step, but requires CFL condition: `K_v / S_s * dt / dz² < 0.5`. The model checks this automatically and raises an error if violated.
- `"crank-nicolson"` -- Unconditionally stable (no CFL restriction), allowing larger timesteps. Slightly more computation per step (tridiagonal solve), but often faster overall because `dt` is not constrained.

### hydrology

| Field                       | Type  | Default | Description |
|-----------------------------|-------|---------|-------------|
| `clay_Sse`                  | dict  | --      | **Required.** Elastic skeletal specific storage (1/m), per layer |
| `clay_Ssv`                  | dict  | --      | **Required.** Inelastic skeletal specific storage (1/m), per layer |
| `sand_Sse`                  | dict  | --      | Elastic specific storage for aquifer sand matrix (1/m), per aquifer |
| `compressibility_of_water`  | float | --      | **Required.** Compressibility of water (1/m), typically `1.5e-6` |
| `rho_w`                     | float | `1000`  | Density of water (kg/m^3) |
| `g`                         | float | `9.81`  | Gravitational acceleration (m/s^2) |
| `specific_yield`            | float | --      | Specific yield of unconfined aquifer (required if overburden is enabled) |

**Typical parameter ranges:**

| Parameter | Typical Range | Units |
|-----------|--------------|-------|
| `clay_Sse` | 1e-5 to 5e-4 | 1/m |
| `clay_Ssv` | 5e-4 to 5e-3 | 1/m |
| `sand_Sse` | 1e-6 to 1e-4 | 1/m |
| `vertical_conductivity` | 1e-8 to 1e-5 | m/day |
| `specific_yield` | 0.05 to 0.30 | dimensionless |

### overburden

Optional section to include overburden stress effects from water-table fluctuations in the shallowest aquifer.

| Field               | Type | Default | Description |
|---------------------|------|---------|-------------|
| `stress_gwflow`     | bool | `false` | Include overburden in head equation |
| `stress_compaction` | bool | `false` | Include overburden in compaction calculation |

When enabled, overburden stress change is computed as:

```
delta_sigma_v = S_y * rho_w * g * (h - h_0)
```

where `h_0` is the initial head in the uppermost aquifer. Requires `specific_yield` to be set in the `hydrology` section.

### compaction

| Field                           | Type       | Default     | Description |
|---------------------------------|------------|-------------|-------------|
| `solver_compressibility_type`   | dict/string | `{}`       | Per-layer compressibility type |
| `debug_include_endnodes`        | bool       | `false`     | Use node values instead of midpoint interpolation |
| `preconsolidation_head_type`    | string     | `"initial"` | `"initial"` or `"initial_plus_offset"` |

### initial_stress

Optional section for specifying how initial effective stress and preconsolidation are set.

```yaml
initial_stress:
  type:
    Upper Aquifer: initial_equilibrium
    Clay Layer: initial_equilibrium
  offset:
    Upper Aquifer: 0
  offset_unit: head     # "head" or "stress"
```

### output

| Field                            | Type      | Default | Description |
|----------------------------------|-----------|---------|-------------|
| `save_output_head_timeseries`    | bool      | `false` | Save head solution arrays |
| `save_effective_stress`          | bool      | `false` | Save effective stress arrays |
| `save_internal_compaction`       | bool      | `false` | Save internal compaction arrays |
| `save_s`                         | bool      | `false` | Save raw deformation arrays |
| `create_output_head_video`       | bool/dict | `false` | Create head-profile video animations |

Large arrays (> 1 million elements) are automatically saved in binary format (float32) rather than CSV for performance. Smaller arrays are saved as CSV.

### Resume Mode

SUB1D supports resuming a simulation from a specific date, using stored state from a previous run:

```yaml
mode: resume
resume_directory: Output/previous_run/
resume_date: 2000-01-01
resume_head_value:
  Upper Aquifer: cst     # Freeze head at first value
  Lower Aquifer: normal  # Continue with new head data
```

---

## Solvers

SUB1D provides four head-equation solvers and a compaction solver:

### Head Equation Solvers

| Solver | Method | Stability | Best For |
|--------|--------|-----------|----------|
| Explicit FTCS, single diffusivity | Forward-Time Central-Space | CFL < 0.5 | Quick runs with uniform storage |
| Explicit FTCS, elastic-inelastic | FTCS with per-node switching | CFL < 0.5 | Full elastic-inelastic simulation |
| Crank-Nicolson, single diffusivity | Implicit tridiagonal | Unconditionally stable | Large timesteps, stiff problems |
| Crank-Nicolson, elastic-inelastic | Implicit with per-node switching | Unconditionally stable | Production runs |

**CFL condition** (explicit solvers only):

```
CFL = (K_v / S_s) * dt / dz^2 < 0.5
```

If violated, the model raises a `SolverError`. Either reduce `dt_master` or increase `dz_clays`, or switch to Crank-Nicolson.

### Numba Acceleration

When [Numba](https://numba.pydata.org/) is installed, the explicit solvers use JIT-compiled inner loops for 50-400x speedup over pure Python. No configuration is needed -- Numba is detected automatically. The implicit (Crank-Nicolson) solvers rely on SciPy's banded solver and are already fast.

### Adaptive Timestepping (Experimental)

An adaptive FTCS solver is available that uses Richardson extrapolation to control the local error. It automatically halves or doubles the timestep to meet a user-specified tolerance. This is accessed programmatically via:

```python
from sub1d.solver import solve_head_equation_adaptive

t_out, hmat = solve_head_equation_adaptive(
    bc_func=lambda t: (h_top(t), h_bot(t)),
    t_start=0.0, t_end=100.0,
    dt_init=0.1, dx=0.5,
    x=np.linspace(0, 10, 21),
    ic=10.0, k=0.01, tol=1e-4,
)
```

### Diagnostics

The `sub1d.diagnostics` module provides tools for solution verification:

- **Mass balance check** -- Compares boundary flux against internal storage change at each timestep.
- **CN residual check** -- Verifies that the tridiagonal solve in Crank-Nicolson is accurate.
- **Richardson extrapolation** -- Estimates spatial discretization error by comparing solutions at `dz` and `dz/2`.

Enable automatic mass-balance checking by setting `solver.mass_balance_check: true` in the config.

---

## Output Files

All output is written to the directory specified by `admin.output_folder`, organized into subdirectories:

```
Output/
  my_run/
    input_data/               # Copy of input CSV files
    gwflow_head_outputs/      # Head solution arrays (if enabled)
    s_outputs/                # Compaction/deformation files
      Upper Aquifer_Total_Deformation_Out.csv
      Clay Layer_Total_Deformation_Out.csv
      Lower Aquifer_Total_Deformation_Out.csv
    plots/                    # Input data visualizations
```

### Deformation Output CSV

The primary result file for each layer is `{layer}_Total_Deformation_Out.csv`. For aquifers with interbeds, this contains columns for each component:

| Column                | Description |
|-----------------------|-------------|
| `dates`               | Timestep dates |
| `Interconnected matrix` | Sand skeleton elastic deformation |
| `total_3.00 clays`   | Total deformation from 3 m interbeds |
| `total_5.00 clays`   | Total deformation from 5 m interbeds |
| `total`               | Sum of all components |

For aquitards, the file has `dates` and `total` columns.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=sub1d

# Run a specific test module
python -m pytest tests/test_solver.py -v
```

The test suite includes 74 tests covering:

- Configuration loading and validation (YAML and .par)
- All four head-equation solvers (convergence, CFL, boundary conditions)
- Compaction solver (elastic, inelastic, preconsolidation tracking)
- Head data I/O (reading, clipping, interpolation, validation)
- Stratigraphy domain model
- Diagnostics (mass balance, CN residual, Richardson extrapolation)
- Integration tests (full model run)
- Golden regression tests (match legacy output)

---

## Project Structure

```
SUB1D/
  sub1d/                    # Main package
    __init__.py             # Public API and version
    cli.py                  # Command-line interface
    config.py               # Configuration loading (YAML + .par)
    model.py                # Simulation orchestrator
    solver.py               # Head equation solvers (FTCS, CN, adaptive)
    compaction.py           # Compaction calculation
    layers.py               # Stratigraphy domain model
    head_io.py              # Head data I/O and validation
    output.py               # File output (CSV, binary, GMT)
    diagnostics.py          # Mass balance, CN residual, Richardson
    exceptions.py           # Custom exception classes
    utils.py                # Logging setup utilities
  tests/                    # Test suite (pytest)
  examples/                 # Example YAML configuration files
  legacy/                   # Original monolithic code (reference only)
  benchmarks/               # Performance benchmarks
  pyproject.toml            # Package metadata and dependencies
```

---

## License

MIT
