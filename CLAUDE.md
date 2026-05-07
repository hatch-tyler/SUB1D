# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e ".[dev]"                          # Install with dev/test dependencies
pip install -e ".[all]"                          # Install with all optional deps (numba, netcdf, seaborn, yaml)
python -m pytest tests/ -v                       # Run all 74 tests
python -m pytest tests/ -v --cov=sub1d           # Run tests with coverage
python -m pytest tests/test_solver.py -v         # Run a single test module
python -m pytest tests/test_solver.py::test_name # Run a single test
```

## CLI Usage

```bash
sub1d config.yaml                                # Run simulation
sub1d config.yaml --solver crank-nicolson -v     # Override solver, verbose
sub1d config.yaml --no-compaction --overwrite     # Head-only, overwrite output
sub1d paramfile.par                              # Legacy .par config format
```

## Architecture

**Simulation pipeline** (each step feeds the next):

1. **config.py** — Loads YAML or legacy .par into `ModelConfig` dataclass. Two independent parsers: `load_yaml_config()` and `load_par_file()`.
2. **layers.py** — `Stratigraphy` class: single source of truth for layer relationships, types (Aquifer/Aquitard), thicknesses, and interbed distributions.
3. **head_io.py** — Reads CSV head data, validates, clips to simulation window, interpolates. `_validate_head_data()` enforces input constraints.
4. **solver.py** — Four head-equation solvers in two families:
   - FTCS explicit (CFL-constrained) and Crank-Nicolson implicit (unconditionally stable)
   - Each has a single-value and elastic-inelastic variant
   - Optional Numba JIT via `HAS_NUMBA` flag (50-400x speedup)
   - `solve_compaction_elastic_inelastic()` also lives here (sparse matrix interp)
5. **compaction.py** — Per-layer compaction solving, interconnected matrix deformation, clay interbed aggregation. Orchestrates calls to solver.py.
6. **model.py** — `run_model()` orchestrator. Optionally solves layers in parallel via `ThreadPoolExecutor`.
7. **diagnostics.py** — Post-solve verification: mass balance, CN residual, Richardson extrapolation.
8. **output.py** — Writes CSV/binary results to structured output directory.
9. **plotting.py** — Matplotlib visualization of heads, deformation, clay distributions.
10. **cli.py** — argparse entry point (`main()`), registered as `sub1d` console script.

**Exception hierarchy** in `exceptions.py`: `SUB1DError` → `ConfigurationError`, `SolverError`, `InputDataError`.

## Key Design Patterns

- **Dual config formats**: YAML (primary) and .par (legacy compat) with completely separate parsing paths that produce the same `ModelConfig`.
- **Numba is optional**: All JIT-decorated solvers have pure-NumPy fallbacks guarded by `HAS_NUMBA`.
- **Parallel layers**: `model.py` uses `ThreadPoolExecutor` when `parallel_layers: true` in config; layers must be independent.
- **Sparse compaction interpolation**: `solver.py` uses `scipy.sparse` for the EI compaction matrix (7-12x over dense).
- **F-order arrays**: Fortran memory layout used in solver hot paths for column-major access patterns.
- **Sigmoid EI blend**: Smooth elastic-inelastic transition instead of hard switch at preconsolidation stress.
- **Resume mode**: Simulations can checkpoint and resume from a date via `mode: resume` config.
