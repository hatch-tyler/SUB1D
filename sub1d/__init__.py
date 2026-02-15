"""SUB1D: 1D Land Subsidence / Compaction Model.

A modular, tested scientific package for solving coupled groundwater flow
and elastic-inelastic compaction equations in 1D.

Main entry points
-----------------
- :func:`sub1d.model.run_model` — Run the full simulation pipeline.
- :func:`sub1d.cli.main` — Command-line interface.
- :func:`sub1d.config.load_par_file` — Load a legacy .par configuration.
- :func:`sub1d.config.load_yaml_config` — Load a YAML configuration.
"""

__version__ = "2.0.0"

from sub1d.model import run_model
from sub1d.config import ModelConfig, load_par_file
from sub1d.exceptions import SUB1DError, ConfigurationError, SolverError, InputDataError

__all__ = [
    "run_model",
    "ModelConfig",
    "load_par_file",
    "SUB1DError",
    "ConfigurationError",
    "SolverError",
    "InputDataError",
]
