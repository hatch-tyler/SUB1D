"""Custom exceptions for the SUB1D subsidence model."""


class SUB1DError(Exception):
    """Base exception for all SUB1D errors."""


class ConfigurationError(SUB1DError):
    """Raised for invalid or missing configuration parameters."""


class SolverError(SUB1DError):
    """Raised for solver failures: CFL violations, non-convergence, etc."""


class InputDataError(SUB1DError):
    """Raised for missing or malformed input data files."""
