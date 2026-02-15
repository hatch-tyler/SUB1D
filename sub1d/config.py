"""Configuration module for the SUB1D land subsidence model.

Defines structured dataclasses to replace the legacy 38-element tuple
returned by parameters.py, along with loaders for both YAML and legacy
.par parameter files.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sub1d.exceptions import ConfigurationError

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: strtobool replacement (distutils.util.strtobool removed in 3.12)
# ---------------------------------------------------------------------------

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def _str_to_bool(val: str | bool | int) -> bool:
    """Convert a string representation of truth to ``True`` or ``False``.

    Replaces the deprecated ``distutils.util.strtobool`` that was removed
    in Python 3.12.

    Accepted true  values: ``"true"``, ``"1"``, ``"yes"``, ``"on"``
    Accepted false values: ``"false"``, ``"0"``, ``"no"``, ``"off"``
    (all comparisons are case-insensitive)

    Parameters
    ----------
    val : str | bool | int
        The value to convert.  If already a ``bool`` or ``int`` it is
        returned directly as ``bool(val)``.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If *val* is a string that cannot be interpreted as a boolean.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)

    lowered = str(val).strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Cannot interpret {val!r} as a boolean.  "
        f"Accepted true values: {sorted(_TRUE_STRINGS)}, "
        f"accepted false values: {sorted(_FALSE_STRINGS)}."
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdminConfig:
    """Administrative / run-management settings."""

    run_name: str
    output_folder: str
    overwrite: bool = False
    internal_time_delay: float = 0.5


@dataclass
class LayerConfig:
    """Description of a single hydrostratigraphic layer."""

    name: str
    layer_type: str  # "Aquifer" or "Aquitard"
    thickness: Any  # float or dict (for step_changes)
    compaction_switch: bool = True
    thickness_type: str = "constant"
    interbeds_switch: bool = False
    interbeds_distributions: Optional[dict] = field(default=None)

    def __post_init__(self) -> None:
        if self.layer_type not in ("Aquifer", "Aquitard"):
            raise ConfigurationError(
                f"Layer {self.name!r}: layer_type must be 'Aquifer' or "
                f"'Aquitard', got {self.layer_type!r}."
            )
        if self.thickness_type not in ("constant", "step_changes"):
            raise ConfigurationError(
                f"Layer {self.name!r}: thickness_type must be 'constant' or "
                f"'step_changes', got {self.thickness_type!r}."
            )


@dataclass
class SolverConfig:
    """Solver discretisation and method parameters."""

    dt_master: dict  # {layer_name: float}
    dz_clays: dict  # {layer_name: float}
    groundwater_flow_solver_type: dict  # {layer_name: str}
    vertical_conductivity: dict  # {layer_name: float}

    # --- Performance & robustness options (Phase 4) ---
    default_solver: str = "explicit"          # "explicit" or "crank-nicolson"
    parallel_layers: bool = True
    smoothing_width: float = 0.0              # sigmoid blend width (0 = sharp switch)
    adaptive_timestepping: bool = False
    mass_balance_check: bool = False
    mass_balance_threshold: float = 1e-6


@dataclass
class HydrologicParams:
    """Physical / hydrologic material properties."""

    clay_Sse: dict  # {layer_name: float}
    clay_Ssv: dict  # {layer_name: float}
    clay_Ssk: Optional[dict]  # {layer_name: float} or None
    sand_Sse: dict  # {layer_name: float}
    sand_Ssk: Optional[dict]  # {layer_name: float} or None
    compressibility_of_water: float
    rho_w: float = 1000.0
    g: float = 9.81
    specific_yield: Optional[float] = None


@dataclass
class OutputConfig:
    """Flags controlling which outputs are written."""

    save_output_head_timeseries: bool = False
    save_effective_stress: bool = False
    save_internal_compaction: bool = False
    create_output_head_video: Union[dict, bool] = False
    save_s: bool = False


@dataclass
class ModelConfig:
    """Top-level configuration container for a SUB1D simulation.

    This replaces the 38-element tuple previously returned by
    ``parameters.read_parameters_noadmin``.
    """

    admin: AdminConfig
    layers: List[LayerConfig]
    layer_names: List[str]
    layer_types: dict  # {layer_name: "Aquifer"|"Aquitard"}
    solver: SolverConfig
    hydro: HydrologicParams
    output: OutputConfig

    overburden_stress_gwflow: bool = False
    overburden_stress_compaction: bool = False
    compaction_solver_compressibility_type: Union[dict, str] = field(
        default_factory=dict
    )
    compaction_solver_debug_include_endnodes: bool = False
    preconsolidation_head_type: str = "initial"
    preconsolidation_head_offset: Any = None  # dict or False/None
    time_unit: str = "days"
    mode: str = "Normal"
    head_data_files: dict = field(default_factory=dict)  # {layer_name: path}

    initial_stress_type: Optional[dict] = None
    initial_stress_offset: Optional[dict] = None
    initial_stress_offset_unit: str = "head"

    resume_directory: Optional[str] = None
    resume_date: Optional[str] = None
    resume_head_value: Optional[dict] = None

    # ---- derived helpers (not stored as fields) ----

    @property
    def no_layers(self) -> int:
        return len(self.layer_names)

    @property
    def no_aquifers(self) -> int:
        return sum(1 for v in self.layer_types.values() if v == "Aquifer")

    @property
    def no_aquitards(self) -> int:
        return sum(1 for v in self.layer_types.values() if v == "Aquitard")

    @property
    def aquitards(self) -> list[str]:
        return [n for n, t in self.layer_types.items() if t == "Aquitard"]

    @property
    def interbedded_layers(self) -> list[str]:
        return [
            lyr.name for lyr in self.layers
            if lyr.layer_type == "Aquifer" and lyr.interbeds_switch
        ]

    @property
    def layers_requiring_solving(self) -> list[str]:
        return self.interbedded_layers + self.aquitards

    @property
    def no_layers_containing_clay(self) -> int:
        return len(self.layers_requiring_solving)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(config: ModelConfig) -> None:
    """Check physical constraints on a :class:`ModelConfig`.

    Raises
    ------
    ConfigurationError
        If any constraint is violated.
    """
    errors: list[str] = []

    # -- Layer consistency ---------------------------------------------------
    declared_names = set(config.layer_names)
    layer_obj_names = {lyr.name for lyr in config.layers}
    if declared_names != layer_obj_names:
        errors.append(
            f"layer_names ({sorted(declared_names)}) do not match "
            f"LayerConfig names ({sorted(layer_obj_names)})."
        )

    type_names = set(config.layer_types.keys())
    if declared_names != type_names:
        errors.append(
            f"layer_types keys ({sorted(type_names)}) do not match "
            f"layer_names ({sorted(declared_names)})."
        )

    # -- Positive thickness --------------------------------------------------
    for lyr in config.layers:
        th = lyr.thickness
        if isinstance(th, dict):
            for period, val in th.items():
                if val <= 0:
                    errors.append(
                        f"Layer {lyr.name!r}, period {period!r}: "
                        f"thickness must be > 0, got {val}."
                    )
        else:
            try:
                if float(th) <= 0:
                    errors.append(
                        f"Layer {lyr.name!r}: thickness must be > 0, got {th}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Layer {lyr.name!r}: thickness is not numeric ({th!r})."
                )

    # -- Positive conductivity -----------------------------------------------
    for name, kv in config.solver.vertical_conductivity.items():
        try:
            if float(kv) <= 0:
                errors.append(
                    f"vertical_conductivity for {name!r} must be > 0, got {kv}."
                )
        except (TypeError, ValueError):
            errors.append(
                f"vertical_conductivity for {name!r} is not numeric ({kv!r})."
            )

    # -- Solver type ---------------------------------------------------------
    valid_solver_types = {"singlevalue", "elastic-inelastic"}
    for name, stype in config.solver.groundwater_flow_solver_type.items():
        if stype not in valid_solver_types:
            errors.append(
                f"groundwater_flow_solver_type for {name!r} must be one of "
                f"{sorted(valid_solver_types)}, got {stype!r}."
            )

    # -- Positive dt and dz --------------------------------------------------
    for name, dt in config.solver.dt_master.items():
        try:
            if float(dt) <= 0:
                errors.append(
                    f"dt_master for {name!r} must be > 0, got {dt}."
                )
        except (TypeError, ValueError):
            errors.append(f"dt_master for {name!r} is not numeric ({dt!r}).")

    for name, dz in config.solver.dz_clays.items():
        try:
            if float(dz) <= 0:
                errors.append(
                    f"dz_clays for {name!r} must be > 0, got {dz}."
                )
        except (TypeError, ValueError):
            errors.append(f"dz_clays for {name!r} is not numeric ({dz!r}).")

    # -- Compressibility of water --------------------------------------------
    if config.hydro.compressibility_of_water <= 0:
        errors.append(
            "compressibility_of_water must be > 0, got "
            f"{config.hydro.compressibility_of_water}."
        )

    # -- Storage coefficients sign check (should be positive) ----------------
    for label, store_dict in [
        ("clay_Sse", config.hydro.clay_Sse),
        ("clay_Ssv", config.hydro.clay_Ssv),
    ]:
        if store_dict:
            for name, val in store_dict.items():
                try:
                    if float(val) <= 0:
                        errors.append(
                            f"{label} for {name!r} must be > 0, got {val}."
                        )
                except (TypeError, ValueError):
                    errors.append(
                        f"{label} for {name!r} is not numeric ({val!r})."
                    )

    # -- Solver options ------------------------------------------------------
    valid_default_solvers = {"explicit", "crank-nicolson"}
    if config.solver.default_solver not in valid_default_solvers:
        errors.append(
            f"default_solver must be one of {sorted(valid_default_solvers)}, "
            f"got {config.solver.default_solver!r}."
        )
    if config.solver.smoothing_width < 0:
        errors.append(
            f"smoothing_width must be >= 0, got {config.solver.smoothing_width}."
        )
    if config.solver.mass_balance_threshold <= 0:
        errors.append(
            f"mass_balance_threshold must be > 0, got {config.solver.mass_balance_threshold}."
        )

    # -- CFL pre-check (warning only) ----------------------------------------
    _cfl_precheck(config)

    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    logger.info("Configuration validation passed.")


def _cfl_precheck(config: ModelConfig) -> None:
    """Emit a logging warning if the CFL number looks large.

    CFL ~ K * dt / (Ss * dz^2).  A value >> 1 may cause numerical
    instability with explicit solvers.
    """
    for name in config.layers_requiring_solving:
        dt = config.solver.dt_master.get(name)
        dz = config.solver.dz_clays.get(name)
        kv = config.solver.vertical_conductivity.get(name)
        # Use inelastic (larger) Ss for worst-case
        ss = None
        if config.hydro.clay_Sse and name in config.hydro.clay_Sse:
            ss = config.hydro.clay_Sse[name]
        if config.hydro.clay_Ssk and name in config.hydro.clay_Ssk:
            ss = config.hydro.clay_Ssk[name]

        if dt is None or dz is None or kv is None or ss is None:
            continue

        try:
            dt_f = float(dt)
            dz_f = float(dz)
            kv_f = float(kv)
            ss_f = float(ss)
            if ss_f > 0 and dz_f > 0:
                cfl = kv_f * dt_f / (ss_f * dz_f * dz_f)
                if cfl > 1.0:
                    logger.warning(
                        "CFL pre-check for layer %r: CFL = %.3f (> 1). "
                        "Consider reducing dt_master or increasing dz_clays "
                        "for numerical stability.",
                        name,
                        cfl,
                    )
        except (TypeError, ValueError, ZeroDivisionError):
            pass


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_yaml_config(path: str | Path) -> ModelConfig:
    """Load a YAML configuration file and return a :class:`ModelConfig`.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    ModelConfig

    Raises
    ------
    ConfigurationError
        If YAML is not installed or the file is malformed.
    """
    if not _YAML_AVAILABLE:
        raise ConfigurationError(
            "The 'pyyaml' package is required to load YAML config files.  "
            "Install it with:  pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    logger.info("Loading YAML configuration from %s", path)

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Expected a YAML mapping at top level, got {type(raw).__name__}."
        )

    try:
        # --- admin ----------------------------------------------------------
        adm = raw["admin"]
        admin = AdminConfig(
            run_name=str(adm["run_name"]),
            output_folder=str(adm["output_folder"]),
            overwrite=_str_to_bool(adm.get("overwrite", False)),
            internal_time_delay=float(adm.get("internal_time_delay", 0.5)),
        )

        # --- layers ---------------------------------------------------------
        raw_layers = raw["layers"]
        layers: list[LayerConfig] = []
        layer_names: list[str] = []
        layer_types: dict[str, str] = {}

        for rl in raw_layers:
            lc = LayerConfig(
                name=str(rl["name"]),
                layer_type=str(rl["layer_type"]),
                thickness=rl["thickness"],
                compaction_switch=_str_to_bool(rl.get("compaction_switch", True)),
                thickness_type=str(rl.get("thickness_type", "constant")),
                interbeds_switch=_str_to_bool(rl.get("interbeds_switch", False)),
                interbeds_distributions=rl.get("interbeds_distributions"),
            )
            layers.append(lc)
            layer_names.append(lc.name)
            layer_types[lc.name] = lc.layer_type

        # --- solver ---------------------------------------------------------
        sv = raw["solver"]
        solver = SolverConfig(
            dt_master=sv["dt_master"],
            dz_clays=sv["dz_clays"],
            groundwater_flow_solver_type=sv["groundwater_flow_solver_type"],
            vertical_conductivity=sv["vertical_conductivity"],
            default_solver=str(sv.get("default_solver", "explicit")),
            parallel_layers=_str_to_bool(sv.get("parallel_layers", True)),
            smoothing_width=float(sv.get("smoothing_width", 0.0)),
            adaptive_timestepping=_str_to_bool(sv.get("adaptive_timestepping", False)),
            mass_balance_check=_str_to_bool(sv.get("mass_balance_check", False)),
            mass_balance_threshold=float(sv.get("mass_balance_threshold", 1e-6)),
        )

        # --- hydro ----------------------------------------------------------
        hy = raw["hydro"]
        hydro = HydrologicParams(
            clay_Sse=hy.get("clay_Sse", {}),
            clay_Ssv=hy.get("clay_Ssv", {}),
            clay_Ssk=hy.get("clay_Ssk"),
            sand_Sse=hy.get("sand_Sse", {}),
            sand_Ssk=hy.get("sand_Ssk"),
            compressibility_of_water=float(hy["compressibility_of_water"]),
            rho_w=float(hy.get("rho_w", 1000.0)),
            g=float(hy.get("g", 9.81)),
            specific_yield=(
                float(hy["specific_yield"]) if hy.get("specific_yield") is not None else None
            ),
        )

        # --- output ---------------------------------------------------------
        ou = raw.get("output", {})
        output = OutputConfig(
            save_output_head_timeseries=_str_to_bool(
                ou.get("save_output_head_timeseries", False)
            ),
            save_effective_stress=_str_to_bool(
                ou.get("save_effective_stress", False)
            ),
            save_internal_compaction=_str_to_bool(
                ou.get("save_internal_compaction", False)
            ),
            create_output_head_video=ou.get("create_output_head_video", False),
            save_s=_str_to_bool(ou.get("save_s", False)),
        )

        # --- top-level extras -----------------------------------------------
        config = ModelConfig(
            admin=admin,
            layers=layers,
            layer_names=layer_names,
            layer_types=layer_types,
            solver=solver,
            hydro=hydro,
            output=output,
            overburden_stress_gwflow=_str_to_bool(
                raw.get("overburden_stress_gwflow", False)
            ),
            overburden_stress_compaction=_str_to_bool(
                raw.get("overburden_stress_compaction", False)
            ),
            compaction_solver_compressibility_type=raw.get(
                "compaction_solver_compressibility_type", {}
            ),
            compaction_solver_debug_include_endnodes=_str_to_bool(
                raw.get("compaction_solver_debug_include_endnodes", False)
            ),
            preconsolidation_head_type=str(
                raw.get("preconsolidation_head_type", "initial")
            ),
            preconsolidation_head_offset=raw.get("preconsolidation_head_offset"),
            time_unit=str(raw.get("time_unit", "days")),
            mode=str(raw.get("mode", "Normal")),
            head_data_files=raw.get("head_data_files", {}),
            initial_stress_type=raw.get("initial_stress_type"),
            initial_stress_offset=raw.get("initial_stress_offset"),
            initial_stress_offset_unit=str(
                raw.get("initial_stress_offset_unit", "head")
            ),
            resume_directory=raw.get("resume_directory"),
            resume_date=raw.get("resume_date"),
            resume_head_value=raw.get("resume_head_value"),
        )

    except KeyError as exc:
        raise ConfigurationError(
            f"Missing required key in YAML configuration: {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Invalid value in YAML configuration: {exc}"
        ) from exc

    validate_config(config)
    return config


# ---------------------------------------------------------------------------
# Legacy .par file loader  (mirrors logic from parameters.py)
# ---------------------------------------------------------------------------

# Default values matching the original parameters.py DEFAULTS_DICTIONARY /
# DEFAULT_VALUES.
_PAR_DEFAULTS: dict[str, Any] = {
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

# Parameters that are allowed to be absent (no error if missing).
_PAR_HAS_DEFAULT: dict[str, bool] = {
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
    "default_solver": True,
    "parallel_layers": True,
    "smoothing_width": True,
    "adaptive_timestepping": True,
    "mass_balance_check": True,
    "mass_balance_threshold": True,
}


def _parse_par_lines(path: str | Path) -> list[str]:
    """Read a .par file and return non-empty, non-comment lines."""
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Parameter file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    lines = [line.strip() for line in lines]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return lines


def _read_par_value(
    name: str,
    lines: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the raw string value for *name* from the parsed lines.

    Returns ``None`` when the parameter is absent and *required* is False.
    Raises :class:`ConfigurationError` when absent and *required* is True
    and no default exists.
    """
    matches = [
        ln for ln in lines if ln.replace(" ", "").startswith(f"{name}=")
    ]
    if len(matches) == 0:
        if _PAR_HAS_DEFAULT.get(name, False):
            logger.debug(
                "Parameter %r not in file; using default %r.",
                name,
                _PAR_DEFAULTS.get(name),
            )
            return None  # caller should use default
        if not required:
            return None
        raise ConfigurationError(
            f"Required parameter {name!r} not found in parameter file."
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple entries for %r found; using the first.", name
        )
    # Strip inline comments and take the value after '='
    raw = matches[0].split("#")[0].split("=", 1)[1].strip()
    return raw


def _parse_simple(raw: str | None, typ: type, name: str, default: Any = None) -> Any:
    """Parse a single-value parameter from its raw string."""
    if raw is None:
        return default if default is not None else _PAR_DEFAULTS.get(name)
    if typ is bool:
        return _str_to_bool(raw)
    return typ(raw)


def _parse_dict_flat(
    raw: str | None,
    val_type: type,
    name: str,
    default: Any = None,
) -> Any:
    """Parse ``key1:val1,key2:val2`` into a dict.

    *val_type* is applied to each value (``float``, ``str``, ``bool``).
    """
    if raw is None:
        return default if default is not None else _PAR_DEFAULTS.get(name)

    if ":" not in raw:
        # Scalar, not a dict
        if val_type is bool:
            return _str_to_bool(raw)
        return val_type(raw)

    pairs = re.split(r",(?![^{]*})", raw)  # split on commas not inside braces
    result: dict[str, Any] = {}
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        key, _, val = pair.partition(":")
        key = key.strip()
        val = val.strip()
        if val_type is bool:
            result[key] = _str_to_bool(val)
        else:
            result[key] = val_type(val)
    return result


def _parse_list(raw: str | None, name: str, default: Any = None) -> list[str]:
    """Parse ``val1,val2,val3`` into a list of stripped strings."""
    if raw is None:
        d = default if default is not None else _PAR_DEFAULTS.get(name)
        if isinstance(d, list):
            return list(d)
        return [d] if d is not None else []
    return [v.strip() for v in raw.split(",")]


def _parse_layer_thicknesses(raw: str | None, name: str) -> dict:
    """Parse the complex ``layer_thicknesses`` parameter.

    Handles both simple ``Layer:val`` and variable-thickness
    ``Layer:{pre-YEAR:T0,YEAR1-YEAR2:T1,...}`` formats.
    """
    if raw is None:
        return dict(_PAR_DEFAULTS.get(name, {}))

    result: dict[str, Any] = {}
    # Split on commas that are NOT inside braces
    tokens = re.split(r",(?![^{]*})", raw)

    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if not token:
            i += 1
            continue

        if ":" not in token:
            i += 1
            continue

        # Check if this token contains a brace-delimited sub-dict
        if "{" in token:
            # Format: LayerName:{key1:val1,key2:val2,...}
            key, _, rest = token.partition(":")
            key = key.strip()
            inner = rest.strip().strip("{}")
            sub_pairs = inner.split(",")
            sub_dict: dict[str, float] = {}
            for sp in sub_pairs:
                sp = sp.strip()
                if ":" in sp:
                    sk, _, sv = sp.partition(":")
                    sub_dict[sk.strip()] = float(sv.strip())
            result[key] = sub_dict
        else:
            # Format: LayerName:value
            key, _, val = token.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val

        i += 1

    return result


def _parse_interbeds_distributions(raw: str | None) -> dict:
    """Parse the nested dict-of-dicts ``interbeds_distributions`` parameter.

    Format: ``Layer1:{thick1:num1,thick2:num2},Layer2:{thick3:num3}``

    Returns a dict of ``{layer_name: {thickness: count, ...}, ...}``.
    """
    if raw is None:
        return {}

    result: dict[str, dict[float, float]] = {}

    # Split on '},' to separate top-level layer entries, then rejoin '}'
    chunks = re.split(r"\},\s*", raw)
    for chunk in chunks:
        chunk = chunk.strip().rstrip("}")
        if not chunk:
            continue
        # chunk is now "LayerName:{thick1:num1,thick2:num2"
        layer_name, _, inner = chunk.partition(":{")
        if not inner:
            # Try alternate format: maybe "LayerName:{inner"
            layer_name, _, inner = chunk.partition(":{")
            if not inner:
                # Could be "LayerName:thick1:num1,thick2:num2"
                layer_name, _, inner = chunk.partition(":")
        layer_name = layer_name.strip()
        inner = inner.strip().strip("{}")

        sub_dict: dict[float, float] = {}
        for pair in inner.split(","):
            pair = pair.strip()
            if ":" in pair:
                k, _, v = pair.partition(":")
                sub_dict[float(k.strip())] = float(v.strip())
        result[layer_name] = sub_dict

    return result


def load_par_file(path: str | Path) -> ModelConfig:
    """Load a legacy ``.par`` parameter file and return a :class:`ModelConfig`.

    This mirrors the parsing logic in the original ``parameters.py``
    (``read_parameters_admin`` + ``read_parameters_noadmin``), but returns
    a structured :class:`ModelConfig` dataclass instead of a 38-element tuple.

    Parameters
    ----------
    path : str or Path
        Path to the ``.par`` file.

    Returns
    -------
    ModelConfig

    Raises
    ------
    ConfigurationError
        On missing required parameters or parse errors.
    """
    path = Path(path)
    lines = _parse_par_lines(path)

    logger.info("Loading legacy parameter file: %s", path)

    # ---- Admin -------------------------------------------------------------
    run_name = _parse_simple(
        _read_par_value("run_name", lines, required=False), str, "run_name"
    )
    output_folder = _parse_simple(
        _read_par_value("output_folder", lines, required=False), str, "output_folder"
    )
    overwrite = _parse_simple(
        _read_par_value("overwrite", lines, required=False), bool, "overwrite", False
    )
    internal_time_delay = _parse_simple(
        _read_par_value("internal_time_delay", lines, required=False),
        float,
        "internal_time_delay",
        0.5,
    )

    admin = AdminConfig(
        run_name=run_name or "",
        output_folder=output_folder or "",
        overwrite=overwrite,
        internal_time_delay=internal_time_delay,
    )

    # ---- Mode --------------------------------------------------------------
    mode = _parse_simple(
        _read_par_value("mode", lines, required=False), str, "mode", "Normal"
    )

    # ---- Layer structure ---------------------------------------------------
    no_layers = _parse_simple(
        _read_par_value("no_layers", lines, required=False), int, "no_layers", 2
    )

    layer_names_raw = _read_par_value("layer_names", lines, required=False)
    layer_names = _parse_list(layer_names_raw, "layer_names")
    if len(layer_names) != no_layers:
        logger.warning(
            "no_layers=%d but %d layer names found. Using the names as given.",
            no_layers,
            len(layer_names),
        )

    layer_types_raw = _read_par_value("layer_types", lines, required=False)
    layer_types = _parse_dict_flat(
        layer_types_raw, str, "layer_types"
    )
    if not isinstance(layer_types, dict):
        raise ConfigurationError(
            f"layer_types must be a dictionary, got {type(layer_types).__name__}."
        )

    layer_thickness_types_raw = _read_par_value(
        "layer_thickness_types", lines, required=False
    )
    layer_thickness_types = _parse_dict_flat(
        layer_thickness_types_raw, str, "layer_thickness_types"
    )
    # If it came back as a single string, apply to all layers
    if isinstance(layer_thickness_types, str):
        layer_thickness_types = {n: layer_thickness_types for n in layer_names}

    layer_thicknesses_raw = _read_par_value(
        "layer_thicknesses", lines, required=False
    )
    layer_thicknesses = _parse_layer_thicknesses(
        layer_thicknesses_raw, "layer_thicknesses"
    )

    layer_compaction_switch_raw = _read_par_value(
        "layer_compaction_switch", lines, required=False
    )
    layer_compaction_switch = _parse_dict_flat(
        layer_compaction_switch_raw, bool, "layer_compaction_switch"
    )
    if not isinstance(layer_compaction_switch, dict):
        layer_compaction_switch = {n: _str_to_bool(layer_compaction_switch) for n in layer_names}

    # ---- Interbeds ---------------------------------------------------------
    aquifer_names = [n for n in layer_names if layer_types.get(n) == "Aquifer"]

    interbeds_switch_raw = _read_par_value(
        "interbeds_switch", lines, required=False
    )
    interbeds_switch = _parse_dict_flat(
        interbeds_switch_raw, bool, "interbeds_switch"
    )
    if not isinstance(interbeds_switch, dict):
        interbeds_switch = {n: False for n in aquifer_names}

    interbeds_distributions_raw = _read_par_value(
        "interbeds_distributions", lines, required=False
    )
    interbeds_distributions = _parse_interbeds_distributions(
        interbeds_distributions_raw
    )

    # ---- Build LayerConfig objects -----------------------------------------
    layers: list[LayerConfig] = []
    for name in layer_names:
        ltype = layer_types.get(name, "Aquifer")
        th_type = (
            layer_thickness_types[name]
            if isinstance(layer_thickness_types, dict)
            else layer_thickness_types
        )
        th = layer_thicknesses.get(name, 100.0)
        comp = (
            layer_compaction_switch.get(name, True)
            if isinstance(layer_compaction_switch, dict)
            else layer_compaction_switch
        )
        ibs = interbeds_switch.get(name, False) if isinstance(interbeds_switch, dict) else False
        ibd = interbeds_distributions.get(name) if ibs else None
        layers.append(
            LayerConfig(
                name=name,
                layer_type=ltype,
                thickness=th,
                compaction_switch=comp,
                thickness_type=th_type,
                interbeds_switch=ibs,
                interbeds_distributions=ibd,
            )
        )

    # ---- Solver parameters -------------------------------------------------
    # Determine which layers require clay solving
    aquitards = [n for n, t in layer_types.items() if t == "Aquitard"]
    interbedded = [n for n, v in interbeds_switch.items() if v]
    layers_requiring_solving = interbedded + aquitards

    dt_master_raw = _read_par_value("dt_master", lines, required=False)
    dt_master = _parse_dict_flat(dt_master_raw, float, "dt_master")
    if not isinstance(dt_master, dict):
        dt_master = {n: float(dt_master) for n in layers_requiring_solving}

    dz_clays_raw = _read_par_value("dz_clays", lines, required=False)
    dz_clays = _parse_dict_flat(dz_clays_raw, float, "dz_clays")
    if not isinstance(dz_clays, dict):
        dz_clays = {n: float(dz_clays) for n in layers_requiring_solving}

    gw_solver_raw = _read_par_value(
        "groundwater_flow_solver_type", lines, required=False
    )
    groundwater_flow_solver_type = _parse_dict_flat(
        gw_solver_raw, str, "groundwater_flow_solver_type"
    )
    if not isinstance(groundwater_flow_solver_type, dict):
        groundwater_flow_solver_type = {
            n: str(groundwater_flow_solver_type) for n in layers_requiring_solving
        }

    vert_cond_raw = _read_par_value(
        "vertical_conductivity", lines, required=False
    )
    vertical_conductivity = _parse_dict_flat(
        vert_cond_raw, float, "vertical_conductivity"
    )
    if not isinstance(vertical_conductivity, dict):
        vertical_conductivity = {
            n: float(vertical_conductivity) for n in layers_requiring_solving
        }

    # --- New solver options (all have defaults) ---
    default_solver = _parse_simple(
        _read_par_value("default_solver", lines, required=False),
        str, "default_solver", "explicit",
    )
    parallel_layers = _parse_simple(
        _read_par_value("parallel_layers", lines, required=False),
        bool, "parallel_layers", True,
    )
    smoothing_width = _parse_simple(
        _read_par_value("smoothing_width", lines, required=False),
        float, "smoothing_width", 0.0,
    )
    adaptive_timestepping = _parse_simple(
        _read_par_value("adaptive_timestepping", lines, required=False),
        bool, "adaptive_timestepping", False,
    )
    mass_balance_check = _parse_simple(
        _read_par_value("mass_balance_check", lines, required=False),
        bool, "mass_balance_check", False,
    )
    mass_balance_threshold = _parse_simple(
        _read_par_value("mass_balance_threshold", lines, required=False),
        float, "mass_balance_threshold", 1e-6,
    )

    solver = SolverConfig(
        dt_master=dt_master,
        dz_clays=dz_clays,
        groundwater_flow_solver_type=groundwater_flow_solver_type,
        vertical_conductivity=vertical_conductivity,
        default_solver=default_solver,
        parallel_layers=parallel_layers,
        smoothing_width=smoothing_width,
        adaptive_timestepping=adaptive_timestepping,
        mass_balance_check=mass_balance_check,
        mass_balance_threshold=mass_balance_threshold,
    )

    # ---- Hydrologic parameters ---------------------------------------------
    clay_Sse_raw = _read_par_value("clay_Sse", lines, required=False)
    clay_Sse = _parse_dict_flat(clay_Sse_raw, float, "clay_Sse", {})
    if not isinstance(clay_Sse, dict):
        clay_Sse = {}

    clay_Ssv_raw = _read_par_value("clay_Ssv", lines, required=False)
    clay_Ssv = _parse_dict_flat(clay_Ssv_raw, float, "clay_Ssv", {})
    if not isinstance(clay_Ssv, dict):
        clay_Ssv = {}

    clay_Ssk_raw = _read_par_value("clay_Ssk", lines, required=False)
    clay_Ssk = _parse_dict_flat(clay_Ssk_raw, float, "clay_Ssk", None)
    if clay_Ssk is not None and not isinstance(clay_Ssk, dict):
        clay_Ssk = None

    sand_Sse_raw = _read_par_value("sand_Sse", lines, required=False)
    sand_Sse = _parse_dict_flat(sand_Sse_raw, float, "sand_Sse", {})
    if not isinstance(sand_Sse, dict):
        sand_Sse = {}

    sand_Ssk_raw = _read_par_value("sand_Ssk", lines, required=False)
    sand_Ssk = _parse_dict_flat(sand_Ssk_raw, float, "sand_Ssk", None)
    if sand_Ssk is not None and not isinstance(sand_Ssk, dict):
        sand_Ssk = None

    compressibility_of_water = _parse_simple(
        _read_par_value("compressibility_of_water", lines, required=False),
        float,
        "compressibility_of_water",
        4.4e-10,
    )
    rho_w = _parse_simple(
        _read_par_value("rho_w", lines, required=False), float, "rho_w", 1000.0
    )
    g = _parse_simple(
        _read_par_value("g", lines, required=False), float, "g", 9.81
    )

    overburden_stress_gwflow = _parse_simple(
        _read_par_value("overburden_stress_gwflow", lines, required=False),
        bool,
        "overburden_stress_gwflow",
        False,
    )
    overburden_stress_compaction = _parse_simple(
        _read_par_value("overburden_stress_compaction", lines, required=False),
        bool,
        "overburden_stress_compaction",
        False,
    )

    if overburden_stress_gwflow or overburden_stress_compaction:
        specific_yield = _parse_simple(
            _read_par_value("specific_yield", lines, required=False),
            float,
            "specific_yield",
            0.2,
        )
    else:
        specific_yield = None

    hydro = HydrologicParams(
        clay_Sse=clay_Sse,
        clay_Ssv=clay_Ssv,
        clay_Ssk=clay_Ssk,
        sand_Sse=sand_Sse,
        sand_Ssk=sand_Ssk,
        compressibility_of_water=compressibility_of_water,
        rho_w=rho_w,
        g=g,
        specific_yield=specific_yield,
    )

    # ---- Output config -----------------------------------------------------
    save_output_head_timeseries = _parse_simple(
        _read_par_value("save_output_head_timeseries", lines, required=False),
        bool,
        "save_output_head_timeseries",
        False,
    )
    save_effective_stress = _parse_simple(
        _read_par_value("save_effective_stress", lines, required=False),
        bool,
        "save_effective_stress",
        False,
    )
    save_internal_compaction = _parse_simple(
        _read_par_value("save_internal_compaction", lines, required=False),
        bool,
        "save_internal_compaction",
        False,
    )
    save_s = _parse_simple(
        _read_par_value("save_s", lines, required=False),
        bool,
        "save_s",
        False,
    )

    create_output_head_video_raw = _read_par_value(
        "create_output_head_video", lines, required=False
    )
    create_output_head_video = _parse_dict_flat(
        create_output_head_video_raw, bool, "create_output_head_video", False
    )

    output = OutputConfig(
        save_output_head_timeseries=save_output_head_timeseries,
        save_effective_stress=save_effective_stress,
        save_internal_compaction=save_internal_compaction,
        create_output_head_video=create_output_head_video,
        save_s=save_s,
    )

    # ---- Remaining top-level parameters ------------------------------------
    compaction_solver_compressibility_type_raw = _read_par_value(
        "compaction_solver_compressibility_type", lines, required=False
    )
    compaction_solver_compressibility_type = _parse_dict_flat(
        compaction_solver_compressibility_type_raw,
        str,
        "compaction_solver_compressibility_type",
        {},
    )

    compaction_solver_debug_include_endnodes = _parse_simple(
        _read_par_value(
            "compaction_solver_debug_include_endnodes", lines, required=False
        ),
        bool,
        "compaction_solver_debug_include_endnodes",
        False,
    )

    preconsolidation_head_type = _parse_simple(
        _read_par_value("preconsolidation_head_type", lines, required=False),
        str,
        "preconsolidation_head_type",
        "initial",
    )
    preconsolidation_head_offset: Any = None
    if preconsolidation_head_type == "initial_plus_offset":
        pho_raw = _read_par_value(
            "preconsolidation_head_offset", lines, required=False
        )
        preconsolidation_head_offset = _parse_dict_flat(
            pho_raw, float, "preconsolidation_head_offset"
        )

    time_unit = _parse_simple(
        _read_par_value("time_unit", lines, required=False),
        str,
        "time_unit",
        "days",
    )

    # ---- head_data_files ---------------------------------------------------
    head_data_files_raw = _read_par_value(
        "head_data_files", lines, required=False
    )
    head_data_files = _parse_dict_flat(
        head_data_files_raw, str, "head_data_files", {}
    )
    if not isinstance(head_data_files, dict):
        head_data_files = {}

    # ---- initial_stress options --------------------------------------------
    initial_stress_type_raw = _read_par_value(
        "initial_stress_type", lines, required=False
    )
    initial_stress_type = (
        _parse_dict_flat(initial_stress_type_raw, str, "initial_stress_type")
        if initial_stress_type_raw is not None
        else None
    )

    initial_stress_offset_raw = _read_par_value(
        "initial_stress_offset", lines, required=False
    )
    initial_stress_offset = (
        _parse_dict_flat(initial_stress_offset_raw, float, "initial_stress_offset")
        if initial_stress_offset_raw is not None
        else None
    )

    initial_stress_offset_unit = _parse_simple(
        _read_par_value("initial_stress_offset_unit", lines, required=False),
        str,
        "initial_stress_offset_unit",
        "head",
    )

    # ---- Resume options ----------------------------------------------------
    resume_directory = _parse_simple(
        _read_par_value("resume_directory", lines, required=False),
        str,
        "resume_directory",
        None,
    )
    resume_date = _parse_simple(
        _read_par_value("resume_date", lines, required=False),
        str,
        "resume_date",
        None,
    )
    resume_head_value_raw = _read_par_value(
        "resume_head_value", lines, required=False
    )
    resume_head_value = (
        _parse_dict_flat(resume_head_value_raw, str, "resume_head_value")
        if resume_head_value_raw is not None
        else None
    )

    # ---- Assemble ----------------------------------------------------------
    config = ModelConfig(
        admin=admin,
        layers=layers,
        layer_names=layer_names,
        layer_types=layer_types,
        solver=solver,
        hydro=hydro,
        output=output,
        overburden_stress_gwflow=overburden_stress_gwflow,
        overburden_stress_compaction=overburden_stress_compaction,
        compaction_solver_compressibility_type=compaction_solver_compressibility_type,
        compaction_solver_debug_include_endnodes=compaction_solver_debug_include_endnodes,
        preconsolidation_head_type=preconsolidation_head_type,
        preconsolidation_head_offset=preconsolidation_head_offset,
        time_unit=time_unit,
        mode=mode,
        head_data_files=head_data_files,
        initial_stress_type=initial_stress_type,
        initial_stress_offset=initial_stress_offset,
        initial_stress_offset_unit=initial_stress_offset_unit,
        resume_directory=resume_directory,
        resume_date=resume_date,
        resume_head_value=resume_head_value,
    )

    validate_config(config)
    return config
