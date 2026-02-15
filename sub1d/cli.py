"""Command-line interface for the SUB1D subsidence model."""
from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list. Defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="sub1d",
        description="SUB1D: 1D Land Subsidence / Compaction Model",
    )
    parser.add_argument(
        "config",
        help="Path to configuration file (.yaml or .par)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output directory if it exists",
    )
    parser.add_argument(
        "--solver",
        choices=["explicit", "crank-nicolson"],
        default=None,
        help="Override solver type (default: use config file setting)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--no-compaction",
        action="store_true",
        default=False,
        help="Skip compaction solving (head equations only)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the SUB1D model.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    args = parse_args(argv)

    from sub1d.utils import setup_logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("SUB1D: 1D Land Subsidence / Compaction Model")
    logger.info("Configuration file: %s", args.config)

    t_start = time.time()

    # Load configuration
    from sub1d.config import load_par_file, load_yaml_config

    config_path = args.config
    if config_path.endswith((".yaml", ".yml")):
        config = load_yaml_config(config_path)
    else:
        config = load_par_file(config_path)

    # Apply CLI overrides
    if args.overwrite:
        config.admin.overwrite = True

    # Run the model
    from sub1d.model import run_model

    results = run_model(config, solver_override=args.solver,
                        skip_compaction=args.no_compaction)

    t_elapsed = time.time() - t_start
    logger.info("Model run complete. Total time: %.1f seconds", t_elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
