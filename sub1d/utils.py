"""Utility functions for the SUB1D subsidence model."""
from __future__ import annotations

import os
import sys
import time
import shutil
import logging
import tempfile
from typing import Callable

import matplotlib
import matplotlib.image

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure logging for the SUB1D model.

    Parameters
    ----------
    level : int
        Logging level (default INFO).
    log_file : str, optional
        If provided, also log to this file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def log_progress(current: int, total: int, description: str = "",
                 interval_pct: int = 10) -> None:
    """Log progress at specified percentage intervals.

    Parameters
    ----------
    current : int
        Current iteration index.
    total : int
        Total number of iterations.
    description : str
        What is being processed.
    interval_pct : int
        Percentage interval at which to log (default 10%).
    """
    if total <= 0:
        return
    step = max(1, total * interval_pct // 100)
    if current % step == 0 or current == total - 1:
        pct = 100.0 * current / total
        logger.info("%s%.0f%% complete (%d/%d)",
                    f"{description}: " if description else "", pct, current, total)


def with_retries(func: Callable, max_retries: int = 3, delay: float = 1.0):
    """Retry a function on PermissionError.

    Parameters
    ----------
    func : callable
        Function to call.
    max_retries : int
        Maximum number of attempts.
    delay : float
        Seconds to wait between retries.

    Returns
    -------
    Any
        Return value of func.

    Raises
    ------
    PermissionError
        If all retries fail.
    """
    for i in range(max_retries):
        try:
            return func()
        except PermissionError as e:
            logger.warning("Permission error on attempt %d: %s", i + 1, e)
            time.sleep(delay)
    raise PermissionError(f"Failed after {max_retries} attempts.")


def make_output_folder(param_filename: str, outdestination: str,
                       overwrite: bool = False) -> None:
    """Create the output directory structure.

    Parameters
    ----------
    param_filename : str
        Path to the parameter file (will be copied to output dir).
    outdestination : str
        Destination directory path.
    overwrite : bool
        If True and directory exists, back it up and recreate.

    Raises
    ------
    FileExistsError
        If directory exists and overwrite is False.
    """
    try:
        if not os.path.isdir(outdestination):
            logger.info("Creating output directory: %s", outdestination)
            with_retries(lambda: os.makedirs(outdestination, exist_ok=True))
            os.makedirs(os.path.join(outdestination, "figures"), exist_ok=True)
            os.makedirs(os.path.join(outdestination, "s_outputs"), exist_ok=True)
            os.makedirs(os.path.join(outdestination, "head_outputs"), exist_ok=True)
        else:
            if overwrite:
                logger.info("Overwriting output directory: %s", outdestination)
                backup_folder = f"{outdestination}_old"
                if os.path.isdir(backup_folder):
                    logger.info("Removing previous backup: %s", backup_folder)
                    with_retries(lambda: shutil.rmtree(backup_folder))
                    time.sleep(1)
                with_retries(lambda: shutil.copytree(outdestination, backup_folder))
                with_retries(lambda: shutil.rmtree(outdestination))
                time.sleep(1)
                with_retries(lambda: os.makedirs(outdestination, exist_ok=True))
                os.makedirs(os.path.join(outdestination, "figures"), exist_ok=True)
                os.makedirs(os.path.join(outdestination, "s_outputs"), exist_ok=True)
                os.makedirs(os.path.join(outdestination, "head_outputs"), exist_ok=True)
            else:
                raise FileExistsError(
                    f"Output folder {outdestination} already exists and overwrite=False."
                )
    except PermissionError as e:
        logger.error("Permission error creating output folder: %s", e)
        raise

    if os.path.exists(param_filename):
        shutil.copy2(param_filename, os.path.join(outdestination, "paramfile.par"))


def get_size(fig, dpi: int = 100) -> tuple[float, float]:
    """Get the actual rendered size of a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to measure.
    dpi : int
        DPI for rendering.

    Returns
    -------
    tuple[float, float]
        (width, height) in inches.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
        height, width, _channels = matplotlib.image.imread(f.name).shape
        return width / dpi, height / dpi


def set_size(fig, size: tuple[float, float], dpi: int = 100,
             eps: float = 1e-2, max_iterations: int = 10,
             min_size_px: int = 10) -> bool:
    """Iteratively adjust figure size to match target.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to resize.
    size : tuple[float, float]
        Target (width, height) in inches.
    dpi : int
        DPI for rendering.
    eps : float
        Convergence tolerance.
    max_iterations : int
        Maximum number of iterations (prevents infinite loop).
    min_size_px : int
        Minimum size in pixels.

    Returns
    -------
    bool
        True if converged within tolerance, False otherwise.
    """
    target_width, target_height = size
    set_width, set_height = target_width, target_height
    deltas = []

    for _ in range(max_iterations):
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))

        if deltas[-1] < eps:
            return True
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False

    return False
