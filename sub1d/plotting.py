"""Plotting functions for the SUB1D subsidence model.

Provides visualization of head timeseries, overburden stress, clay distributions,
and animations of head/compaction evolution. Duplicate video functions from the
legacy code have been merged into a single `create_animation` function.
"""
from __future__ import annotations

import os
import glob
import time
import logging
import datetime
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# Try to import seaborn; fall back gracefully
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _set_style(context: str = "talk", style: str = "darkgrid") -> None:
    """Set plotting style if seaborn is available."""
    if HAS_SEABORN:
        sns.set_style(style)
        sns.set_context(context)


def plot_overburden_stress(overburden_df, out_path: str | Path) -> None:
    """Create and save plot of overburden stress as a function of time.

    Parameters
    ----------
    overburden_df : pd.DataFrame
        DataFrame with columns [dates, overburden stress].
    out_path : str or Path
        Directory or file path for output.
    """
    _set_style("notebook")
    fig, ax = plt.subplots(figsize=(18, 12))

    ax.plot(overburden_df.iloc[:, 0].to_numpy(), overburden_df.iloc[:, 1].to_numpy())
    ax.set_title("Overburden stress for aquifer as a function of time")
    ax.set_ylabel(r"Overburden stress ($\frac{N}{m^2}$)")
    ax.set_xlabel("Time")

    out_path = Path(out_path)
    if out_path.is_dir():
        fig.savefig(out_path / "overburden_stress_series.png")
    else:
        fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved overburden stress plot")


def plot_head_timeseries(head_data: dict, aquifers: list[str],
                         out_path: str | Path) -> None:
    """Create and save plot of groundwater head timeseries.

    Parameters
    ----------
    head_data : dict[str, pd.DataFrame]
        Head data for each aquifer.
    aquifers : list[str]
        Aquifer names to plot.
    out_path : str or Path
        Directory or file path for output.
    """
    _set_style("notebook")
    fig, ax = plt.subplots(figsize=(18, 12))

    for aquifer in aquifers:
        df = head_data[aquifer]
        ax.plot(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), label=aquifer)

    ax.set_title("Aquifer Head as a function of time")
    ax.set_ylabel("Head (masl)")
    ax.set_xlabel("Time")
    ax.legend()

    out_path = Path(out_path)
    if out_path.is_dir():
        fig.savefig(out_path / "input_head_timeseries.png")
        fig.savefig(out_path / "input_head_timeseries.pdf")
    else:
        fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved head timeseries plot")


def plot_clay_distributions(layers_requiring_solving: list[str],
                            layer_types: dict, layer_thickness_types: dict,
                            layer_thicknesses: dict,
                            interbeds_distributions: dict,
                            out_path: str | Path) -> None:
    """Create and save bar chart of clay distributions.

    Parameters
    ----------
    layers_requiring_solving : list[str]
        Layers with clay to solve.
    layer_types : dict
        Mapping of layer name to type.
    layer_thickness_types : dict
        Mapping of layer name to thickness type.
    layer_thicknesses : dict
        Mapping of layer name to thickness value.
    interbeds_distributions : dict
        Mapping of layer name to {thickness: count} dict.
    out_path : str or Path
        Output directory or file path.
    """
    thicknesses_tmp = []
    for layer in layers_requiring_solving:
        if layer_types[layer] == "Aquifer":
            for key in interbeds_distributions.get(layer, {}).keys():
                thicknesses_tmp.append(key)
        if layer_types[layer] == "Aquitard":
            if layer_thickness_types.get(layer) == "constant":
                thicknesses_tmp.append(layer_thicknesses[layer])

    if not thicknesses_tmp:
        logger.warning("No clay thicknesses to plot")
        return

    # Calculate bar width
    if len(thicknesses_tmp) > 1:
        arr = np.array(thicknesses_tmp, dtype=float)
        diffs = np.abs(arr[:, None] - arr[None, :])
        np.fill_diagonal(diffs, np.inf)
        smallest_width = min(np.min(diffs), 0.5)
    else:
        smallest_width = 0.5

    barwidth = smallest_width / max(len(layers_requiring_solving), 1)

    _set_style("poster")
    fig, ax = plt.subplots(figsize=(18, 12))

    palette = plt.cm.tab10.colors if not HAS_SEABORN else sns.color_palette()

    for i, layer in enumerate(layers_requiring_solving):
        color = palette[i % len(palette)]
        if layer_types[layer] == "Aquifer" and layer in interbeds_distributions:
            dist = interbeds_distributions[layer]
            ax.bar(
                np.array(list(dist.keys()), dtype=float) + i * barwidth,
                list(dist.values()),
                width=barwidth, label=layer, color="None",
                edgecolor=color, linewidth=5, alpha=0.6,
            )
        elif layer_types[layer] == "Aquitard":
            ax.bar(
                float(layer_thicknesses[layer]) + i * barwidth,
                1, width=barwidth, label=layer, color="None",
                edgecolor=color, linewidth=5, alpha=0.6,
            )

    ax.legend()
    ax.set_title("Clay Interbed Distribution")
    max_thick = max(float(t) for t in thicknesses_tmp)
    ax.set_xticks(np.arange(0, max_thick + barwidth + 1, max(1, barwidth)))
    ax.set_xlabel("Layer thickness (m)")
    ax.set_ylabel("Number of layers")

    out_path = Path(out_path)
    if out_path.is_dir():
        fig.savefig(out_path / "clay_distributions.png", bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved clay distribution plot")


def create_animation(data: np.ndarray, z: np.ndarray, times: np.ndarray,
                     inelastic_flag: np.ndarray, outputfolder: str,
                     layer: str, mode: str = "head",
                     delt: float = 100, startyear: int | None = None,
                     endyear: int | None = None,
                     datelabels: str = "year") -> None:
    """Create animation frames and stitch into video using ffmpeg.

    This is a unified function replacing the separate `create_head_video_elasticinelastic`
    and `create_compaction_video` functions which were ~90% identical.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (nodes x timesteps). Head values or compaction rates.
    z : np.ndarray
        Spatial coordinates.
    times : np.ndarray
        Time array (numeric matplotlib dates or similar).
    inelastic_flag : np.ndarray
        Boolean array (nodes x timesteps) indicating inelastic nodes.
    outputfolder : str
        Base output directory.
    layer : str
        Layer name (used in subfolder and filenames).
    mode : str
        "head" for head evolution, "compaction" for compaction rate.
    delt : float
        Time interval between frames (days).
    startyear : int, optional
        Start year for the video.
    endyear : int, optional
        End year for the video.
    datelabels : str
        "year" or "monthyear" for title format.
    """
    t_start_wall = time.time()
    layer_safe = layer.replace(" ", "_")

    if mode == "head":
        subdir = f"headvideo_{layer_safe}"
        xlabel = "Head in the clay (m)"
        prefix = "vid"
        fps = 10
    else:
        subdir = f"compactionvideo_{layer_safe}"
        xlabel = "node compaction rate (cm yr-1)"
        prefix = "vid_b_inst"
        fps = 5

    frame_dir = os.path.join(outputfolder, subdir)
    os.makedirs(frame_dir, exist_ok=True)

    plt.ioff()
    _set_style("talk")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (m)")
    ax.set_xlim([np.max(data) + 1, np.min(data) - 1])
    z_margin = 0.1 * (z[-1] - z[1]) if len(z) > 1 else 1.0
    ax.set_ylim([np.min(z) - z_margin, np.max(z) + z_margin])
    ax.invert_xaxis()
    ax.invert_yaxis()
    if mode == "compaction":
        ax.set_xscale("symlog")

    t_tmp = np.round(times, 5)

    # Determine start/end times
    if startyear:
        starting_t = mdates.date2num(
            datetime.datetime.strptime(f"01-09-{startyear}", "%d-%m-%Y"))
    else:
        starting_t = t_tmp[0]

    if endyear:
        ending_t = mdates.date2num(
            datetime.datetime.strptime(f"01-09-{endyear}", "%d-%m-%Y"))
    else:
        ending_t = t_tmp[-1]

    logger.info("Animation %s-%s for %s", starting_t, ending_t, layer)
    ts_to_do = np.arange(starting_t, ending_t + 0.0001, delt)

    l1 = l2 = l3 = None
    log_step = max(1, len(ts_to_do) // 20)

    for frame_idx, t in enumerate(ts_to_do):
        matches = np.argwhere(t_tmp == t)
        if len(matches) == 0:
            continue
        i = matches[0][0]

        if frame_idx % log_step == 0:
            logger.info("Animation frame %d/%d", frame_idx, len(ts_to_do))

        ifl = inelastic_flag[:, i].astype(bool) if inelastic_flag is not None else np.zeros(len(z), dtype=bool)

        if l2 is None:
            # First frame
            if mode == "head":
                (l1,) = ax.plot(np.min(data[:, :i + 1], axis=1), z,
                                "b--", label="Preconsolidation head")
            (l2,) = ax.plot(data[:, i], z, "r.",
                            label="inelastic node" if mode == "compaction" else None)
            (l3,) = ax.plot(data[:, i][~ifl], z[~ifl], "g.",
                            label="elastic node" if mode == "compaction" else None)
            ax.legend()
        else:
            if mode == "head" and l1 is not None:
                l1.set_xdata(np.min(data[:, :i + 1], axis=1))
            l2.set_xdata(data[:, i])
            l3.remove()
            (l3,) = ax.plot(data[:, i][~ifl], z[~ifl], "g.")

        if datelabels == "year":
            ax.set_title("t=%s" % mdates.num2date(t_tmp[i]).strftime("%Y"))
        elif datelabels == "monthyear":
            ax.set_title("t=%s" % mdates.num2date(t_tmp[i]).strftime("%b-%Y"))

        fig.savefig(os.path.join(frame_dir, f"frame{i:06d}.jpg"),
                    dpi=60, bbox_inches="tight")

    plt.close(fig)
    t_elapsed = time.time() - t_start_wall
    logger.info("Frame generation took %.1f seconds", t_elapsed)

    # Stitch frames with ffmpeg (no shell=True)
    start_str = mdates.num2date(starting_t).strftime("%Y")
    end_str = mdates.num2date(ending_t).strftime("%Y")
    video_name = f"{prefix}_years{start_str}to{end_str}.mp4"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-r", str(fps), "-f", "image2",
        "-pattern_type", "glob", "-i", "*.jpg",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        video_name,
    ]

    logger.info("Stitching frames: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=frame_dir,
                       capture_output=True, text=True)

        # Clean up frames on success
        video_path = os.path.join(frame_dir, video_name)
        if os.path.isfile(video_path):
            logger.info("Video created successfully, cleaning up frames")
            for jpg in glob.glob(os.path.join(frame_dir, "*.jpg")):
                os.remove(jpg)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("ffmpeg failed (frames retained): %s", e)


def plot_deformation(deformation: dict, groundwater_solution_dates,
                     layer: str, outdestination: str) -> None:
    """Plot deformation timeseries for a layer.

    Parameters
    ----------
    deformation : dict
        Deformation data for the layer.
    groundwater_solution_dates : array-like
        Dates for the solution.
    layer : str
        Layer name.
    outdestination : str
        Output directory.
    """
    _set_style("talk", "darkgrid")
    fig, ax = plt.subplots(figsize=(18, 12))

    if "total" in deformation:
        total = deformation["total"]
        if total.ndim == 2:
            ax.plot_date(total[0, :], total[1, :], "-", label="total")
        else:
            ax.plot(total, "-", label="total")

    ax.legend()
    ax.set_ylabel("Deformation (m)")
    ax.set_title(layer)

    layer_safe = layer.replace(" ", "_")
    fig_dir = os.path.join(outdestination, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f"compaction_{layer_safe}.png"),
                bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved deformation plot for %s", layer)


def plot_total_deformation(deformation_by_layer: dict, layer_names: list[str],
                           layer_types: dict, layer_compaction_switch: dict,
                           outdestination: str) -> None:
    """Plot total deformation across all layers.

    Parameters
    ----------
    deformation_by_layer : dict
        Deformation data keyed by layer name.
    layer_names : list[str]
        All layer names.
    layer_types : dict
        Layer types.
    layer_compaction_switch : dict
        Which layers compact.
    outdestination : str
        Output directory.
    """
    _set_style("talk", "whitegrid")
    fig, ax = plt.subplots(figsize=(18, 12))

    for layer in layer_names:
        if not layer_compaction_switch.get(layer, False):
            continue
        if layer not in deformation_by_layer:
            continue

        total = deformation_by_layer[layer].get("total")
        if total is not None and total.ndim == 2:
            ax.plot_date(total[0, :], total[1, :], "-", label=layer)

    ax.set_ylabel("Deformation (m)")
    ax.legend()

    fig_dir = os.path.join(outdestination, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "total_deformation_figure.png"),
                bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved total deformation plot")
