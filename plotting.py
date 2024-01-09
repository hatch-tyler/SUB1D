import os
import sys
import time
import glob
import datetime
import matplotlib
import subprocess

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import printProgressBar


def plot_overburden_stress(overburden_df, out_path):
    """
    Create and save plot of overburden stress as a function of time

    Parameters
    ----------
    overburden_df : pd.DataFrame
        pandas DataFrame containing dates and overburden stress

    out_path : str
        path to save overburden time series plot

    Returns
    -------
    None
        saves plot to out_path
    """
    plt.figure(figsize=(18, 12))

    # set plotting styles
    sns.set_style("darkgrid")
    sns.set_context("notebook")

    # plot overburden stress
    plt.plot(overburden_df.iloc[:, 0].to_numpy(), overburden_df.iloc[:, 1].to_numpy())

    # set labels and title
    plt.title("Overburden stress for aquifer as a function of time")
    plt.ylabel(r"Overburden stress ($\frac{N}{m^2}$)")
    plt.xlabel("Time")

    # save figure
    if os.path.isfile(out_path):
        plt.savefig(out_path)
    elif os.path.isdir(out_path):
        plt.savefig(os.path.join(out_path, "overburden_stress_series.png"))

    plt.close()


def plot_head_timeseries(head_data, aquifers_needing_heads, out_path):
    """
    Create and save plot of groundwater head as a function of time

    Parameters
    ----------
    head_data : dict[str, pd.DataFrame]
        dictionary containing pandas DataFrames of heads for each aquifer

    aquifers_needing_heads : list
        list of aquifer names needing head data

    out_path : str
        path to save head time series plots

    Returns
    -------
    None
        saves plots to output_path
    """
    # plot input head timeseries
    plt.figure(figsize=(18, 12))
    sns.set_style("darkgrid")
    sns.set_context("notebook")

    for aquifer in aquifers_needing_heads:
        plt.plot(
            head_data[aquifer].iloc[:, 0].to_numpy(),
            head_data[aquifer].iloc[:, 1].to_numpy(),
            label=aquifer,
        )

    plt.title("Aquifer Head as a function of time")
    plt.ylabel("Head (masl)")
    plt.xlabel("Time")
    plt.legend()

    if os.path.isfile(out_path):
        plt.savefig(out_path)
    elif os.path.isdir(out_path):
        plt.savefig(os.path.join(out_path, "input_head_timeseries.png"))
        plt.savefig(os.path.join(out_path, "input_head_timeseries.pdf"))

    plt.close()


def plot_clay_distributions(
    layers_requiring_solving,
    layer_types,
    layer_thickness_types,
    layer_thicknesses,
    interbeds_distributions,
    out_path,
):
    """
    Create and save a bar chart of clay distributions

    Parameters
    ----------


    Returns
    -------
    None
        saves plot to output path
    """
    thicknesses_tmp = []
    for layer in layers_requiring_solving:
        if layer_types[layer] == "Aquifer":
            for key in interbeds_distributions[layer].keys():
                thicknesses_tmp.append(key)
        if layer_types[layer] == "Aquitard":
            if layer_thickness_types[layer] == "constant":
                thicknesses_tmp.append(layer_thicknesses[layer])
            else:
                raise NotImplementedError(
                    "Error, aquitards with varying thickness not (yet) supported."
                )

    # Find the smallest difference between two clay layer thicknesses. 
    # If that is greater than 1, set the bar width to be 1. Otherwise, set the bar width to be that difference.
    print(thicknesses_tmp)

    # calculate the width of the bars in the bar chart
    if len(thicknesses_tmp) > 1:
        diffs = [np.array(thicknesses_tmp) - t for t in thicknesses_tmp]
        smallest_width = np.min(np.abs(np.array(diffs)[np.where(diffs)]))
    else:
        smallest_width = 1
    if smallest_width > 0.5:
        smallest_width = 0.5

    barwidth = smallest_width / len(layers_requiring_solving)

    # Make the clay distribution plot
    sns.set_context("poster")
    plt.figure(figsize=(18, 12))

    layeri = 0
    for layer in layers_requiring_solving:
        if layer_types[layer] == "Aquifer":
            plt.bar(
                np.array(list(interbeds_distributions[layer].keys()))
                + layeri * barwidth,
                list(interbeds_distributions[layer].values()),
                width=barwidth,
                label=layer,
                color="None",
                edgecolor=sns.color_palette()[layeri],
                linewidth=5,
                alpha=0.6,
            )
            layeri += 1
        if layer_types[layer] == "Aquitard":
            plt.bar(
                layer_thicknesses[layer] + layeri * barwidth,
                1,
                width=barwidth,
                label=layer,
                color="None",
                edgecolor=sns.color_palette()[layeri],
                linewidth=5,
                alpha=0.6,
            )
            layeri += 1

    # format plot
    plt.legend()
    plt.title("Clay Interbed Distribution")
    plt.xticks(
        np.arange(0, np.max(thicknesses_tmp) + barwidth + 1, np.max([1, barwidth]))
    )
    plt.xlabel("Layer thickness (m)")
    plt.ylabel("Number of layers")

    # save plot
    if os.path.isdir(out_path):
        plt.savefig(
            os.path.join(out_path, "clay_distributions.png"), bbox_inches="tight"
        )
    elif os.path.isfile(out_path):
        plt.savefig(out_path, bbox_inches="tight")

    plt.close()


def create_head_video_elasticinelastic(
    hmat,
    z,
    inelastic_flag,
    dates_str,
    outputfolder,
    layer,
    delt=100,
    startyear=None,
    endyear=None,
    datelabels="year",
):
    # I think delt is in units of days; see what happens with the variable t_jumps below. startyear and endyear should be YYYY. datelabels can be 'year' or 'monthyear' and changes the title of the plots.
    if not os.path.isdir("%s/headvideo_%s" % (outputfolder, layer.replace(" ", "_"))):
        os.mkdir("%s/headvideo_%s" % (outputfolder, layer.replace(" ", "_")))

    t1_start = time.time()

    # Make the video frames; use ffmpeg -f image2 -i %*.png vid.mp4 to make the vid itself
    matplotlib.pyplot.ioff()
    sns.set_context("talk")
    print("\t\tCreating frames.")
    matplotlib.pyplot.figure(figsize=(6, 6))
    matplotlib.pyplot.xlabel("Head in the clay (m)")
    matplotlib.pyplot.ylabel("Z (m)")
    matplotlib.pyplot.xlim([np.max(hmat) + 1, np.min(hmat) - 1])
    matplotlib.pyplot.ylim(
        [np.min(z) - 0.1 * (z[-1] - z[1]), np.max(z) + 0.1 * (z[-1] - z[1])]
    )
    matplotlib.pyplot.gca().invert_xaxis()
    matplotlib.pyplot.gca().invert_yaxis()
    t_tmp = matplotlib.dates.date2num(
        [
            datetime.datetime.strptime(date, "%d-%b-%Y").datetime.date()
            for date in dates_str
        ]
    )

    t_jumps = [
        t_tmp[i] - t_tmp[0] for i in range(len(t_tmp))
    ]  # It looks like we get t in days.
    if not delt in t_jumps:
        print("\tERROR MAKING VIDEO! Selected dt not compatible with dates given.")
        sys.exit(1)

    if startyear:
        starting_t = matplotlib.dates.date2num(
            datetime.datetime.strptime("01-09-%s" % startyear, "%d-%m-%Y")
        )
        print(
            "Movie starting at date %s"
            % matplotlib.dates.num2date(starting_t).strftime("%d-%b-%Y")
        )
    else:
        starting_t = t_tmp[0]
        print(
            "Movie starting at date %s"
            % matplotlib.dates.num2date(starting_t).strftime("%d-%b-%Y")
        )

    if endyear:
        ending_t = matplotlib.dates.date2num(
            datetime.datetime.strptime("01-09-%s" % endyear, "%d-%m-%Y")
        )
        print(
            "Movie ending at date %s"
            % matplotlib.dates.num2date(ending_t).strftime("%d-%b-%Y")
        )
    else:
        ending_t = t_tmp[-1]
        print(
            "Movie ending at date %s"
            % matplotlib.dates.num2date(ending_t).strftime("%d-%b-%Y")
        )

    ts_to_do = np.arange(starting_t, ending_t + 0.0001, delt)

    firsttime = 0

    for t in ts_to_do:
        i = np.argwhere(t_tmp == t)[0][
            0
        ]  # If there are multiple frames on the same day, this line means we take the first of them
        if (
            firsttime == 2
        ):  # This whole "firsttime" bit is just to help print every 5%, it's really not important.
            if (
                i % (int(len(ts_to_do) / 20)) <= di
            ):  # This should make it so only 20 progress bars are printed
                printProgressBar(np.argwhere(ts_to_do == t)[0][0], len(ts_to_do))
        else:
            printProgressBar(np.argwhere(ts_to_do == t)[0][0], len(ts_to_do))

        # matplotlib.pyplot.plot(hmat[:,0],np.linspace(0,40,20),'k--',label='t=0 position')
        if firsttime == 1:
            firsttime = 2
            di = np.argwhere(ts_to_do == t)[0][0]
        if t == starting_t:
            (l1,) = matplotlib.pyplot.plot(
                np.min(hmat[:, : i + 1], axis=1),
                z,
                "b--",
                label="Preconsolidation head",
            )
            (l2,) = matplotlib.pyplot.plot(hmat[:, i], z, "r.")
            (l3,) = matplotlib.pyplot.plot(
                hmat[:, i][~inelastic_flag[:, i]], z[~inelastic_flag[:, i]], "g."
            )
            matplotlib.pyplot.legend()
            firsttime = 1
        else:
            l1.set_xdata(np.min(hmat[:, : i + 1], axis=1))
            l2.set_xdata(hmat[:, i])
            l3.remove()
            (l3,) = matplotlib.pyplot.plot(
                hmat[:, i][~inelastic_flag[:, i]], z[~inelastic_flag[:, i]], "g."
            )
        if datelabels == "year":
            matplotlib.pyplot.title(
                "t=%s" % matplotlib.dates.num2date(t_tmp[i]).strftime("%Y")
            )
        if datelabels == "monthyear":
            matplotlib.pyplot.title(
                "t=%s" % matplotlib.dates.num2date(t_tmp[i]).strftime("%b-%Y")
            )

        #            set_size(matplotlib.pyplot.gcf(), (12, 12))
        matplotlib.pyplot.savefig(
            "%s/headvideo_%s/frame%06d.jpg"
            % (outputfolder, layer.replace(" ", "_"), i),
            dpi=60,
            bbox_inches="tight",
        )
    t1_stop = time.time()
    print("")
    print("\t\tElapsed time in seconds:", t1_stop - t1_start)
    print("")
    print("\t\tStitching frames together using ffmpeg.")
    # cmd='ffmpeg -hide_banner -loglevel warning -r 10 -f image2 -i %*.jpg vid.mp4'
    cmd = (
        'ffmpeg -hide_banner -loglevel warning -r 10 -f image2 -i %%*.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" vid_years%sto%s.mp4'
        % (
            matplotlib.dates.num2date(starting_t).strftime("%Y"),
            matplotlib.dates.num2date(ending_t).strftime("%Y"),
        )
    )

    print("\t\t\t%s." % cmd)
    cd = os.getcwd()
    os.chdir("%s/headvideo_%s" % (outputfolder, layer.replace(" ", "_")))
    subprocess.call(cmd, shell=True)

    if os.path.isfile(
        "vid_years%sto%s.mp4"
        % (
            matplotlib.dates.num2date(starting_t).strftime("%Y"),
            matplotlib.dates.num2date(ending_t).strftime("%Y"),
        )
    ):
        print("\tVideo seems to have been a success; deleting excess .jpg files.")
        jpg_files_tmp = glob.glob("*.jpg")
        for file in jpg_files_tmp:
            os.remove(file)
        print("\tDone.")

    os.chdir(cd)


def create_compaction_video(
    outputfolder,
    layer,
    db_plot,
    z,
    time_sim,
    Inelastic_Flag,
    delt=20,
    startyear=None,
    endyear=None,
    datelabels="year",
):
    t1_start = time.time()

    if not os.path.isdir(
        "%s/compactionvideo_%s" % (outputfolder, layer.replace(" ", "_"))
    ):
        os.mkdir("%s/compactionvideo_%s" % (outputfolder, layer.replace(" ", "_")))

    matplotlib.pyplot.ioff()
    sns.set_context("talk")
    print("\t\tCreating frames.")
    matplotlib.pyplot.figure(figsize=(8, 8))
    matplotlib.pyplot.xlabel("node compaction rate (cm yr-1)")
    matplotlib.pyplot.ylabel("Z (m)")
    matplotlib.pyplot.xlim([np.max(db_plot), np.min(db_plot)])
    matplotlib.pyplot.ylim(
        [np.min(z) - 0.1 * (z[-1] - z[1]), np.max(z) + 0.1 * (z[-1] - z[1])]
    )
    matplotlib.pyplot.gca().invert_xaxis()
    matplotlib.pyplot.gca().invert_yaxis()
    matplotlib.pyplot.xscale("symlog")
    t_tmp = np.round(time_sim, 5)

    t_jumps = np.round(
        [t_tmp[i] - t_tmp[0] for i in range(len(t_tmp))], 4
    )  # It looks like we get t in days.
    if not delt in t_jumps:
        print("\tERROR MAKING VIDEO! Selected dt not compatible with dates given.")
        sys.exit(1)

    if startyear:
        starting_t = matplotlib.dates.date2num(
            datetime.datetime.strptime("01-09-%s" % startyear, "%d-%m-%Y")
        )
        print(
            "Movie starting at date %s"
            % matplotlib.dates.num2date(starting_t).strftime("%d-%b-%Y")
        )
    else:
        starting_t = t_tmp[0]
        print(
            "Movie starting at date %s"
            % matplotlib.dates.num2date(starting_t).strftime("%d-%b-%Y")
        )

    if endyear:
        ending_t = matplotlib.dates.date2num(
            datetime.datetime.strptime("01-09-%s" % endyear, "%d-%m-%Y")
        )
        print(
            "Movie ending at date %s"
            % matplotlib.dates.num2date(ending_t).strftime("%d-%b-%Y")
        )
    else:
        ending_t = t_tmp[-1]
        print(
            "Movie ending at date %s"
            % matplotlib.dates.num2date(ending_t).strftime("%d-%b-%Y")
        )

    ts_to_do = (
        1 / 1000 * np.arange(1000 * starting_t, 1000 * ending_t + 0.0001, 1000 * delt)
    )  # avoid weird rounding issues

    firsttime = 0

    for t in ts_to_do:
        i = np.argwhere(t_tmp == t)[0][
            0
        ]  # If there are multiple frames on the same day, this line means we take the first of them
        if (
            firsttime == 2
        ):  # This whole "firsttime" bit is just to help print every 5%, it's really not important.
            if (
                i % (int(len(ts_to_do) / 20)) <= di
            ):  # This should make it so only 20 progress bars are printed
                printProgressBar(np.argwhere(ts_to_do == t)[0][0], len(ts_to_do))
        else:
            printProgressBar(np.argwhere(ts_to_do == t)[0][0], len(ts_to_do))

        # matplotlib.pyplot.plot(hmat[:,0],np.linspace(0,40,20),'k--',label='t=0 position')
        if firsttime == 1:
            firsttime = 2
            di = np.argwhere(ts_to_do == t)[0][0]
        if t == starting_t:
            (l2,) = matplotlib.pyplot.plot(
                db_plot[:, i], z, "r.", label="inelastic node"
            )
            (l3,) = matplotlib.pyplot.plot(
                db_plot[:, i][~Inelastic_Flag[:, i]],
                z[~Inelastic_Flag[:, i]],
                "g.",
                label="elastic node",
            )
            matplotlib.pyplot.legend()
            firsttime = 1
        else:
            l2.set_xdata(db_plot[:, i])
            l3.remove()
            (l3,) = matplotlib.pyplot.plot(
                db_plot[:, i][~Inelastic_Flag[:, i]], z[~Inelastic_Flag[:, i]], "g."
            )
        if datelabels == "year":
            matplotlib.pyplot.title(
                "t=%s" % matplotlib.dates.num2date(t_tmp[i]).strftime("%Y")
            )
        if datelabels == "monthyear":
            matplotlib.pyplot.title(
                "t=%s" % matplotlib.dates.num2date(t_tmp[i]).strftime("%b-%Y")
            )

        #            set_size(matplotlib.pyplot.gcf(), (12, 12))
        matplotlib.pyplot.savefig(
            "%s/compactionvideo_%s/frame%06d.jpg"
            % (outputfolder, layer.replace(" ", "_"), i),
            dpi=60,
            bbox_inches="tight",
        )
    print("")
    print("")
    print("\t\tStitching frames together using ffmpeg.")
    # cmd='ffmpeg -hide_banner -loglevel warning -r 10 -f image2 -i %*.jpg vid.mp4'
    cmd = (
        'ffmpeg -hide_banner -loglevel warning -r 5 -f image2 -i %%*.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" vid_b_inst_years%sto%s.mp4'
        % (
            matplotlib.dates.num2date(starting_t).strftime("%Y"),
            matplotlib.dates.num2date(ending_t).strftime("%Y"),
        )
    )

    print("\t\t\t%s." % cmd)
    cd = os.getcwd()
    os.chdir("%s/compactionvideo_%s" % (outputfolder, layer.replace(" ", "_")))
    subprocess.call(cmd, shell=True)

    if os.path.isfile(
        "vid_b_inst_years%sto%s.mp4"
        % (
            matplotlib.dates.num2date(starting_t).strftime("%Y"),
            matplotlib.dates.num2date(ending_t).strftime("%Y"),
        )
    ):
        print("\tVideo seems to have been a success; deleting excess .jpg files.")
        jpg_files_tmp = glob.glob("*.jpg")
        for file in jpg_files_tmp:
            os.remove(file)
        print("\tDone.")

    os.chdir(cd)
