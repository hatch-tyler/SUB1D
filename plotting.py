import os
import sys
import time
import glob
import datetime
import matplotlib
import subprocess

import numpy as np
import seaborn as sns

from utils import printProgressBar


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
