import os
import sys
import time
import shutil
import tempfile
import matplotlib


class Logger(object):
    def __init__(self, output_folder, run_name):
        self.terminal = sys.stdout
        self.log = open("%s/%s/logfile.log" % (output_folder, run_name), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def __enter__(self):
        # sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.flush()
        self.log.close()
        sys.stdout = self.terminal


def with_retries(
    func, max_retries=3, delay=1
):  # I added this function to help facilitate running from server. Sometimes file locks can cause issues otherwise.
    for i in range(max_retries):
        try:
            return func()
        except PermissionError as e:
            print(f"Permission error on attempt {i+1}: {e}")
            time.sleep(delay)
    raise PermissionError(f"Failed after {max_retries} attempts.")


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    # print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_size(fig, dpi=100):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
        height, width, _channels = matplotlib.image.imread(f.name).shape
        return width / dpi, height / dpi


def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height  # reasonable starting point
    deltas = []  # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(
            abs(actual_width - target_width) + abs(actual_height - target_height)
        )
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False


def make_output_folder(param_filename, outdestination, overwrite, OverwriteYes):
    try:
        if not os.path.isdir(outdestination):
            print(
                "\t\tMaking output directory at %s. This stdout will now be directed to a log in that folder as well as displayed here."
                % (outdestination)
            )
            with_retries(lambda: os.mkdir(outdestination))
            with_retries(lambda: os.mkdir("%s/figures" % outdestination))
            with_retries(lambda: os.mkdir("%s/s_outputs" % outdestination))
            with_retries(lambda: os.mkdir("%s/head_outputs" % outdestination))
        else:
            if overwrite:
                check = OverwriteYes
                # check=input('\t\tOutput folder %s already exists and overwrite flag specified. Do you want to overwrite? (Y/N):  ' % outdestination)
                check = check.strip().lower()
                if check == "y":
                    print("Overwriting folder")
                    backup_folder = "%s_old" % outdestination
                    if os.path.isdir(backup_folder):
                        print(
                            "There is another "
                            + "%s_old" % outdestination
                            + " directory. Removing..."
                        )
                        with_retries(lambda: shutil.rmtree(backup_folder))
                        time.sleep(1)  # Give some time after removing the directory
                    with_retries(
                        lambda: shutil.copytree(outdestination, backup_folder)
                    )  # I changed this line because it seemed like it wasn't working...
                    with_retries(lambda: shutil.rmtree(outdestination))
                    time.sleep(1)  # Give some time after removing the directory
                    with_retries(lambda: os.mkdir(outdestination))
                    with_retries(lambda: os.mkdir("%s/figures" % outdestination))
                    with_retries(lambda: os.mkdir("%s/s_outputs" % outdestination))
                    with_retries(lambda: os.mkdir("%s/head_outputs" % outdestination))
                else:
                    print("\t\tNot overwriting. Aborting.")
                    sys.exit(1)
            else:
                print(
                    "\t\tAdmin error: terminal. Output folder %s already exists and overwrite flag not specified."
                    % outdestination
                )
                sys.exit(1)
    except PermissionError as e:
        print(
            f"Permission error: {e}. Please ensure no files or folders are open or being used by another process."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    #time.sleep(2)  # Sleep for 2 seconds before moving the log file
    #with_retries(lambda: shutil.move("logfile.log", "%s/logfile.log" % outdestination))
    if os.path.exists(param_filename):
        shutil.copy2(param_filename, "%s/paramfile.par" % outdestination)
    else:
        if param_filename.split("/")[-1] == "paramfile.par":
            print("\tAssuming this is a rerun of old run; copying paramfile over.")
            shutil.copy2(
                "%s_old/paramfile.par" % outdestination,
                "%s/paramfile.par" % outdestination,
            )
        else:
            print("\tSomething has gone wrong setting up output directories, ABORT.")
            sys.exit(1)
