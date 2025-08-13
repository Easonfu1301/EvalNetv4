from .setting import *
import os
from .cprint import *
import sys





dirs  = [
    "RawData",
    "PreProcess",
    os.path.join("PreProcess", "csv_with_hits"),
    os.path.join("PreProcess", "csv_with_tracks"),
    "Model",
    "Eval",
    os.path.join("Eval", "BackPre"),
    os.path.join("Eval", "BackResult"),
    os.path.join("Eval", "BackResultConvert"),
    "Plot",
]


def mkdirs():
    FLAG = False
    if not os.path.exists(workdir):
        FLAG = True
        os.makedirs(workdir, exist_ok=True)



    for dir in dirs:
        if not os.path.exists(os.path.join(workdir, dir)):
            os.makedirs(os.path.join(workdir, dir), exist_ok=True)

    if FLAG:

        gprint("Create workdir: ", end="")
        yprint(workdir)
        bprint("Should Do Next: The workdir is empty now., Please put the data in the workdir and run the code again.")

        exit(0)
