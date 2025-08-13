import time

import os
import shutil
import subprocess
import sys
import numpy as np


def change_params(x: int, y: int):
    """
    修改参数
    :param x: 参数x
    :param y: 参数y
    :return: None
    """
    # print(setting, type(setting))
    with open(os.path.join("utils", "setting.py"), "r") as f:
        content = f.read()

    # replace line with EVT_NUM and SMEAR_N_X, SMEAR_N_Y
    for line in content.splitlines():
        # print(line)
        if "SMEAR_N_X = " in line:
            content = content.replace(line, f"SMEAR_N_X = {x} / 1000 / np.sqrt(12)")
            print(f"SMEAR_N_X = {x} / 1000 / np.sqrt(12)")
        if "SMEAR_N_Y = " in line:
            content = content.replace(line, f"SMEAR_N_Y = {y} / 1000 / np.sqrt(12)")
            print(f"SMEAR_N_Y = {y} / 1000 / np.sqrt(12)")


    with open(os.path.join("utils", "setting.py"), "w") as f:
        f.write(content)







if __name__ == "__main__":

    # for xx in [50, 75, 85, 100]:#, 125, 150, 200]:
    for xx in [250, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
        for yy in [500, 1000, 1500, 2000, 3000, 4000, 5000]:
            print(f"正在处理 x={xx}, y={yy}")

            change_params(xx, yy)

            subprocess.run([sys.executable, "-m", "pipeline_v5"], check=True)

            workdir = fr"D:\files\pyproj\GNN_bk_test\resol\x_{xx}_y_{yy}"  # with shuffle

            shutil.copy(r"D:\files\pyproj\GNN_bk_test\baseline_tuple.root", os.path.join(workdir, "RawData", "baseline_tuple.root"))
            shutil.copy(r"D:\files\pyproj\GNN_bk_test\filtering_model.pth", os.path.join(workdir, "Model", "filtering_model.pth"))

            subprocess.run([sys.executable, "-m", "pipeline_v5"], check=True)

