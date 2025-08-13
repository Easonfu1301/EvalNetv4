import numpy as np


SMEAR_N_X = 50 / 1000 / np.sqrt(12)
SMEAR_N_Y = 150 / 1000 / np.sqrt(12)

# workdir = r"D:\files\pyproj\GNN\formal_test\work_dir_0x0um2"
# workdir = r"D:\files\pyproj\GNN\EvalNetv2\workdir_MOORE_test"
# workdir = r"D:\files\pyproj\GNN\EvalNetv2\workdir_MOORE_test_with_moredata"
# workdir = r"D:\files\pyproj\GNN\v9_TEST\v9_best"
# workdir = r"D:\files\pyproj\GNN_Diff\diff_v1\workdir" # FYS: 我忘了干啥用的了
# workdir = r"D:\files\pyproj\GNN_bk_test\workdir" # no shuffle
workdir = fr"D:\files\pyproj\UP-newDesign\x_{SMEAR_N_X*1000*np.sqrt(12):.0f}_y_{SMEAR_N_Y*1000*np.sqrt(12):.0f}" # with shuffle



EVT_NUM = 9999




CUT_L0 = 0.
CUT_L1 = 0.
CUT_L2 = 0.

COMBINE_EVT = 1


ITER_TIME = 5 # default to 5
ITER_MULTI = 2 # default to 2
JUDGE_PRESERVE_RATE = 0.3 # default to 0.3


EVAL_PROCESSOR = 10 # default to 10
CHECKER_PROCESSOR = 24 # default to 16






