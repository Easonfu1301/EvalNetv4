import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from .mkdirs import mkdirs
import matplotlib
matplotlib.use('tkAgg')
import uproot
import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils.setting import *
from .print_deco import *
from .plot_hits import *
from .construct_sample_var import construct_sample_var
# from .construct_sample_var_dict import construct_sample_var as construct_sample_var_dict
from .cprint import *
from .PARAM import  *
from .calc_chi2 import *

import numba
from numba import jit
from .time_analysis import timer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mkdirs()
matplotlib.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman'],
    'mathtext.fontset' : 'cm',           # 数学符号继续用 Computer Modern
    'mathtext.rm'      : 'Times New Roman',
    'axes.titlesize': 18,  # 主标题 plt.title
    'axes.labelsize': 14,  # x/y 轴标题 plt.xlabel / plt.ylabel
    'xtick.labelsize': 12,  # x 轴刻度
    'ytick.labelsize': 12,  # y 轴刻度
    'legend.fontsize': 12,  # 图例
})