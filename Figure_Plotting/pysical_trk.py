import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')


def vis_hit(df, ax, color="r"):
    xyz = df[["x", "y", "z"]].values
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=color)
    return ax


def fit_chi2(hits):
    '''
    :param hits: xyz
    :return: direction vec and chi2 by line fit
    '''
    centroid = np.mean(hits, axis=0)

    # 中心化数据
    centered = hits - centroid

    # 计算协方差矩阵
    cov_matrix = np.cov(centered, rowvar=False)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 取最大特征值对应的特征向量作为方向
    direction_vector = eigenvectors[:, np.argmax(eigenvalues)]

    # 确保方向向量单位化
    direction_vector /= np.linalg.norm(direction_vector)

    return direction_vector, centroid






def cal_angle(direction):

    direction = direction[[0, 2]]
    dir2 = [0, 1]

    cos_theta = np.dot(direction, dir2) / (np.linalg.norm(direction) * np.linalg.norm(dir2))
    theta = np.arccos(np.abs(cos_theta))
    return np.degrees(theta)



def get_theta_array(df):
    theta_array = []

    for idx, evt in tqdm(enumerate(df["eventID"].unique()), total=df["eventID"].nunique()):
        df_evt = df[df["eventID"] == evt]
        for hit_id in df_evt["mcparticleID"].unique():
            hit = df_evt[df_evt["mcparticleID"] == hit_id]
            if hit.shape[0]  < 2:
                continue


            # hits = hit[["y", "z"]].values
            hits = hit[["x", "y", "z"]].values
            # hits[:, 0] = np.sqrt(hits[:, 0] ** 2 + 0* hits[:, 1] ** 2)
            # hits = hits[:, [0, 2]]

            # print(hits.shape)


            direction, centroid = fit_chi2(hits)
            theta = cal_angle(direction)

            theta_array.append(theta)

            # if idx > 10:
            #     break

    return np.array(theta_array)

def plot_true_trk_ditribution(df, bin):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df = df[(df["isDecay"] == 1)]
    # df = df[df["isPrim"] == 0]
    # df = df[df["p"] > 2500]

    theta_array = get_theta_array(df)
    print(theta_array)

    theta_abs = np.abs(theta_array)
    theta_abs_sort = np.sort(theta_abs)
    print(theta_abs_sort)
    # ax.plot(theta_abs_sort, label="True Track angle Distribution")
    print(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.9)])
    print(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.95)])

    ax.hist(theta_array, bins=bin, color="k", alpha=0.3, label="True Track angle Distribution")

    # draw 90% and 95% line
    ax.axvline(x=theta_abs_sort[int(theta_abs_sort.shape[0] * 0.9)], color="r", label="90%, = {:.2f}".format(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.9)]))
    ax.axvline(x=theta_abs_sort[int(theta_abs_sort.shape[0] * 0.95)], color="b", label="95%, = {:.2f}".format(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.95)]))
    ax.axvline(x=-theta_abs_sort[int(theta_abs_sort.shape[0] * 0.9)], color="r")
    ax.axvline(x=-theta_abs_sort[int(theta_abs_sort.shape[0] * 0.95)], color="b")

    plt.legend()

    ax.set_xlabel("Theta(°)")
    ax.set_ylabel("Count")

    plt.show()






    # print(df)







if __name__ == "__main__":
    df = pd.read_csv(r"D:\files\pyproj\GNN\formal_test\work_dir_0x0um\RawData\0x0.csv")
    plot_true_trk_ditribution(df, 100)