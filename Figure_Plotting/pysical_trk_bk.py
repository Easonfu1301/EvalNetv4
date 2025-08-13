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
    :param hits: xz
    :return: direction and chi2
    '''
    x = hits[:, 0]
    z = hits[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, z, rcond=None)[0]
    chi2 = np.sum((k * x + b - z) ** 2)
    return k, b, chi2



def cal_angle(k, b):
    theta = np.arctan(1/k)
    return np.degrees(theta)



def get_theta_array(df):
    theta_array = []

    for idx, evt in tqdm(enumerate(df["eventID"].unique()), total=df["eventID"].nunique()):
        df_evt = df[df["eventID"] == evt]
        for hit_id in df_evt["mcparticleID"].unique():
            hit = df_evt[df_evt["mcparticleID"] == hit_id]
            if hit.shape[0] < 2:
                continue




            hits = hit[["x", "z"]].values
            # hits2 = hit[["x", "y", "z"]].values
            #
            # x_mean = np.abs(np.mean(hits2[:, 0]))
            # y_mean = np.abs(np.mean(hits2[:, 1]))
            #
            # if x_mean < 60 and y_mean < 60:
            #     continue
            #
            # if x_mean > 500 or y_mean > 500:
            #     continue


            # hits[:, 0] = np.sqrt(hits[:, 0] ** 2 + 0* hits[:, 1] ** 2)
            # hits = hits[:, [0, 2]]

            # print(hits.shape)

            k, b, chi2 = fit_chi2(hits)
            theta = cal_angle(k, b)
            theta_array.append(theta)

            # if idx > 5:
            #     break

    return np.array(theta_array)

def plot_true_trk_ditribution(df, bin):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df = df[(df["isDecay"] == 1) | (df["isPrim"] == 1)]
    # df = df[(df["isDecay"] == 1)]
    # df = df[df["isPrim"] == 1]
    # df = df[df["p"] > 2500]

    theta_array = get_theta_array(df)
    print(theta_array)

    theta_abs = np.abs(theta_array)
    theta_abs_sort = np.sort(theta_abs)
    print(theta_abs_sort)
    # ax.plot(theta_abs_sort, label="True Track angle Distribution")
    print(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.9)])
    print(theta_abs_sort[int(theta_abs_sort.shape[0] * 0.99)])

    ax.hist(theta_array, bins=bin, log=False, color="k", alpha=0.3, label="True Track angle Distribution")

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