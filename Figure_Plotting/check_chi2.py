import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('tkAgg')



def check_chi2(path):
    df = pd.read_csv(path)
    df = df[df["eventID"].isin([i for i in range(1, 50)])]


    chi2s = []

    # fig = plt.figure()
    for i in tqdm(range(1, 50)):
        df2 = df[df["eventID"] == i]
        pids = df2["mcparticleID"].unique()
        for pid in pids:
            xyz = df2[df2["mcparticleID"] == pid][["x", "y", "z"]].values

            # check chi2 of straight line
            centroid = np.mean(xyz, axis=0)

            # 去中心化
            centered_points = xyz - centroid

            # 使用奇异值分解 (SVD) 找到方向向量
            _, _, vh = np.linalg.svd(centered_points)
            direction_vector = vh[0]  # 直线的方向向量

            # 点到直线的距离
            distances = np.linalg.norm(np.cross(centered_points, direction_vector), axis=1) / np.linalg.norm(
                direction_vector)

            # 计算 chi^2 (假设误差 sigma_i=1)
            chi2 = np.sum(distances ** 2)
            chi2s.append(chi2)

        # print(pid, chi2)
    print("mean chi2:", len(chi2s), np.mean(chi2s), np.std(chi2s), np.median(chi2s))

    plt.hist(chi2s, bins=100, range=(0, 0.001), log=True, alpha=0.5, label=path.split("\\")[-1])





def main():
    check_chi2(r"D:\files\pyproj\GNN\EvalNet\work_dir_with_50_smear\RawData\run5_mj_BO_clangenb_500eV_smearing.csv")
    check_chi2(r"D:\files\pyproj\GNN\EvalNet\work_dir_with_50_smear\RawData\run5_mj_BO_clangenb_500eV_smearing.csv")
    check_chi2(r"D:\files\pyproj\GNN\EvalNet\work_dir_with_2_smear\RawData\run5_mj_BO_clangenb_500eV_smearing.csv")
    check_chi2(r"D:\files\pyproj\GNN\EvalNet\work_dir_with_5_smear\RawData\run5_mj_BO_clangenb_500eV_smearing.csv")
    plt.legend()
    plt.show()

    pass




if __name__ == "__main__":
    main()