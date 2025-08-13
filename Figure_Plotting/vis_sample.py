import matplotlib.pyplot as plt
import numpy as np

from utils import *
# kd-tree
from scipy.spatial import KDTree


def vis_track_p(df):
    # df = df[df["p"] < 200]
    # df = df[df["p"] > 10000]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6), projection='3d')

    pids = df["mcparticleID"].unique()

    # colors  by  "p"
    # max_p = np.log10(1 + df["p"].max())
    max_p = df["p"].max()
    min_p = df["p"].min()
    ax.scatter(0, 0, 000, c='r', marker='o', label="Origin")

    for pid in tqdm(pids):
        df_pid = df[df["mcparticleID"] == pid]
        xyz = df_pid[["x", "y", "z"]].values
        # p = np.log10(1 + df_pid["p"].values)
        p = df_pid["p"].values
        p = np.log10(1+(p - min_p)) / np.log10(1+(max_p - min_p))
        colors = plt.cm.jet(p)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.-', c=colors[0], alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=np.log10(1+(max_p - min_p))))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label("log10(1+p)")

    plt.title("Relation between track and momentum(p > 2500 MeV/c)")
    # view from top
    ax.view_init(elev=90, azim=0)

    plt.tight_layout()

    # orth projection
    ax.set_proj_type('ortho')

    # ax.legend()
    # fig.savefig("vis_track_p_cut2500.png", dpi=300)
    plt.show()


def vis_trk_hit_num(df):
    cut = 1500
    df = df[df["p"] < cut]


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    l0_with_hit = []
    l0_without_hit = []
    for pid in tqdm(df["mcparticleID"].unique()):
        df_pid = df[df["mcparticleID"] == pid]
        layer0 = df_pid[df_pid["layer"] == 0]
        l0_hit_num = layer0.shape[0]
        # print(l0_hit_num)

        hit_num = df_pid.shape[0]

        if l0_hit_num > 0:
            l0_with_hit.append(hit_num)

        l0_without_hit.append(hit_num)

    ax.hist(l0_without_hit, range=(1, 21), bins=50, alpha=1, label="Layer0 without hit", stacked=True)
    ax.hist(l0_with_hit, range=(1, 21), bins=50, alpha=1, label="Layer0 with hit")

    ax.set_xlabel("Hit number")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Hit number distribution(>{cut} MeV/c)")

    plt.legend()

    plt.tight_layout()
    # fig.savefig(f"hit_num_dist_{cut}.png", dpi=300)
    plt.show()


def vis_trk_angle(df):
    p_cut = 1500
    df = df[df["p"] > p_cut]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    pids = df["mcparticleID"].unique()
    thetas = []
    for pid in tqdm(pids):
        df_pid = df[df["mcparticleID"] == pid]
        if df_pid.shape[0] < 2:
            continue
        xyz = df_pid[["x", "y", "z"]].values

        xyz0 = xyz[0, :]
        xyz123 = xyz[:-1, :]
        cos_theta = np.dot(xyz123, xyz0) / (np.linalg.norm(xyz123, axis=1) * np.linalg.norm(xyz0))
        cos_theta[cos_theta > 1] = 1
        theta = np.arccos(cos_theta)
        # print(theta)
        theta = np.mean(theta)
        thetas.append(theta)

    ax.hist(thetas, bins=50, alpha=1)

    ax.set_xlabel("Angle")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Angle distribution(>{p_cut} MeV/c)")

    plt.tight_layout()
    fig.savefig(f"angle_dist_{p_cut}.png", dpi=300)
    plt.show()


def vis_trk_dist(df):
    p_cut = 2500
    r_cut_l = 50
    r_cut_h = 100
    df = df[df["p"] > p_cut]

    df = df[df["x"] ** 2 + df["y"] ** 2 < r_cut_h ** 2]
    df = df[df["x"] ** 2 + df["y"] ** 2 > r_cut_l ** 2]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    pids = df["mcparticleID"].unique()
    xyzs = []
    for pid in tqdm(pids):
        df_pid = df[df["mcparticleID"] == pid]
        if df_pid.shape[0] < 2:
            continue
        xyz = df_pid[["x", "y", "z"]].values
        xyz = np.mean(xyz, axis=0)
        # xyz = xyz[0, :]
        xyzs.append(xyz)

    xyzs = np.array(xyzs)

    tree = KDTree(xyzs)
    dists = tree.query(xyzs, k=2)[0][:, 1]
    # print(dists)

    max_dist = np.max(dists)
    min_dist = np.min(dists)
    # print(max_dist)
    # color = plt.cm.jet(dists / max_dist)

    ax.hist(dists, bins=50, alpha=1)

    for pid in tqdm(pids):
        df_pid = df[df["mcparticleID"] == pid]
        if df_pid.shape[0] < 2:
            continue
        xyz = df_pid[["x", "y", "z"]].values
        xyz_m = np.mean(xyz, axis=0)
        # xyz_m = xyz[0, :]

        dist = tree.query(xyz_m, k=2)[0][1]
        # print(dist / max_dist)
        colors = plt.cm.jet((dist - min_dist) / (max_dist - min_dist))
        ax2.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.-', c=colors, alpha=0.5)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min_dist, vmax=max_dist))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
    cbar.set_label("log10(1+p)")

    plt.title(f"track and dist(p > {p_cut} MeV/c)")
    # view from top
    ax2.view_init(elev=90, azim=0)

    plt.tight_layout()

    fig.savefig(f"dist_dist_{p_cut}.png", dpi=300)

    plt.show()


def main():
    raw_path = os.path.join(workdir, "RawData",
                            r"D:\files\pyproj\GNN\EvalNet\work_dir_with_large_smear\RawData\run5_mj_BO_clangenb_50eV_largerSmearing.csv")

    df = pd.read_csv(raw_path)
    df = df[df["eventID"] == 1]
    print(df["p"].describe())
    df_more_2500 = df[df["p"] > 0]

    print()

    # df_less_2500 = df[df["p"] < 1000]

    # print(df_more_2500.shape)
    # print(df_less_2500.shape)

    plot_hits(df_more_2500)
    # plot_hits(df_less_2500)
    # plot_hits(df)
    plt.show()

    pass


if __name__ == "__main__":
    # raw_path = r"D:\files\pyproj\GNN\EvalNet\work_dir_with_2_smear\RawData\run5_mj_BO_clangenb_500eV_smearing.csv"
    path = os.path.join(workdir, "RawData")
    path = os.listdir(path)
    raw_path = [os.path.join(workdir, "RawData", p) for p in path if p.endswith(".csv")][0]
    print(raw_path)

    df = pd.read_csv(raw_path)
    df = df[df["eventID"] == 1]
    df = df[df["p"] > 0]
    # counts = df["mcparticleID"].value_counts()
    # count2 = counts.value_counts()
    # print(count2, count2.sum())

    vis_track_p(df)
    # vis_trk_hit_num(df)




    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(df["x"].values, df["y"].values, df["z"].values, '.-')
    # print(df)
    # vis_track_p(df)
    # vis_trk_hit_num(df)
    # vis_trk_angle(df)
    # vis_trk_dist(df)


