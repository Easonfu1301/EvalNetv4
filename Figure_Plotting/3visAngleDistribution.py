from matplotlib import colors as mcolors
from tqdm import tqdm
from utils import *


def vis_trk_angle(df0, p_cut = 1500):

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    thetas = []
    for evt in tqdm(range(1, df0["eventID"].max() + 1), desc="Drawing angle"):
    # for evt in tqdm(range(1, 6), desc="Drawing angle"):
        df = df0[(df0["eventID"] == evt) & (df0["p"] > p_cut)]
        pids = df["mcparticleID"].unique()

        for pid in pids:
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

    hh = ax.hist(thetas, bins=50, alpha=1, log=True, histtype='step', color='blue', edgecolor='blue', linewidth=1)
    errs = np.sqrt(hh[0])

    centers = 0.5 * (hh[1][:-1] + hh[1][1:])

    ax.errorbar(centers, hh[0], yerr=errs, fmt='o', color='black', markersize=3, elinewidth=1, capsize=2)

    ax.set_xlabel(r"theta")
    ax.set_ylabel("Counts")
    ax.set_title(f"Angular separation between two hits in same track  (p > {p_cut} MeV/c)")



    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    # set inner minor ticks
    ax.tick_params(which='major', direction='in', length=6, width=1.5, colors='black', grid_color='gray', grid_alpha=0.5)
    ax.tick_params(which='minor', direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.5)

    # set frame width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()



    plt.tight_layout()
    fig.savefig(os.path.join(workdir, "Plot", f"angle_rel_Ori_{p_cut}.png"), dpi=300)
    # plt.show()

def main() -> None:
    raw_dir = os.path.join(workdir, "RawData")
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No CSV files found in RawData/")

    df = pd.read_csv(os.path.join(raw_dir, raw_files[0]))
    # vis_trk_angle(df)
    # vis_trk_angle(df, 0)
    vis_trk_angle(df, 1500)


if __name__ == "__main__":
    main()
