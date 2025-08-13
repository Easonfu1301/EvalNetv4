import matplotlib.pyplot as plt

from utils import *





def get_trk_x(evt_df):
    pids = evt_df["mcparticleID"].unique()
    # trk_xy_df = pd.DataFrame(columns=["mcparticleID", "p", "r"])
    trk_xy_df = []


    for pid in pids:
        trk_df = evt_df[evt_df["mcparticleID"] == pid]
        trk_xy_df.append({
            "mcparticleID": pid,
            "p": trk_df["p"].mean() / 1000,
            "r": np.sqrt(trk_df["x"].mean()**2 + trk_df["y"].mean()**2)
        })

    trk_xy_df = pd.DataFrame(trk_xy_df)

    return trk_xy_df



def plot_trk_x(trk_xy_df):
    fig, ax = plt.subplots()

    # plot avg r vs momentum
    p_min = 0
    p_max = 20
    r_min = 0
    r_max = 800

    p_bins = np.linspace(p_min, p_max, 6)
    r_bins = np.linspace(r_min, r_max, 35)

    trk_xy_df = trk_xy_df[(trk_xy_df["p"] >= 0) & (trk_xy_df["p"] < 20)]
    trk_xy_df = trk_xy_df[(trk_xy_df["r"] >= 0) & (trk_xy_df["r"] < 900)]

    p_r_avg = np.zeros((len(p_bins) - 1, len(r_bins) - 1))

    for i in range(len(p_bins) - 1):
        p_low, p_high = p_bins[i], p_bins[i + 1]
        df_p = trk_xy_df[(trk_xy_df["p"] >= p_low) & (trk_xy_df["p"] < p_high)]
        # df_p = trk_xy_df[(trk_xy_df["p"] >= p_low)]
        for j in range(len(r_bins) - 1):
            r_low, r_high = r_bins[j], r_bins[j + 1]
            p_r_avg[i, j] = df_p[(df_p["r"] >= r_low) & (df_p["r"] < r_high)].shape[0]

    err_r = np.sqrt(p_r_avg)
    p_r_avg_sum = np.sum(p_r_avg, axis=1)[:, None]
    p_r_avg = p_r_avg / p_r_avg_sum
    err_r = err_r / p_r_avg_sum
    print("p_r_avg shape:", p_r_avg.shape)
    print("err_r shape:", err_r.shape)


    # for i in p_bins:
    # ax.plot(r_bins[:-1], p_r_avg.T, ".-", label=[f"p = {pp:.1f} GeV" for pp in p_bins[:-1]])

    for i in range(len(p_bins) - 1):
        ax.errorbar(r_bins[:-1], p_r_avg[i, :], yerr=err_r[i, :], fmt='.-', elinewidth=2, capsize=4, label=f"p = {p_bins[i]:.1f} GeV")

    # ax.plot(p_bins[:-1], r_avg, marker="o")
    ax.set_xlabel("r (mm)")
    # ax.set_ylabel("Fraction of tracks of radius")
    ax.title.set_text("Track distribution of radius in different momentum")
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    # set inner minor ticks
    ax.tick_params(which='major', direction='in', length=6, width=1.5, colors='black', grid_color='gray',
                   grid_alpha=0.5)
    ax.tick_params(which='minor', direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.5)

    # set frame width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, "Plot", "trk_r_vs_p.png"), dpi=700)

    plt.show()

if __name__ == "__main__":
    raw_dir = os.path.join(workdir, "RawData")
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No CSV files found in RawData/")

    data_df = pd.read_csv(os.path.join(raw_dir, raw_files[0]))
    hit_ratio_df_list = []
    for i in tqdm(range(1, data_df["eventID"].max() + 1)):
    # for i in tqdm(range(1, 21)):
        # data_df[data_df["eventID"] == i]
        hit_ratio_df_list.append(get_trk_x(data_df[data_df["eventID"] == i]))


    trk_xy_df = pd.concat(hit_ratio_df_list)

    plot_trk_x(trk_xy_df)