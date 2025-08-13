from utils import *





def get_hit_ratio(evt_df):

    pids = evt_df["mcparticleID"].unique()
    # hit_ratio_df = pd.DataFrame(columns=["mcparticleID", "p", "hit_num"])
    hit_ratio_df = []

    for pid in pids:
        trk_df = evt_df[evt_df["mcparticleID"] == pid]

        hit_ratio_df.append({
            "mcparticleID": pid,
            "p": trk_df["p"].mean() / 1000,
            "hit_num": len(trk_df)
        })

    hit_ratio_df = pd.DataFrame(hit_ratio_df)

    return hit_ratio_df


def gen_color_by_hit_num(hit_num):
    # generate color by hit number from 1-40 not continue
    if hit_num == 1:
        return "red"
    elif hit_num == 2:
        return "blue"
    elif hit_num == 3:
        return "green"
    elif hit_num == 4:
        return "orange"
    elif hit_num == 5:
        return "purple"
    elif hit_num == 6:
        return "black"
    elif hit_num == 7:
        return "pink"
    elif hit_num == 8:
        return "yellow"
    elif hit_num == 9:
        return "brown"
    elif hit_num == 10:
        return "gray"
    else:
        return "white"




def plot_hit_ratio(hit_ratio_df):
    # draw bar for hit ratio in each momentum bin
    fig, ax = plt.subplots(figsize=(6, 4))

    p_min = 0
    p_max = 50

    p_bins = np.linspace(p_min, p_max, 100)


    for i in range(len(p_bins) - 1):
        p_low, p_high = p_bins[i], p_bins[i + 1]
        hit_ratio = hit_ratio_df[(hit_ratio_df["p"] >= p_low) & (hit_ratio_df["p"] < p_high)]
        # hit_ratio = hit_ratio_df[hit_ratio_df["p"] >= p_low]
        hit_ratio = hit_ratio["hit_num"].value_counts()
        hit_ratio = hit_ratio[hit_ratio.index <= 10]
        hit_ratio = hit_ratio / hit_ratio.sum()
        hit_ratio = hit_ratio.sort_index()

        bottom = 0
        for hit_num, ratio in hit_ratio.items():

            ax.bar(p_low, ratio, width=p_high - p_low, bottom = bottom, align="edge", label=f"HitNum: {hit_num}", color=gen_color_by_hit_num(hit_num))
            bottom += ratio




    ax.set_xlabel("Momentum (GeV)")
    ax.set_ylabel("Hit ratio")

    # no repeat label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # outside legend
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Hit Ratio vs Momentum")
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, "Plot", "HitRatio.png"), dpi=600)


    plt.show()


def main():
    raw_dir = os.path.join(workdir, "RawData")
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No CSV files found in RawData/")

    data_df = pd.read_csv(os.path.join(raw_dir, raw_files[0]))
    hit_ratio_df_list = []
    for i in tqdm(range(1, data_df["eventID"].max() + 1)):
    # for i in tqdm(range(1, 5)):
        # data_df[data_df["eventID"] == i]
        hit_ratio_df_list.append(get_hit_ratio(data_df[data_df["eventID"] == i]))

    hit_ratio_df = pd.concat(hit_ratio_df_list)
    plot_hit_ratio(hit_ratio_df)

if __name__ == "__main__":
    main()


