from matplotlib import colors as mcolors
from tqdm import tqdm
from utils import *

def vis_trk_hit_num(df0, p_cut: float = 1500) -> None:
    """Visualise the x–y projection of tracks coloured by momentum.

    Replaces the ``LogNorm`` colour‐scale with a ``1 + \log_{10}`` transformation
    so the colour bar still reflects the *actual* momentum (MeV/c) values but the
    mapping is handled explicitly rather than via ``matplotlib.colors.LogNorm``.
    """
    # Keep only the first event and tracks above the requested p cut

    l0_with_hit = []
    l0_without_hit = []
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for evt in tqdm(range(1, df0["eventID"].max() + 1), desc="Drawing tracks"):
        df = df0[(df0["eventID"] == evt) & (df0["p"] > p_cut)]
        for pid in df["mcparticleID"].unique():
            df_pid = df[df["mcparticleID"] == pid]
            layer0 = df_pid[df_pid["layer"] == 0]
            l0_hit_num = layer0.shape[0]
            # print(l0_hit_num)

            hit_num = df_pid.shape[0]

            if l0_hit_num > 0:
                l0_with_hit.append(hit_num)

            l0_without_hit.append(hit_num)

    ax.hist(l0_without_hit, range=(1, 16), bins=30, alpha=1, label="Layer 0 without hit", stacked=True)
    ax.hist(l0_with_hit, range=(1, 16), bins=30, alpha=1, label="Layer 0 with hit")

    ax.set_xlabel("Hit number")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Hit number distribution(>{p_cut} MeV/c)")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    # set inner minor ticks
    ax.tick_params(which='major', direction='in', length=6, width=1.5, colors='black', grid_color='gray', grid_alpha=0.5)
    ax.tick_params(which='minor', direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.5)

    # set frame width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()

    plt.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(workdir, "Plot", f"hit_num_dist_{p_cut}.png"), dpi=300)
    plt.show()

def main() -> None:
    raw_dir = os.path.join(workdir, "RawData")
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not raw_files:
        raise FileNotFoundError("No CSV files found in RawData/")

    df = pd.read_csv(os.path.join(raw_dir, raw_files[0]))
    vis_trk_hit_num(df)
    vis_trk_hit_num(df, 0)
    vis_trk_hit_num(df, 50000)


if __name__ == "__main__":
    main()
