import matplotlib.pyplot as plt
import numpy as np

from utils import *
from scipy.stats import binom

alpha = 0.05


def get_p_efficiency(restore_df0, bin):
    restore_df = restore_df0.copy()

    data = np.nan * np.zeros((bin, 12))
    P_MIN = 0 * 1e3
    P_MAX = 50 * 1e3

    P = np.linspace(P_MIN, P_MAX, bin)
    # P = np.exp( np.linspace(np.log(P_MIN), np.log(P_MAX), bin))
    # restore_df = restore_df[(restore_df["feta"] > 2) & (restore_df["feta"] < 5)]


    # restore_df = restore_df[(restore_df["kind"] == 11) | (restore_df["kind"] == -11)]
    # restore_df = restore_df[(restore_df["kind"] != 11) & (restore_df["kind"] != -11)]
    # restore_df = restore_df[(restore_df["isPrim"] == True) | (restore_df["isDecay"] == True)]
    # K_S
    # print(np.array(restore_df['kind'].unique(), dtype=np.int32))
    # restore_df = restore_df[(restore_df["kind"] == 321)]




    for idx in range(bin - 1):
        p_low = P[idx]
        p_high = P[idx + 1]

        df_recon = restore_df[(restore_df["p"] >= p_low) & (restore_df["p"] < p_high)]
        df_recon0 = restore_df0[(restore_df0["p"] >= p_low) & (restore_df0["p"] < p_high)]



        # df_recon = restore_df[(restore_df["p"] < p_high)]
        # df_recon = restore_df[restore_df["p"] >= p_low]
        # df_recon0 = restore_df0[restore_df0["p"] >= p_low]
        data[idx, 0] = len(df_recon0)
        data[idx, 1] = df_recon["recon_count"].sum()
        data[idx, 2] = df_recon["if_recon"].sum()
        # data[idx, 2] = df_recon["success_times"].sum()
        data[idx, 3] = data[idx, 1] - data[idx, 2]
        data[idx, 4] = data[idx, 3] / data[idx, 1]  # ghost rate

        lower_bound_frac = binom.ppf(alpha / 2, data[idx, 0], data[idx, 4]) / data[idx, 0]
        upper_bound_frac = binom.ppf(1 - alpha / 2, data[idx, 0], data[idx, 4]) / data[idx, 0]

        data[idx, 5] = lower_bound_frac
        data[idx, 6] = upper_bound_frac

        df_recon = df_recon[df_recon["total"] > 2]
        data[idx, 7] = len(df_recon)
        data[idx, 8] = len(df_recon[df_recon["if_recon"] == 1])
        data[idx, 9] = data[idx, 8] / data[idx, 7]



        lower_bound_frac = binom.ppf(alpha / 2, data[idx, 7], data[idx, 9]) / data[idx, 7]
        upper_bound_frac = binom.ppf(1 - alpha / 2, data[idx, 7], data[idx, 9]) / data[idx, 7]

        data[idx, 10] = lower_bound_frac
        data[idx, 11] = upper_bound_frac

    return P, data


def plot_efficiency(restore_df):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax2 = ax.twinx()
    P, data = get_p_efficiency(restore_df, 100)

    P = P / 1000

    # ax.plot(P, data[:, 1], ".-", label="Efficiency")

    ax.errorbar(P, data[:, 4], yerr=[data[:, 4] - data[:, 5], data[:, 6] - data[:, 4]], fmt='.-', color='r',
                ecolor='b', elinewidth=2, capsize=4, label="Ghost rate")
    ax.errorbar(P, data[:, 9], yerr=[data[:, 9] - data[:, 10], data[:, 11] - data[:, 9]], fmt='.-', color='y',
                ecolor='gray', elinewidth=2, capsize=4, label="Efficiency")

    # ax.plot(P, data[:, 7], "g", label="False recon count / right recon count")

    ax2.plot(P, data[:, 0], ".-", label="Trk count total")
    ax2.plot(P, data[:, 7], ".-", label="Reconstruct-able( >2 hit )")
    ax2.plot(P, data[:, 8], ".-", label="Reconstruct-able and successfully reconstructed")

    # set minor ticks on
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    # set 0-1
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Momentum [GeV/c]")
    ax.set_ylabel("Ghost rate & Efficiency")
    ax2.set_ylabel("Count")
    # ax set log
    ax2.set_yscale("log")
    # ax.set_xscale("log")

    # legend best location
    # ❶ 让右边空出一点位置给图例
    fig.subplots_adjust(right=0.78)  # 0.78 可按需要微调

    # ❷ 把 ax 的图例放到右上角外侧
    ax.legend(
        loc='upper left',  # 图例自身的锚点
        bbox_to_anchor=(1.1, 1.00),  # (x>1 把它推到外面)
        borderaxespad=0.
    )

    # ❸ 把 ax2 的图例放到右侧中部外侧
    ax2.legend(
        loc='center left',
        bbox_to_anchor=(1.1, 0.70),
        borderaxespad=0.
    )

    path = os.path.join(workdir, "Eval", "BackResult")
    len_path = len(os.listdir(path))
    plt.title(f"Efficiency, Ghost rate and Count (with {len_path} evt)")

    plt.tight_layout()
    save_path = os.path.join(workdir, "Eval", f"Efficiency.png")
    fig.savefig(save_path, dpi=300)
    gprint(f"Efficiency, Ghost rate and Count plot saved in ", end="")
    yprint(save_path)
    # plt.show()
    # plt.plot(data[:, 4], data[:, 9], ".", label="efficiency vs ghost rate")
    # plt.xlabel("Ghost rate")
    # plt.ylabel("Efficiency")
    plt.show()


    #### save the data to csv ######
    data = np.hstack((P.reshape(-1, 1), data))
    data_df = pd.DataFrame(data, columns=["p", "total", "recon_count", "if_recon",
                                         "ghost", "ghost_rate",
                                         "ghost_lower_band", "ghost_higher_band", "reconstructable_count",
                                         "successful_recon", "efficiency","efficiency_lower_band", "efficiency_higher_band"])

    data_df.to_csv(os.path.join(workdir, "Eval", "Efficiency-table.csv"), index=False)









def main():
    gprint("Visualizing Efficiency, Ghost rate and Count...")

    restore_df_path = os.path.join(workdir, "Eval", "Efficiency.csv")
    restore_df = pd.read_csv(restore_df_path)
    plot_efficiency(restore_df)


if __name__ == "__main__":
    main()
    # df = pd.read_csv(os.path.join(workdir, "Eval", "Efficiency.csv"))
    # print("------------------- p all -------------------")
    #
    # print(df["total"].value_counts(), "%")
    # print("------------------- p > 5GeV -------------------")
    #
    # df = df[df["p"] > 5000]
    # print(df["total"].value_counts() , "%")
    # print("------------------- p > 50GeV -------------------")
    #
    # df = df[df["p"] > 50000]
    # print(df["total"].value_counts() , "%")
    #
    # print("------------------- p > 100GeV -------------------")
    #
    # df = df[df["p"] > 100000]
    # print(df["total"].value_counts(), "%")
    #
    #
    # print("------------------- p > 200GeV -------------------")
    # df = df[df["p"] > 200000]
    # print(df["total"].value_counts(), "%")