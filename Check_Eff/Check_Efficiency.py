import pandas as pd

from utils import *


def raw_trk_df(df_raw):
    pids = df_raw["mcparticleID"].unique()

    raw_trk_df = pd.DataFrame(
        columns=["mcparticleID", "total", "p", "found_min", "found_max", "if_recon", "recon_count", "success_times"])

    for idx, pid in enumerate(pids):
        raw_trk_df.loc[idx, "mcparticleID"] = pid
        raw_trk_df.loc[idx, "total"] = len(df_raw[df_raw["mcparticleID"] == pid])

        # # exclude the track without layer 3 (since it is not possible to be reconstructed by the current algorithm)
        track_df = df_raw[df_raw["mcparticleID"] == pid]
        # print(track_df["layer"].unique())

        # raw_trk_df.loc[idx, "total"] = len(track_df) if len(track_df[track_df["layer"]==3]) > 0 else -1

        unique_layers = track_df["layer"].unique()
        # judge if the track has 01 and 23
        judge_01 = 0 in unique_layers or 1 in unique_layers
        judge_23 = 2 in unique_layers or 3 in unique_layers
        judgee = judge_01 and judge_23

        raw_trk_df.loc[idx, "total"] = len(track_df) if ((len(track_df[track_df["layer"] == 3]) > 0) & judgee) else -1

        raw_trk_df.loc[idx, "p"] = df_raw[df_raw["mcparticleID"] == pid]["p"].values[0]
        raw_trk_df.loc[idx, "feta"] = df_raw[df_raw["mcparticleID"] == pid]["feta"].values[0]
        raw_trk_df.loc[idx, "kind"] = df_raw[df_raw["mcparticleID"] == pid]["kind"].values[0]
        raw_trk_df.loc[idx, "found_min"] = 99999
        raw_trk_df.loc[idx, "found_max"] = 0
        raw_trk_df.loc[idx, "if_recon"] = 0
        raw_trk_df.loc[idx, "recon_count"] = 0
        raw_trk_df.loc[idx, "success_times"] = 0
        raw_trk_df.loc[idx, "isPrim"] = df_raw[df_raw["mcparticleID"] == pid]["isPrim"].values[0]
        raw_trk_df.loc[idx, "isDecay"] = df_raw[df_raw["mcparticleID"] == pid]["isDecay"].values[0]

    # print(raw_trk_df)
    return raw_trk_df


def eval_efficiency(df_raw, trk_pred):
    raw_trk = raw_trk_df(df_raw)

    for idx in range(trk_pred.shape[0]):
        trk = trk_pred[idx, :]
        pid0 = int(df_raw.loc[trk[0], "mcparticleID"])
        pid1 = int(df_raw.loc[trk[1], "mcparticleID"])
        pid2 = int(df_raw.loc[trk[2], "mcparticleID"])
        pid3 = int(df_raw.loc[trk[3], "mcparticleID"])

        judges = [trk[4] > CUT_L0, trk[5] > CUT_L1, trk[6] > CUT_L2]
        if judges.count(True) < 2:
            continue

        pids = np.array([pid0, pid1, pid2, pid3])
        main_pid = pid3
        # print(pids, main_pid, pids==main_pid)
        found = list(pids == main_pid).count(True)
        # print(found, pids, main_pid, pids==main_pid)
        # print(raw_trk[raw_trk["mcparticleID"] == main_pid]["found_min"].values[0], found)
        raw_trk.loc[raw_trk["mcparticleID"] == main_pid, "found_min"] = np.min(
            [raw_trk[raw_trk["mcparticleID"] == main_pid]["found_min"].values[0], found])
        raw_trk.loc[raw_trk["mcparticleID"] == main_pid, "found_max"] = np.max(
            [raw_trk[raw_trk["mcparticleID"] == main_pid]["found_max"].values[0], found])
        if raw_trk[raw_trk["mcparticleID"] == main_pid]["if_recon"].values[0] == 0:
            raw_trk.loc[raw_trk["mcparticleID"] == main_pid, "if_recon"] = 1 if found >= 3 else 0
        raw_trk.loc[raw_trk["mcparticleID"] == main_pid, "recon_count"] += 1
        raw_trk.loc[raw_trk["mcparticleID"] == main_pid, "success_times"] += (1 if found >= 3 else 0)

    return raw_trk


def process_one_evt(evt_path):
    path_trk_pred = os.path.join(workdir, "Eval", "BackResult", evt_path + ".npy")
    path_raw = os.path.join(workdir, "Preprocess", "csv_with_tracks", evt_path + "_trk.csv")

    trk_pred = np.load(path_trk_pred)
    raw = pd.read_csv(path_raw, index_col=0)
    # print(raw)

    restore_df = eval_efficiency(raw, trk_pred)

    return restore_df


def main():
    path = os.path.join(workdir, "Eval", "BackResult")
    evt_paths = os.listdir(path)
    evt_paths = [evt_path.split(".")[0] for evt_path in evt_paths]  # [:1000]

    gprint("Start to check efficiency...")

    with Pool(CHECKER_PROCESSOR) as p:
        res = [p.apply_async(process_one_evt, args=(evt_path,)) for evt_path in evt_paths]
        res = [r.get() for r in tqdm(res)]

    res = pd.concat(res)
    res.to_csv(os.path.join(workdir, "Eval", "Efficiency.csv"))

    gprint(f"Finish checking efficiency with {len(res)} Track, results are saved in: ", end="")
    yprint(os.path.join(workdir, "Eval", "Efficiency.csv"))


if __name__ == "__main__":
    main()
