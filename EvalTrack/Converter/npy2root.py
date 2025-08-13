import uproot
from utils import *
import numpy as np
import pandas as pd


def npy2root(file_path, targetFolder, df):
    # print("Converting %s to root file..." % file_path)
    tracks = np.load(file_path, allow_pickle=True)
    # print(np.array(tracks[:, :4], dtype=int), tracks.shape)

    tracks_converted = np.zeros_like(tracks[:, :4], dtype=int)
    for i in range(4):
        tracks_converted[:, i] = df.loc[tracks[:, i], "layer_id"].values

    print(tracks_converted, tracks_converted.shape)

    # store the converted tracks to root file
    root_file = uproot.recreate(os.path.join(targetFolder, file_path.split("\\")[-1].replace(".npy", ".root")))
    root_file.mktree("ReconTrack", {"layer0": "int32", "layer1": "int32", "layer2": "int32", "layer3": "int32"})

    root_file["ReconTrack"].extend({"layer0": tracks_converted[:, 0], "layer1": tracks_converted[:, 1],
                                    "layer2": tracks_converted[:, 2], "layer3": tracks_converted[:, 3]})




    root_file.close()

    pass


if __name__ == "__main__":
    targetFolder = os.path.join(workdir, "Eval", "BackResultConvert")
    OriFolder = os.path.join(workdir, "Eval", "BackResult")
    files = os.listdir(OriFolder)

    csv_files = os.listdir(os.path.join(workdir, "RawData"))
    csv_files = [csv_files[i] for i in range(len(csv_files)) if csv_files[i].endswith(".csv")][0]

    df = pd.read_csv(os.path.join(workdir, "RawData", csv_files), index_col=0)
    print(df)

    for file in files:
        if file.endswith(".npy"):
            npy2root(os.path.join(OriFolder, file), targetFolder, df)
