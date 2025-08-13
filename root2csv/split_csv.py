from utils import *
from multiprocessing    import Pool


def process_csv_files(df, evt, force=False):
    """
    Process the csv files
    """


    # df = df[df["eventID"] == evt]

    store_path_hits = os.path.join(workdir, "PreProcess", "csv_with_hits", "evt_%i_hit.csv" % evt)
    store_path_tracks = os.path.join(workdir, "PreProcess", "csv_with_tracks", "evt_%i_trk.csv" % evt)

    if os.path.exists(store_path_hits) and os.path.exists(store_path_tracks) and not force:
        yprint(f"Event {int(evt)} already processed, skipping...")
        return

    df.to_csv(store_path_hits,
              columns=["hit_id", "x", "y", "z", "eventID", "layer"])

    df.to_csv(store_path_tracks,
              columns=["hit_id", "p", "isPrim", "isDecay", "vertex_x", "vertex_y", "vertex_z", "feta", "kind",
                       "layer_id", "mcparticleID", "eventID", "layer"])

    return


def main(force=False):

    gprint("Loading the csv files, may take a while...")

    csv_path = os.listdir(os.path.join(workdir, "RawData"))
    csv_path = [csv_path[i] for i in range(len(csv_path)) if csv_path[i].endswith(".csv")]
    df = pd.read_csv(os.path.join(workdir, "RawData", csv_path[0]), index_col=0)
    print(df)
    yprint(csv_path, end="")
    gprint(f" loaded successfully, splitting the csv files...")


    with Pool(16) as p:
        re = [p.apply_async(process_csv_files, args=(df[df["eventID"] == evt], evt, force)) for evt in df["eventID"].unique()]

        re = [r.get() for r in tqdm(re)]





    pass

if __name__ == "__main__":
    main(True)