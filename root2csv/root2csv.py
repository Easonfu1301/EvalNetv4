import numpy as np

from utils import *






def root2df(file_path):
    """
    Convert root file to pandas dataframe
    """
    file_path = os.path.join(workdir, "RawData", file_path)
    export_path = file_path.replace(".root", ".csv")

    root_file = uproot.open(file_path)
    keys = [k for k in root_file.keys() if k.startswith("FakeClusteringUP_")]
    print(root_file.keys())
    try:
        root_file = root_file[keys[0]]
    except Exception as e:
        pass
    dfs = []
    # Get the tree
    for i in range(4):
        gprint("\tProcessing layer %i..." % i)
        tree = root_file["layer%i" % i]
        df_temp = pd.DataFrame()

        df_temp["x"] = tree["layer%i_x" % i].array(library="np") + np.random.normal(0, SMEAR_N_X, len(tree["layer%i_y" % i].array(library="np")))
        df_temp["y"] = tree["layer%i_y" % i].array(library="np") + np.random.normal(0, SMEAR_N_Y, len(tree["layer%i_y" % i].array(library="np")))
        df_temp["z"] = tree["layer%i_z" % i].array(library="np")
        df_temp["layer_id"] = tree["layer%i_id" % i].array(library="np")
        df_temp["mcparticleID"] = tree["mcparticleID_L%i" % i].array(library="np")
        # df_temp["mcparticleID"] = tree["layer%i_id" % i].array(library="np")
        df_temp["eventID"] = tree["eventID_L%i" % i].array(library="np")
        df_temp["p"] = tree["p_L%i" % i].array(library="np")
        df_temp["isPrim"] = tree["isPrim_L%i" % i].array(library="np")
        df_temp["isDecay"] = tree["isDecay_L%i" % i].array(library="np")

        df_temp["feta"] = tree["layer%i_pseudoRapidity" % i].array(library="np")
        df_temp["kind"] = tree["layer%i_particle_kind" % i].array(library="np")

        df_temp["vertex_x"] = tree["layer%i_vertex_x" % i].array(library="np")
        df_temp["vertex_y"] = tree["layer%i_vertex_y" % i].array(library="np")
        df_temp["vertex_z"] = tree["layer%i_vertex_z" % i].array(library="np")

        df_temp["layer"] = i * np.ones(df_temp.shape[0])

        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df)

    #print(df["hit_id"])
    set_id = np.arange(df.shape[0])
    # set_id = np.random.permutation(set_id)
    # print(len(set_id))
    df["hit_id"] = set_id
    df["index"] = set_id
    # print(df)

    # plot_hits(df)



    #################################### only for test time


    df["eventID"] = (df["eventID"]-1) // COMBINE_EVT + 1


    ####################################


    df = df[df["eventID"] <= EVT_NUM]
    # df = df[df["eventID"] > 450]




    bprint("Number of events:\t", df["eventID"].nunique())
    bprint("Number of hits:\t\t", df.shape[0])

    gprint(f"Exporting the dataframe to ", end="")
    yprint(export_path, end="")
    gprint(" file. Might take a while...")

    # set hit_id as index



    df.to_csv(export_path,
              columns=["index", "hit_id", "x", "y", "z", "p", "isPrim", "isDecay", "vertex_x", "vertex_y", "vertex_z", "feta", "kind",
                       "layer_id", "mcparticleID", "eventID", "layer"], index=False)





    return df












def main(force=False):
    root_path = os.listdir(os.path.join(workdir, "RawData"))
    root_path = [root_path[i] for i in range(len(root_path)) if root_path[i].endswith(".root")]
    csv_path = os.listdir(os.path.join(workdir, "RawData"))
    csv_path = [csv_path[i] for i in range(len(csv_path)) if csv_path[i].endswith(".csv")]
    if csv_path:
        if not force:
            yprint("Warning -- Found", end="")
            bprint(f" {csv_path} ", end="")
            yprint("files in RawData folder. Check if you want to overwrite them.")

            return False
        else:
            pass


    if not root_path:
        rprint("FATAL ERROR -- No (.root) file found, check the", end="")
        yprint(" RawFile ", end="")
        rprint("and run the code again.")
        exit(0)
    elif len(root_path) > 1:
        rprint("FATAL ERROR -- More than one (.root) file found, check the", end="")
        yprint(" RawFile ", end="")
        rprint("and run the code again.")
        exit(0)
    elif len(root_path) == 1:
        gprint("Found", end = "")
        yprint(f" {root_path[0]} ", end="")
        gprint("files in RawData folder.")

    root_path = root_path[0]

    root2df(root_path)








if __name__ == "__main__":
    main()
    # df = pd.read_csv(os.path.join(workdir, "RawData", "tuple_reco_baseline.csv"))
    # df = df[df["p"] <= 200]
    # df = df[df["p"] >= 100]
    # dfc = df[""].value_counts()
    #
    # print(df.shape[0])
    #
    # print(dfc)
    #
    # print(dfc / dfc.sum() * 100)