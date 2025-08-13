from utils import *
from sklearn.neighbors import KDTree
import warnings

warnings.filterwarnings("ignore")
import natsort



@timer
def find_nearest_hit(hit_df):
    # xyz = hit_df[["x", "y", "z"]].values

    layer_hit_frame = []
    for layer in range(4):
        layer_hits = hit_df[hit_df["layer"] == layer]
        layer_xyz = layer_hits[["x", "y", "z"]].values

        tree = KDTree(layer_xyz)
        dist, ind = tree.query(layer_xyz, k=2)

        layer_hits["nearest_dist"] = dist[:, 1]

        layer_hit_frame.append(layer_hits)

    layer_hit_frame = pd.concat(layer_hit_frame)
    # print(layer_hit_frame)
    layer_hit_frame.sort_values("hit_id", inplace=True)
    return layer_hit_frame

@timer
def cut_range(hit_df_012, hit, xyz_all, id_all):
    # t = time.perf_counter()
    # print(hit_index)
    # hit = hit_df.loc[hit_index:hit_index]
    # print(hit_df.loc[hit_index:hit_index])
    # print(hit)
    xyz_hit = hit[["x", "y", "z"]].values[0]
    # hit_df_012 = hit_df[hit_df["layer"] != 3]
    # print(hit_df_123)
    # xyz_all = hit_df_012[["x", "y", "z"]].values
    # id_all = hit_df_012["hit_id"].values



    # print("cut_range1\t", time.perf_counter() - t)

    cos_theta = np.dot(xyz_all, xyz_hit) / (np.linalg.norm(xyz_all, axis=1) * np.linalg.norm(xyz_hit))
    cos_theta[cos_theta > 1] = 1
    theta = np.arccos(cos_theta)

    index = np.where(theta < 0.03)
    # print(index, theta.shape, index[0].shape)

    # print("cut_range2\t", time.perf_counter() - t)


    hit_df_012 = hit_df_012.loc[id_all[index]]
    # print("cut_range3a\t", time.perf_counter() - t)

    hit_df_0123 = pd.concat([hit, hit_df_012])
    # print("cut_range3b\t", time.perf_counter() - t)

    hit_df_0123.sort_values("hit_id", inplace=True)

    # print("cut_range3c\t", time.perf_counter() - t)

    # print(hit_df)

    return hit_df_0123

@timer
def process_one_evt(path, store_path, force=False):

    if os.path.exists(store_path) and not force:
        if len(os.listdir(store_path)) > 0:
            yprint(f"Hit {store_path.split('_')[2]} exists, skippping...")
            return False


    df = pd.read_csv(path, index_col=0)
    df = find_nearest_hit(df)

    df_l3 = df[df["layer"] == 3]



    hit_df_012 = df[df["layer"] != 3]

    xyz_all = hit_df_012[["x", "y", "z"]].values
    id_all = hit_df_012["hit_id"].values

    for i in df_l3["hit_id"].values:
        hit = df.loc[i:i]
        cut_df = cut_range(hit_df_012, hit, xyz_all, id_all)
        cut_df.to_csv(os.path.join(store_path, f"hit_{i}.csv"))

    return 1



def main(force=False):
    file_list = os.listdir(os.path.join(workdir, "PreProcess", "csv_with_hits"))
    file_list = natsort.natsorted(file_list)


    gprint("Start Spliting hits...")
    gprint(f"Processing {len(file_list)} files...")



    re = []
    # process_one_evt(os.path.join(workdir, "PreProcess", "csv_with_hits", file_list[0]), os.path.join(workdir, "Eval", "BackPre", file_list[0].split("_hit.csv")[0]), force)

    with Pool(16) as p:
        for file in file_list:
            store_path = os.path.join(workdir, "Eval", "BackPre", file.split("_hit.csv")[0])
            os.makedirs(store_path, exist_ok=True)
            re.append(p.apply_async(process_one_evt, args=(os.path.join(workdir, "PreProcess", "csv_with_hits", file), store_path, force)))
        re = [r.get() for r in tqdm(re)]

    # for file in file_list:
    #     store_path = os.path.join(workdir, "Eval", "BackPre", file.split("_hit.csv")[0])
    #     os.makedirs(store_path, exist_ok=True)
    #     r = process_one_evt(os.path.join(workdir, "PreProcess", "csv_with_hits", file), store_path, force)
    #     re.append(r)





if __name__ == "__main__":
    main(True)
