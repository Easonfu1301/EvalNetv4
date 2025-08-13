import os.path
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

from dataset import FilteringData_B as FilteringData
from Net import Filter_MLP_Backward as Filter_MLP
from torch.utils.data import DataLoader
from utils import *
import torch.nn.functional as F

import natsort

model = Filter_MLP()
dict_path = os.listdir(os.path.join(workdir, "Model"))[-1]
dict_path = os.path.join(workdir, "Model", dict_path)
# dict_path = os.path.join(workdir, "Model", "filtering_model_back.pth")
model.load_state_dict(torch.load(dict_path))
model.to(device)
model.eval()

# torch.compile(model, backend='nvprims')
# model = torch.compile(model, backend='aot_eager')    # 启用编译优化

P_MIN = 0

# @timer
@jit(nopython=True)
def gen_pairs(hit_ids, max_hits, ITER_MULTI, df_list_len, hits_len):
    pairs = np.empty((ITER_MULTI * max_hits, df_list_len), dtype=hit_ids[0].dtype)

    # 填充 pairs 数组
    for hit_nb in range(ITER_MULTI * max_hits):
        pairs[hit_nb] = [
            hit_ids[i][hit_nb % hits_len[i]] for i in range(df_list_len)
        ]
    return pairs


@timer
def generate_hit_pairs(df_list, max_hits):
    """
    生成包含多个 DataFrame hit_id 的组合对。

    参数:
    df_list (list of pd.DataFrame): 包含 hit_id 列的 DataFrame 列表。
    max_hits (int): 最大 hit 数，用于确定生成对的数量。

    返回:
    np.ndarray: 包含生成的 hit_id 对的二维数组。
    """
    # 获取每个 DataFrame 的长度
    hits_len = [len(df) for df in df_list]
    # random_indices = [np.random.permutation(length) for length in hits_len]

    # 提取 hit_id 列并转换为 NumPy 数组
    hit_ids = [df["hit_id"].to_numpy() for df in df_list]

    # xyz0 = df_list[0][["x", "y", "layer"]].to_numpy()
    # xyz1 = df_list[1][["x", "y", "layer"]].to_numpy()
    # xyz2 = df_list[2][["x", "y", "layer"]].to_numpy()
    # xyz3 = df_list[3][["x", "y", "layer"]].to_numpy()
    #
    # indices0 = np.lexsort((xyz0[:, 2], xyz0[:, 1], xyz0[:, 0]))
    # indices1 = np.lexsort((xyz1[:, 2], xyz1[:, 1], xyz1[:, 0]))
    # indices2 = np.lexsort((xyz2[:, 2], xyz2[:, 1], xyz2[:, 0]))
    # indices3 = np.lexsort((xyz3[:, 2], xyz3[:, 1], xyz3[:, 0]))
    #
    #
    #
    # indices = [indices0, indices1, indices2, indices3]

    # # 提取所有数据框的 x/y/layer 列并转为 NumPy 数组
    # xyzs = [df[["x", "y", "layer"]].to_numpy() for df in df_list]

    # # 向量化生成排序索引：每个数组按 x → y → layer 优先级排序
    # indices = [np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0])) for xyz in xyzs]
    #
    #
    # 预分配结果数组
    pairs = gen_pairs(hit_ids, max_hits, ITER_MULTI, len(df_list), hits_len)


    # for hit_nb in range(ITER_MULTI * max_hits):
    #     pairs[hit_nb] = [
    #         hit_ids[i][indices[i][hit_nb % hits_len[i]]]
    #         for i in range(len(df_list))
    #     ]

    return pairs



@timer
def eval_one_track(path):
    df = pd.read_csv(path, index_col=0)
    # print(df)

    df_l0 = df[df["layer"] == 0]
    df_l1 = df[df["layer"] == 1]
    df_l2 = df[df["layer"] == 2]
    df_l3 = df[df["layer"] == 3]

    if_hit = [len(df_l0) > 0, len(df_l1) > 0, len(df_l2) > 0, len(df_l3) > 0]

    if if_hit.count(True) < 4:
        raise ValueError(f"Not enough hits, maximum {if_hit.count(True)} hits")

    weight_dict = [{}, {}, {}, {}]
    # t = time.time()
    for iter in range(ITER_TIME):
        max_hits = np.max([len(df_l0), len(df_l1), len(df_l2), len(df_l3)])

        pairs = generate_hit_pairs([df_l0, df_l1, df_l2, df_l3], max_hits)



        data = construct_sample(pairs, df)

        pred = predict(data)

        weight_dict = get_weight(pairs, pred, df)



        hits = [df_l0, df_l1, df_l2, df_l3]



        for i in range(1, 4):
            # get last keys
            hit_id_cut = list(weight_dict[i].keys())[:1 + int(JUDGE_PRESERVE_RATE * len(weight_dict[i]))]
            # print(hit_id_cut)
            hits[i] = hits[i][hits[i]["hit_id"].isin(hit_id_cut)]

        df_l0, df_l1, df_l2, df_l3 = hits
        if len(df_l0) == 1 and len(df_l1) == 1 and len(df_l2) == 1 and len(df_l3) == 1:
            # yprint("Early stop ITER")
            break
        # print("time to weight: ", time.time() - t)
    # rprint("time to weight: ", time.time() - t)
    hit_keys = [int(list(weight_dict[i].keys())[0]) for i in range(4)]
    weight = [list(weight_dict[i].values())[0][0] * 4 for i in range(4)]


    hit_xyz = df.loc[hit_keys, ["x", "y", "z"]]

    fit_info = calc_chi2(hit_xyz)
    # print(fit_info["chi2"], weight)

    # print(hit_keys, weight)

    return 0, 0, hit_keys, weight, fit_info["chi2"]

@timer
def construct_sample(sample, df):
    data = construct_sample_var(sample, df)

    return data

@timer
def predict(data, model=model):
    data = FilteringData(data)
    data = DataLoader(data, batch_size=int(np.min([len(data), 1024])), shuffle=False)

    preds = []
    with torch.no_grad():

        for embed_link, label in data:
            link_pred = model(embed_link)
            preds.append(link_pred)

    preds = torch.cat(preds, dim=0)

    # softmax

    preds = F.softmax(preds, dim=1)

    return preds








# def get_weight(pair, pred, df):
#     t = time.time()
#     weight_dict = {}
#     trans_kind = {0: [0, 0, 0, 1],
#                   1: [1, 0, 0, 1],
#                   2: [0, 1, 0, 1],
#                   3: [0, 0, 1, 1],
#                   4: [1, 1, 0, 1],
#                   5: [1, 0, 1, 1],
#                   6: [0, 1, 1, 1],
#                   7: [1, 1, 1, 1]}
#
#     pred = pred.cpu().detach().numpy()
#
#     print("1: ", time.time() - t)
#
#     for i in range(pair.shape[0]):
#         for kind in range(8):
#             kind_transed = trans_kind[kind]
#             for j in range(4):
#                 # if kind_transed[j - 1] == 1:
#                 try:
#                     weight_dict[pair[i, j]][0] += pred[i, kind] * kind_transed[j]
#                     weight_dict[pair[i, j]][1] += 1 * kind_transed[j]
#                 except Exception as e:
#                     weight_dict[pair[i, j]] = [pred[i, kind], 1]
#
#     print("2: ", time.time() - t)
#
#     for key in weight_dict.keys():
#         weight_dict[key] = [weight_dict[key][0] / weight_dict[key][1]]
#
#     print("3: ", time.time() - t)
#
#     weight_by_layer = [{}, {}, {}, {}]
#     for key in weight_dict.keys():
#         layer = int(df.loc[key, "layer"])
#         weight_by_layer[layer][key] = weight_dict[key]
#         # sort by value\
#         weight_by_layer[layer] = dict(sorted(weight_by_layer[layer].items(), key=lambda x: x[1][0], reverse=True))
#
#     print("4: ", time.time() - t)
#
#     return weight_by_layer


# @jit(nopython=True)
@timer
def get_weight(pair, pred, df):
    t = time.time()
    weight_dict = {}
    # trans_kind = {0: [0, 0, 0, 1],
    #               1: [1, 0, 0, 1],
    #               2: [0, 1, 0, 1],
    #               3: [0, 0, 1, 1],
    #               4: [1, 1, 0, 1],
    #               5: [1, 0, 1, 1],
    #               6: [0, 1, 1, 1],
    #               7: [1, 1, 1, 1]}

    trans_array = np.array([[0, 0, 0, 1],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [1, 1, 0, 1],
                            [1, 0, 1, 1],
                            [0, 1, 1, 1],
                            [1, 1, 1, 1]])


    size = [4, 4, 4, 8]



    pred = pred.cpu().detach().numpy()

    # print("1: ", time.time() - t)


    grade = np.dot(pred, trans_array)


    for i in range(grade.shape[0]):
        for j in range(4):
            try:
                weight_dict[pair[i, j]][0] += grade[i, j]
                weight_dict[pair[i, j]][1] += size[j]
            except Exception as e:
                weight_dict[pair[i, j]] = [grade[i, j], size[j]]





    for key in weight_dict.keys():
        weight_dict[key] = [weight_dict[key][0] / weight_dict[key][1]]

    # print("3: ", time.time() - t)

    weight_by_layer = [{}, {}, {}, {}]

    # 先将所有权重存入weight_by_layer

    keys = [int(key) for key in weight_dict.keys()]
    layers = df.loc[keys, "layer"].values
    # print(layers)
    for i in range(len(keys)):
        # key = keys[i]
        # layer = layers[i]
        weight_by_layer[int(layers[i])][keys[i]] = weight_dict[keys[i]]


    # for key in weight_dict.keys():
    #     layer = int(df.loc[key, "layer"])
    #     weight_by_layer[layer][key] = weight_dict[key]

    # 对每一层的字典进行排序
    for layer in range(4):
        weight_by_layer[layer] = dict(sorted(weight_by_layer[layer].items(), key=lambda x: x[1][0], reverse=True))

    # for key in weight_dict.keys():
    #     layer = int(df.loc[key, "layer"])
    #     weight_by_layer[layer][key] = weight_dict[key]
    #     # sort by value\
    #     weight_by_layer[layer] = dict(sorted(weight_by_layer[layer].items(), key=lambda x: x[1][0], reverse=True))

    # print("4: ", time.time() - t)

    return weight_by_layer



@timer
def process_one_evt(path_evt, force=False):
    t = time.time()
    hit_keys = []

    if os.path.exists(os.path.join(workdir, "Eval", "BackResult", path_evt + ".npy")) and not force:
        yprint(f"Already processed {path_evt}.npy, skipping...")
        return -1, -1

    path_hit = os.path.join(workdir, "Eval", "BackPre", path_evt)
    files = os.listdir(path_hit)

    for file in tqdm(files, desc=f"Processing {path_evt}"):
        path = os.path.join(path_hit, file)
        try:
            found, total, hit_key, weight, chi2 = eval_one_track(path)

            # judges = [weight[0] > CUT_L0, weight[1] > CUT_L1, weight[2] > CUT_L2]
            # if judges.count(True) < 2:
            #     continue

            hits_info  = [*hit_key, *weight, chi2]
            # print(hits_info)

            hit_keys.append(hits_info)
        except Exception as e:
            yprint(f"Warning in {path_evt} --- {file.split('.csv')[0]}: {e}")
            # raise e
            continue

    store_path = os.path.join(workdir, "Eval", "BackResult", path_evt + ".npy")

    np.save(store_path, np.array(hit_keys))

    # print(f"Processed {path_evt} in {time.time() - t:.2f}s, {len(files)} files")
    return len(files), time.time() - t





@timer
def main(force=False):
    evt_path = os.path.join(workdir, "Eval", "BackPre")
    evt_path = os.listdir(evt_path)
    evt_path = natsort.natsorted(evt_path)

    gprint(f"Start Restoring Trk for {len(evt_path)} events...")
    gprint(f"Start Time:", end="")
    yprint(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start_time = time.time()

    res = []
    # for file in evt_path:
    #     res.append(process_one_evt(file, force))

    with Pool(EVAL_PROCESSOR) as p:
        res = []
        for file in evt_path:
            res.append(p.apply_async(process_one_evt, args=(file, force,)))

        res = [r.get() for r in res]
        res0 = [r[0] for r in res]

    gprint(f"Restoring Trk Done! {np.sum(res0)} events processed!")

    res = np.array(res)
    time_df = pd.DataFrame(res, columns=["n_files", "time"])
    # print(time_df)

    time_df.to_csv(os.path.join(workdir, "Eval", "time_used.csv"), index=False)


    gprint(f"End Time", end="")
    yprint(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    gprint(f"Total Time", end="")
    yprint(f"{time.time() - start_time:.2f} s")
    # save time used to txt
    t = time.time() - start_time
    with open(os.path.join(workdir, "Eval", "time_used.txt"), "w") as f:
        f.write(f"{t:.3f} s")







if __name__ == "__main__":
    main()
