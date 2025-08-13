from utils import *
from Net import Filter_MLP_Backward as Filter_MLP
from dataset import FilteringData_B as FilteringData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.construct_sample_var_dict import construct_sample_var_dict


model = Filter_MLP()
dict_path = os.path.join(workdir, "Model", "filtering_model_back.pth")
model.load_state_dict(torch.load(dict_path))
model.to(device)



class Evt_Proccessor:
    def __init__(self, Evt_path):
        self.Evt_path = Evt_path

    def __len__(self):
        files = os.listdir(self.Evt_path)
        files = [files[i] for i in range(len(files)) if files[i].endswith(".csv")]
        return len(files)

    def __str__(self):
        return f"Event path: {self.Evt_path}, Number of Hits: {len(self)}"

    def read_all_hits(self):
        files = os.listdir(self.Evt_path)
        files = [files[i] for i in range(len(files)) if files[i].endswith(".csv")]
        self.hits = []
        for file in tqdm(files):
            self.hits.append(Hit(os.path.join(self.Evt_path, file)))

    def build_pairs(self):
        pairs = []
        pairs_var = []
        pairs_len = []
        for hit in tqdm(self.hits):
            pair, pair_var, pair_len = hit.get_track_pairs()
            if pair_len > 0:
                pairs.append(pair)
                pairs_var.append(pair_var)
                pairs_len.append(pair_len)


        pairs = np.concatenate(pairs)
        pairs_var = np.concatenate(pairs_var)
        pairs_len = np.array(pairs_len)
        pairs_start = np.zeros(pairs_len.shape[0])
        pairs_start[1:] = np.cumsum(pairs_len)[:-1]

        return pairs, pairs_var, pairs_start, pairs_len


    def predict(self):
        pairs, pairs_var, pairs_start, pairs_len = self.build_pairs()

        # pairs = construct_sample_var(pairs, )

        dataset = FilteringData(pairs_var)
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=int(np.min([pairs.shape[0], 8192])), shuffle=False)

        preds = []

        for embed_link, label in tqdm(dataloader):
            link_pred = model(embed_link)
            preds.append(link_pred)

        preds = torch.cat(preds, dim=0)

        # softmax

        preds = F.softmax(preds, dim=1)

        return preds


class Hit:
    def __init__(self, hit_path):
        self.hit_path = hit_path
        self.hit = pd.read_csv(self.hit_path, index_col=0)
        self.layer_hits = self.convert_to_dict()
        self.hit_df = self.convert_to_df()
        # self.get_track_pairs()


    def convert_to_dict(self):
        layer_hits = {}
        for hit in self.hit.iterrows():
            hit = hit[1]

            hit_info = {
                "hit_id": hit["hit_id"],
                "x": hit["x"],
                "y": hit["y"],
                "z": hit["z"],
                "eventID": hit["eventID"],
                "layer": hit["layer"],
                "weight": 0,
            }

            try:
                layer_hits[hit["layer"]].append(hit_info)
            except KeyError:
                layer_hits[hit["layer"]] = [hit_info]

        if len(layer_hits.keys()) < 4:
            print(f"Event {self.hit_path.split('BackPre')[1]} has only {len(layer_hits.keys())} layers")
            return None
        else:
            return layer_hits

    def convert_to_df(self):
        hit_df = {}
        for hit in self.hit.iterrows():
            hit = hit[1]

            hit_info = np.array([hit["x"], hit["y"], hit["z"], hit["nearest_dist"]])

            hit_df[hit["hit_id"]] = hit_info


        return hit_df


    def get_track_pairs(self):
        if self.layer_hits is None:
            return None, None, 0
        dict_weights = self.layer_hits

        hits_len = [len(dict_weights[0]), len(dict_weights[1]), len(dict_weights[2]), len(dict_weights[3])]
        max_hits = np.max(hits_len)

        pairs = []

        for hit_nb in range(2 * max_hits):
            hit_l0 = dict_weights[0][hit_nb % hits_len[0]]
            hit_l1 = dict_weights[1][hit_nb % hits_len[1]]
            hit_l2 = dict_weights[2][hit_nb % hits_len[2]]
            hit_l3 = dict_weights[3][hit_nb % hits_len[3]]
            pairs.append([hit_l0["hit_id"], hit_l1["hit_id"], hit_l2["hit_id"], hit_l3["hit_id"]])
        pairs = np.array(pairs)

        pairs_var = construct_sample_var_dict(pairs, self.hit_df)


        return pairs, pairs_var, pairs.shape[0]

        # print(pairs)


if __name__ == "__main__":
    evt = Evt_Proccessor(os.path.join(workdir, "Eval", "BackPre", "evt_1"))
    print(evt)
    evt.read_all_hits()
    # evt.build_pairs()
    evt.predict()
