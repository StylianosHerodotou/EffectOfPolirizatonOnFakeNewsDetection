from DatasetRepresentation.BaseDataset import BaseDataset
from torch_geometric.data import Data
import json
import os
import torch
import pandas as pd


def read_pyg_dataset(name, dir_to_dataset= "/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/Datasets"):
    path = os.path.join(dir_to_dataset, name)
    dataset = pd.read_csv(path)
    dataset = PyGDataset(dataset)
    dataset.turn_dicts_to_pyg_graphs()
    try:
        dataset.turn_json_to_mapping()
    except:
        pass

    try:
        dataset.turn_mappings_to_int(dataset, in_json=True)
    except:
        dataset.turn_mappings_to_int(dataset, in_json=False)

    return dataset

class PyGDataset(BaseDataset):
    def __init__(self, df):
        super().__init__(df)

    def turn_pyg_graphs_to_dicts(self):
        graphs = self.df[self.graph_column_name].tolist()
        new_graphs = list()

        for graph in graphs:
            graph = graph.to_dict()

            for key in graph.keys():
                graph[key] = graph[key].cpu().numpy().tolist()

            graph = json.dumps(graph)
            new_graphs.append(graph)
        self.df[self.graph_column_name] = new_graphs

    # turn_pyg_graphs_to_dicts(only_large_df)

    def turn_dicts_to_pyg_graphs(self):
        graphs = self.df[self.graph_column_name].tolist()
        new_graphs = list()

        for graph in graphs:
            graph = json.loads(graph)

            for key in graph.keys():
                graph[key] = torch.Tensor(graph[key])
            data = Data(x=graph["x"], edge_index=graph["edge_index"], edge_attr=graph["edge_attr"])
            new_graphs.append(data)
        self.df[self.graph_column_name] = new_graphs

    # turn_dicts_to_pyg_graphs(only_large_self.df)
    # only_large_df["graph"].tolist()[0]

    def save_dataframe_with_pyg(self, filename,
                                dir_path="/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset"):
        path_to_save = os.path.join(dir_path, filename)
        self.turn_pyg_graphs_to_dicts()
        self.turn_mapping_to_json()
        self.df.to_csv(path_to_save, index=False)