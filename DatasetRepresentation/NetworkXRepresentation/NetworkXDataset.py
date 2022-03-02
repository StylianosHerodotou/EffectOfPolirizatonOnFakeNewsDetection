import os
import json
import pandas as pd
import networkx as nx

from DatasetRepresentation.BaseDataset import BaseDataset, turn_str_keys_to_int, get_nodes_wiki_id_using_mapping
from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import remove_nodes_from_gragh_not_in_large, \
    add_centrality_node_features_to_graph, add_identidy_node_feature_to_graph, \
    add_identidy_fellowship_node_feature_to_graph, add_large_node_embeddings_to_graph
from Utilities import JoinRawDatasetUtils

global dir_to_large

def read_networkx_dataset(name,
                          dir_to_dataset= "/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/Datasets"):
    path = os.path.join(dir_to_dataset, name)
    dataset = pd.read_csv(path)
    dataset = NetworkXDataset(dataset)
    dataset.turn_dicts_to_networkx_graphs()
    try:
        dataset.turn_json_to_mapping(mapping_column_name="int_to_node_mapping")
        dataset.turn_json_to_mapping(mapping_column_name="node_to_int_mapping")
    except:
        pass

    try:
        dataset.turn_mappings_to_int(in_json=True)
    except:
        dataset.turn_mappings_to_int(in_json=False)

    return dataset


class NetworkXDataset(BaseDataset):
    def __init__(self, df):
        super().__init__(df)

    def turn_dicts_to_networkx_graphs(self):
        graphs = self.df[self.graph_column_name].tolist()
        new_graphs = list()

        for graph in graphs:
            graph = json.loads(graph)
            # turn keys back to int
            graph = turn_str_keys_to_int(graph)
            for key in graph.keys():
                graph[key] = turn_str_keys_to_int(graph[key])

            new_graphs.append(nx.from_dict_of_dicts(graph))
        self.df[self.graph_column_name] = new_graphs

    def turn_networkx_graphs_to_dicts(self):
        graphs = self.df[self.graph_column_name].tolist()
        new_graphs = list()

        for graph in graphs:
            graph = nx.to_dict_of_dicts(graph)
            graph = json.dumps(graph)
            new_graphs.append(graph)
        self.df[self.graph_column_name] = new_graphs

    def save_dataframe_with_networkx(self, dir_path, filename):
        path_to_save = os.path.join(dir_path, filename)
        self.turn_networkx_graphs_to_dicts()
        self.turn_mapping_to_json()
        if "node_to_int_mapping" in self.df.columns:
            self.df = self.df.drop("node_to_int_mapping", axis=1)
        self.df.to_csv(path_to_save, index=False)

    def remove_nodes_not_in_large(self, nodes_to_delete_names,
                                  ):
        # find all large nodes
        int_to_node_mapping_large = JoinRawDatasetUtils.read_int_to_node_mapping(dir_to_large)
        all_large_nodes = get_nodes_wiki_id_using_mapping(int_to_node_mapping_large)

        # new_graphs = list()
        # indexes_to_remove = list()

        for index, row in self.df.iterrows():
            # before = row[self.graph_column_name].number_of_nodes()
            # remove_nodes_from_gragh_not_in_large(row[self.graph_column_name], all_large_nodes, nodes_to_delete_names, row[int_to_node_mapping_column])
            self.df.at[index, self.graph_column_name] = remove_nodes_from_gragh_not_in_large(row[self.graph_column_name], all_large_nodes,
                                                                                   nodes_to_delete_names,
                                                                                   row[self.mapping_column_name])
            # after =row[self.graph_column_name].number_of_nodes()
            # if(before != after):
            #   print("before", before, "after ", after)

        for index, row in self.df.iterrows():
            if row[self.graph_column_name].number_of_nodes() == 0 or row[self.graph_column_name].number_of_edges() == 0:
                self.df.drop(index, inplace=True)

    def add_centrality_node_features_to_df(self, features=None):
        for index, row in self.df.iterrows():
            add_centrality_node_features_to_graph(row[self.graph_column_name],features)

    def add_identidy_node_feature_to_df(self, node_to_int_large):
        for index, row in self.df.iterrows():
            add_identidy_node_feature_to_graph(row[self.graph_column_name], row[self.mapping_column_name],
                                               node_to_int_large)

    def add_identidy_fellowship_node_feature_to_df(self, node_to_int_large_fellowship):
        for index, row in self.df.iterrows():
            add_identidy_fellowship_node_feature_to_graph(row[self.graph_column_name], row[self.mapping_column_name],
                                                          node_to_int_large_fellowship)

    def add_large_node_embeddings_to_df(self, large_embedings, large_emb_names=["POLE", "signed"]):
        for index, row in self.df.iterrows():
            add_large_node_embeddings_to_graph(row[self.graph_column_name], row[self.mapping_column_name],
                                               large_embedings, large_emb_names=large_emb_names)

    def remove_empty_graphs(self):
      for index, row in self.df.iterrows():
        if row[self.graph_column_name].number_of_nodes()==0 or row[self.graph_column_name].number_of_edges()==0:
          self.df.drop(index, inplace=True)