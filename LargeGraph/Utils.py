import networkx as nx
import numpy as np
from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph
from torch_geometric.data import HeteroData
import torch


def add_centrality_node_features_to_graph_to_large(base_large):
    add_centrality_node_features_to_graph(base_large)


def add_identidy_node_feature_to_graph_large(base_large):
    identidy = dict()
    for node in base_large.nodes:
        identidy[node] = node

    nx.set_node_attributes(base_large, identidy, "identidy_self")


def turn_data_to_positive_and_negative_edges(edge_index, edge_attr):
    positive_index_from = list()
    positive_index_to = list()

    negative_index_from = list()
    negative_index_to = list()
    numberOfEdges = edge_attr.size()[0]
    for index in range(0, numberOfEdges):
        if (edge_attr[index][0] == 1.0):
            positive_index_from.append(edge_index[0][index])
            positive_index_to.append(edge_index[1][index])
        else:
            negative_index_from.append(edge_index[0][index])
            negative_index_to.append(edge_index[1][index])

    positive_index = torch.tensor(np.array([positive_index_from, positive_index_to])).long()
    negative_index = torch.tensor(np.array([negative_index_from, negative_index_to])).long()
    return positive_index, negative_index


class DataInBatchesOnlySigned(HeteroData):
    def __init__(self, signed_x, pos_edge_index, neg_edge_index):
        super().__init__()
        self.signed_x = signed_x
        self.pos_edge_index = pos_edge_index
        self.neg_edge_index = neg_edge_index