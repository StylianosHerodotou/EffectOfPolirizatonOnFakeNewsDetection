import networkx as nx

from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph


def add_identidy_node_feature_to_graph_large(base_large):
    identidy = dict()
    for node in base_large.nodes:
        identidy[node] = node

    nx.set_node_attributes(base_large, identidy, "identidy_self")


def add_centrality_node_features_to_graph_to_large(base_large):
    add_centrality_node_features_to_graph(base_large)