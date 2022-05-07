import networkx as nx

from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph


def add_identidy_node_feature_to_graph_large(base_large):
    identidy = dict()
    for node in base_large.nodes:
        identidy[node] = node

    nx.set_node_attributes(base_large, identidy, "identidy_self")


def add_centrality_node_features_to_graph_to_large(base_large):
    add_centrality_node_features_to_graph(base_large)


import networkx as nx

def remove_nodes_based_on_frequency(large_graph,int_to_node_mapping_lagre,threshold):
  to_remove_ids=set()
  to_remove_names=set()
  for node in large_graph.nodes(data=True):
    if(node[1]["coverage"]<threshold):
      to_remove_names.add(int_to_node_mapping_lagre[node[0]])
      to_remove_ids.add(node[0])

  # print(len(to_remove_ids))
  # print(large_graph.number_of_nodes())
  large_graph.remove_nodes_from(to_remove_ids)
  large_graph.remove_nodes_from(list(nx.isolates(large_graph)))
  # print(large_graph.number_of_nodes())
  return to_remove_names


def remove_edges_based_on_frequency(large_graph, int_to_node_mapping_lagre, threshold):
    to_remove_edges = set()
    to_remove_names = set()

    for edge in large_graph.edges(data=True):
        if (edge[2]["frequency"] < threshold):
            to_remove_edges.add((edge[0], edge[1]))
            to_remove_names.add((int_to_node_mapping_lagre[edge[0]], int_to_node_mapping_lagre[edge[1]]))

    # print("i am going to remove this many ", len(to_remove_edges))
    # print("before, ", large_graph.number_of_edges())
    large_graph.remove_edges_from(to_remove_edges)
    large_graph.remove_nodes_from(list(nx.isolates(large_graph)))
    # print(large_graph.number_of_nodes())
    # print("\nafter ",large_graph.number_of_edges())
    return to_remove_names