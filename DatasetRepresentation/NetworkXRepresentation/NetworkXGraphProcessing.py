import os
import json
import networkx as nx

from DatasetRepresentation.BaseDataset import get_nodes_wiki_id_using_mapping, switch_values_and_keys
from Utilities import JoinRawDatasetUtils

from Utilities.InitGlobalVariables import dir_to_large



def get_embedings_node_feature(graph, embeddings, int_to_node_map, emb_names=None):
    node_features = dict()
    list_of_emb_types = []

    if emb_names is None:
        for model_name in embeddings.keys():
            node_features[model_name] = dict()
            list_of_emb_types.append(model_name)
    else:
        for model_name in embeddings.keys():
            if model_name in emb_names:
                node_features[model_name] = dict()
                list_of_emb_types.append(model_name)

    for node in graph.nodes:
        key = int_to_node_map[node]
        for model_name in list_of_emb_types:
            # print(model_name)
            current_emb = embeddings[model_name][key]
            node_features[model_name][node] = current_emb
    return node_features


def read_embedings_file(path_to_dir):
    file_name = os.path.join(path_to_dir, "emb.json")
    f = open(file_name)
    read_dic = json.load(f)
    return read_dic


def get_identidy_node_feature(graph, int_to_node_small, node_to_int_large):
    identidy = dict()
    for node in graph.nodes:
        key = int_to_node_small[node]
        id = node_to_int_large[key]
        identidy[node] = id
    return identidy


def remove_nodes_from_gragh_not_in_large(graph, all_large_names, nodes_to_delete_names, int_to_node_mapping):
    ####find which nodes to remove
    # find all nodes in small.
    small_nodes = get_nodes_wiki_id_using_mapping(int_to_node_mapping)
    # print("initially there are", len(small_nodes))
    # --> find common nodes in large and small
    small_nodes_in_large = small_nodes.intersection(all_large_names)
    # print("left after removing those not in large", len(small_nodes_in_large))
    # --> remove nodes that i removed from the original large one.
    small_nodes_in_large = small_nodes_in_large.difference(nodes_to_delete_names)
    # print("left after removing those i cut before", len(small_nodes_in_large))
    # --> find which nodes to remove
    nodes_to_remove = small_nodes.difference(small_nodes_in_large)

    # find ids of those nodes
    map = switch_values_and_keys(int_to_node_mapping)
    to_delete_ids = []
    for key in nodes_to_remove:
        to_delete_ids.append(map[key])

    # remove them from the graph
    # print("to_delete_ids", len(to_delete_ids))
    # print("before", graph.number_of_nodes())
    graph.remove_nodes_from(to_delete_ids)
    # print("after", graph.number_of_nodes())
    return graph



# dir_to_raw="/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset"
# raw_filename="joined_dataset_no_preprosessing.csv"
# path_to_raw_file=os.path.join(dir_to_raw, raw_filename)




# no_pre_df= read_networkx_dataset(raw_filename)
# no_pre_df.shape

def get_large_node_ids(dir_to_large, threshold=None):
    int_to_node_mapping = JoinRawDatasetUtils.read_int_to_node_mapping(dir_to_large)
    all_nodes_ids = get_nodes_wiki_id_using_mapping(int_to_node_mapping)

    if (threshold == None):
        return all_nodes_ids

    large_graph = JoinRawDatasetUtils.read_graph_file(dir_to_large)
    to_remove = set()

    for node in large_graph.nodes(data=True):
        if (node[1]["coverage"] < threshold):
            to_remove.add(int_to_node_mapping[node[0]])

    to_return = all_nodes_ids - to_remove
    return to_return

# large_nodes= get_large_node_ids(dir_to_large, threshold=None)
# large_embedings=read_embedings_file(dir_to_large)
# node_to_int_large =JoinRawDatasetUtils.read_node_to_int_mapping(dir_to_large)

# remove_nodes_not_in_large(no_pre_df)
# no_pre_df.shape

# dir_path= "/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset"
# filename= "joined_dataset_only_large_nodes.csv"
# save_dataframe_with_networkx(no_pre_df, dir_path, filename)

def add_centrality_node_features_to_graph(graph, features=None):
  if features is None:
      features = ["degree_centrality", "betweenness_centrality",
                  "closeness_centrality", "clustering_coefficinet",
                  "hits"]
  if "betweenness_centrality" in features:
      betweenness_centrality = nx.betweenness_centrality(graph)
      nx.set_node_attributes(graph, betweenness_centrality, "betweenness_centrality")

  if "degree_centrality" in features:
      degree_centrality = nx.degree_centrality(graph)
      nx.set_node_attributes(graph, degree_centrality, "degree_centrality")

  if "closeness_centrality" in features:
      closeness_centrality = nx.closeness_centrality(graph)
      nx.set_node_attributes(graph, closeness_centrality, "closeness_centrality")

  if "clustering_coefficinet" in features:
      clustering_coefficinet = nx.clustering(graph)
      nx.set_node_attributes(graph, clustering_coefficinet, "clustering_coefficinet")

  if "hits" in features:
      hits = nx.hits(graph)
      hits_hubs = hits[0]
      hits_authorities = hits[1]
      nx.set_node_attributes(graph, hits_hubs, "hits_hubs")
      nx.set_node_attributes(graph, hits_authorities, "hits_authorities")

def add_identidy_node_feature_to_graph(graph, int_to_node_small,node_to_int_large):
  identidy=get_identidy_node_feature(graph,int_to_node_small,node_to_int_large)
  nx.set_node_attributes(graph, identidy, "identidy")

def add_identidy_fellowship_node_feature_to_graph(graph, int_to_node_small,node_to_int_large_fellowship):
  identidy=get_identidy_node_feature(graph,int_to_node_small,node_to_int_large_fellowship)
  nx.set_node_attributes(graph, identidy, "identidy_fellowship")


def add_large_node_embeddings_to_graph(graph, int_to_node_small,large_embedings, large_emb_names=["POLE", "signed"]):
  large_node_feature=get_embedings_node_feature(graph, large_embedings, int_to_node_small,large_emb_names)
  for key in large_node_feature.keys():
    nx.set_node_attributes(graph, large_node_feature[key], "large_" + key)