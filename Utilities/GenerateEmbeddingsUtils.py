import json
import os

from DatasetRepresentation.BaseDataset import switch_values_and_keys
from Utilities.InitGlobalVariables import dir_to_large
from Utilities.JoinRawDatasetUtils import read_node_to_int_mapping, read_int_to_node_mapping
import json


def get_nodes_wiki_id_using_path(dir_path):
    nodes = read_node_to_int_mapping(dir_path)
    nodes = set(nodes.keys())
    return nodes


def create_embs_dic(graph, emb_dic, map):
    to_write = {}
    for emb_name in emb_dic.keys():
        to_write[emb_name] = {}

    emb_index = 0;
    for node in graph.nodes():
        key = map[node]
        for emb_name, emb_value in emb_dic.items():
            to_write[emb_name][key] = emb_dic[emb_name][emb_index].detach().numpy().tolist()
        emb_index += 1
    return to_write


def write_emd_to_file(graph, emb_dic, to_write_path=None, overwrite_prev_embeddings=False):
    if (to_write_path == None):
        to_write_path = dir_to_large
    map = read_int_to_node_mapping(dir_to_large)
    to_write = create_embs_dic(graph, emb_dic, map)
    file_name = os.path.join(to_write_path, "emb.json")
    if overwrite_prev_embeddings:
        f = open(file_name, "w")
    else:
        f = open(file_name, "a")
    json.dump(to_write, f)
    f.close()


def removeNodesFromGraph(setOfNodeNamesToRemove, graph, int_to_node_mapping):
    node_to_int_mapping = switch_values_and_keys(int_to_node_mapping)
    setOfNodeIdsToRemove = set();

    for node_name in setOfNodeNamesToRemove:
        node_id = node_to_int_mapping[node_name]
        setOfNodeIdsToRemove.add(node_id)
    graph.remove_nodes_from(setOfNodeIdsToRemove)


def map_embeddings_to_node_in_graph(graph):
    index = 0
    mapping = {}
    for node in graph.nodes():
        mapping[index] = node
        index += 1

    return mapping


def map_embeddings_to_node_name(embeddings, graph, pyg_data):
    to_return_mapping = {}
    int_to_node_mapping = map_embeddings_to_node_in_graph(graph)
    for node_type in embeddings.keys():
        current_node_type_embeddings = embeddings[node_type]
        node_id_mapping = pyg_data[node_type]["id_to_graph"]
        for index, node_embeddings in enumerate(current_node_type_embeddings):
            node_id = node_id_mapping[index]
            node_name = int_to_node_mapping[node_id.item()]
            to_return_mapping[node_name] = node_embeddings.detach().numpy().tolist()
    return to_return_mapping


def write_clear_emd_to_file(embeddings, graph, pyg_data,
                            to_write_path="Datasets/EmbeddingsForPublicGraph",
                            name_file="hetero_GAT.json"):
    embeddings_dic = map_embeddings_to_node_name(embeddings, graph, pyg_data)
    file_name = os.path.join(to_write_path, name_file)
    f = open(file_name, "w")
    json.dump(embeddings_dic, f)
    f.close()
