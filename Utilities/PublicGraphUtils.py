from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph


def add_centrality_node_features_to_graph_to_large(base_large):
    add_centrality_node_features_to_graph(base_large)


import networkx as nx


def remove_nodes_based_on_frequency(large_graph, int_to_node_mapping_lagre, threshold):
    to_remove_ids = set()
    to_remove_names = set()
    for node in large_graph.nodes(data=True):
        if (node[1]["coverage"] < threshold):
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


#################turn to hetero
from collections import defaultdict
import torch


def map_nodes_from_old_to_new(G, node_attrs, type_name):
    indexes_per_type_mapping = dict()
    global_to_type_id_mapping = dict()

    for node_type in node_attrs.keys():
        indexes_per_type_mapping[node_type] = 0

    for node in G.nodes(data=True):
        # 1 is the index to the node features dictionary
        global_index = node[0]
        current_node_type = node[1][type_name]

        type_index = indexes_per_type_mapping[current_node_type]
        indexes_per_type_mapping[current_node_type] += 1

        global_to_type_id_mapping[global_index] = type_index
    return global_to_type_id_mapping


def find_nodes_per_node_type(G, node_attrs, type_name):
    nodes_per_type = {}
    for node_type in node_attrs.keys():
        nodes_per_type[node_type] = list()

    for node in G.nodes(data=True):
        # 1 is the index to the node features dictionary
        current_node_type = node[1][type_name]
        node_list = nodes_per_type[current_node_type]
        node_list.append(node[:1])
    return nodes_per_type


def find_edges_per_edge_type(G, edge_attrs, type_name, global_to_type_id_mapping):
    edges_per_type = {}
    for edge_type in edge_attrs.keys():
        edges_per_type[edge_type] = list()

    for edge in G.edges(data=True):
        # 2 is the index to the edge features dictionary.
        current_edge_type = edge[2][type_name]
        edge_list = edges_per_type[current_edge_type]

        #####new
        type_id_from=global_to_type_id_mapping[edge[0]]
        type_id_to = global_to_type_id_mapping[edge[1]]
        new_edge = [type_id_from, type_id_to]
        edge_list.append(new_edge)
    return edges_per_type


def find_edge_type_old_to_new_mapping(G, type_name):
    edge_type_old_to_new_mapping = {}
    temp_set = set()

    for edge in G.edges(data=True):
        edge_type = edge[2][type_name]
        if edge_type not in temp_set:
            from_node_type = G.nodes[edge[0]][type_name]
            to_node_type = G.nodes[edge[1]][type_name]
            edge_key = (from_node_type, edge_type, to_node_type)
            edge_type_old_to_new_mapping[edge_type] = edge_key
            temp_set.add(edge_type)
    return edge_type_old_to_new_mapping


def create_edge_index_for_each_edge_type(edges_per_type, edge_type_old_to_new_mapping):
    edge_index_for_type = {}

    for edge_type in edges_per_type.keys():
        edge_list = edges_per_type[edge_type]
        if len(edge_list) == 0:
            continue
        edge_key = edge_type_old_to_new_mapping[edge_type]
        edge_index_for_type[edge_key] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index_for_type


def add_node_features_for_each_node_type(G, node_attrs, type_name, dict_data):
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        node_type = feat_dict[type_name]
        current_data = dict_data[node_type]

        if set(feat_dict.keys()) != node_attrs[node_type]:
            raise ValueError('Not all nodes of same type have the same attributes')
        for key, value in feat_dict.items():
            if key == type_name:
                continue
            current_data[str(key)].append(value)


def add_edge_features_for_each_node_type(G, edge_attrs, type_name, edge_type_old_to_new_mapping, dict_data):
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        old_edge_type = feat_dict[type_name]
        new_edge_type = edge_type_old_to_new_mapping[old_edge_type]
        current_data = dict_data[new_edge_type]

        if set(feat_dict.keys()) != edge_attrs[old_edge_type]:
            raise ValueError('Not all edges of the same type contain the same attributes')
        for key, value in feat_dict.items():
            #             key = f'edge_{key}' if key in node_attrs else key
            if key == type_name:
                continue
            current_data[str(key)].append(value)


def add_graph_features(G, dict_data):
    for key, value in G.graph.items():
        #         key = f'graph_{key}' if key in node_attrs else key
        dict_data[str(key)] = value


def turn_content_of_dict_data_into_tensors(dict_data):
    for key, value in dict_data.items():
        if (type(value) == defaultdict):
            for second_level_key, second_level_value in value.items():
                try:
                    dict_data[key][second_level_key] = torch.tensor(second_level_value).float()
                except ValueError:
                    pass
        else:
            try:
                dict_data[key] = torch.tensor(value).float()
            except ValueError:
                pass


def add_edge_index_for_each_type_pyg_data(pyg_data, edge_index_for_type):
    for edge_type, current_edge_index in edge_index_for_type.items():
        #         pyg_data[edge_type[0], edge_type[1], edge_type[2]].edge_index =current_edge_index.view(2, -1)
        pyg_data[edge_type].edge_index = current_edge_index.view(2, -1)


def add_node_attributes_to_pyg_data(pyg_data, dict_data, node_attrs, type_name, add_node_features_as_labels=True):
    for node_type, type_attributes in node_attrs.items():
        # check if there are any node features for that node except type
        if type_attributes == {type_name}:
            #             print("no node features for ", node_type)
            continue

        current_dict_data = dict_data[node_type]
        current_node_features = []
        for attribute in type_attributes:
            if attribute == type_name:
                continue
            x = current_dict_data[attribute]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            ###add as label
            if (add_node_features_as_labels):
                pyg_data[node_type][attribute] = torch.squeeze(x)
            ###
            current_node_features.append(x)

        pyg_data[node_type].x = torch.cat(current_node_features, dim=-1)


def add_edge_attributes_to_pyg_data(pyg_data, dict_data, edge_attrs, type_name, edge_type_old_to_new_mapping,
                                    add_edge_features_as_labels=True):
    for old_edge_type, type_attributes in edge_attrs.items():
        # check if there are any edge features for that edge except type
        if type_attributes == {type_name}:
            #             print("no edge features for ", old_edge_type)
            continue

        new_edge_type = edge_type_old_to_new_mapping[old_edge_type]
        current_dict_data = dict_data[new_edge_type]

        current_edge_features = []
        for attribute in type_attributes:
            if attribute == type_name:
                continue

            x = current_dict_data[attribute]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            ###add as label
            if add_edge_features_as_labels:
                pyg_data[new_edge_type][attribute] = torch.squeeze(x)
            ###
            current_edge_features.append(x)

        #         pyg_data[new_edge_type[0], new_edge_type[1], new_edge_type[2]].edge_attr =torch.cat(current_edge_features, dim=-1)
        pyg_data[new_edge_type].edge_attr = torch.cat(current_edge_features, dim=-1)


def add_node_type_label(pyg_data):
    for node_label, node_type in enumerate(pyg_data.node_types):
        num_nodes_of_type = pyg_data[node_type].x.size()[0]
        node_labels = torch.full([num_nodes_of_type], fill_value=node_label, dtype=torch.long)
        pyg_data[node_type].node_type_labels = node_labels


def add_edge_type_label(pyg_data):
    for edge_label, edge_type in enumerate(pyg_data.edge_types):
        num_edge_of_type = pyg_data[edge_type].edge_index.size()[1]
        edge_labels = torch.full([num_edge_of_type], fill_value=edge_label, dtype=torch.long)
        pyg_data[edge_type].edge_type_labels = edge_labels


def generate_node_mapping(G, node_attrs, type_name, dict_data):
    index = 0;

    # init the lists
    for node_type, dict_for_node_type in dict_data.items():
        dict_for_node_type["id_to_graph"] = list()

    for node, feat_dict in G.nodes(data=True):
        # put id on node
        node_type = feat_dict[type_name]
        current_data = dict_data[node_type]

        current_data["id_to_graph"].append(index)
        index += 1


def add_node_mapping(pyg_data, dict_data):
    for current_type in dict_data.keys():
        if "id_to_graph" in dict_data[current_type].keys():
            pyg_data[current_type]["id_to_graph"] = dict_data[current_type]["id_to_graph"]


def from_networkx_to_hetero(G, node_attrs=None,
                            edge_attrs=None,
                            type_name="type"):
    import networkx as nx

    from torch_geometric.data import HeteroData

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G

    # map_nodes_from_old_to_new
    global_to_type_id_mapping = map_nodes_from_old_to_new(G, node_attrs, type_name)

    # find nodes per node type:
    nodes_per_type = find_nodes_per_node_type(G, node_attrs, type_name)

    # find edges per edge type
    edges_per_type = find_edges_per_edge_type(G, edge_attrs, type_name, global_to_type_id_mapping)

    # find edge_type_old_to_new_mapping
    edge_type_old_to_new_mapping = find_edge_type_old_to_new_mapping(G, type_name)

    # create edge index for each edge type.
    edge_index_for_type = create_edge_index_for_each_edge_type(edges_per_type, edge_type_old_to_new_mapping)

    # create dict_data
    dict_data = {}
    for node_type in node_attrs.keys():
        dict_data[node_type] = defaultdict(list)

    # add node_id to graph
    generate_node_mapping(G, node_attrs, type_name, dict_data)

    for edge_type in edge_index_for_type.keys():
        dict_data[edge_type] = defaultdict(list)

    # for each node type add node features:
    add_node_features_for_each_node_type(G, node_attrs, type_name, dict_data)

    # for each edge type add edge features:
    add_edge_features_for_each_node_type(G, edge_attrs, type_name, edge_type_old_to_new_mapping, dict_data)

    # add other graph featues
    add_graph_features(G, dict_data)

    # turn everything into tensors.
    turn_content_of_dict_data_into_tensors(dict_data)

    # Create HeteroData Object
    pyg_data = HeteroData()

    # Add edge_index
    add_edge_index_for_each_type_pyg_data(pyg_data, edge_index_for_type)

    # Add node attributes
    add_node_attributes_to_pyg_data(pyg_data, dict_data, node_attrs, type_name)

    # Add edge attributes
    add_edge_attributes_to_pyg_data(pyg_data, dict_data, edge_attrs, type_name, edge_type_old_to_new_mapping)

    # Add node type label:
    add_node_type_label(pyg_data)

    # add node_mapping
    add_node_mapping(pyg_data, dict_data)

    # Add edge type label:
    add_edge_type_label(pyg_data)

    return pyg_data


def find_node_features(graph):
    node_features = {}
    for node in graph.nodes(data=True):
        node_type = node[1]["type"]

        if node_type not in node_features.keys():
            node_features[node_type] = set();

        node_features_for_type = node_features[node_type]
        for key in node[1].keys():
            node_features_for_type.add(key)
    return node_features


def find_edge_features(graph):
    edge_features = {}
    for edge in graph.edges(data=True):
        edge_type = edge[2]["type"]

        if edge_type not in edge_features.keys():
            edge_features[edge_type] = set();

        edge_features_for_type = edge_features[edge_type]
        for key in edge[2].keys():
            edge_features_for_type.add(key)

    return edge_features


# add identity feature:
def add_identity_feature_to_public(graph):
    identity_feature = dict()
    index = 0
    for node in graph.nodes():
        identity_feature[node] = index
        index += 1
    nx.set_node_attributes(graph, identity_feature, "identity")
