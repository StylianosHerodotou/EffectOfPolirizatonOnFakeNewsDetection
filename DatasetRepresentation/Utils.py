from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
from torch_geometric.data import Data

from DatasetRepresentation.BaseDataset import copy_df
from DatasetRepresentation.PyGRepresentation.PyGDataset import PyGDataset


def turn_to_hetero_graph(graph, node_features, edge_features):
    # get edges with different sign
    positive = list()
    negative = list()
    for edge in graph.edges(data=True):
        if (edge[2]["weight"] == 1.0):
            positive.append((edge[0], edge[1]))
        else:
            negative.append((edge[0], edge[1]))
    # get subgraph contaninng only the edges with the appropriate sign
    pos_sub = graph.edge_subgraph(positive)
    neg_sub = graph.edge_subgraph(negative)

    # remove the weight attriute since it doesnt add anything when they are seperate
    edge_features.remove("weight")
    # print(graph, pos_sub, neg_sub)

    # get three different representations of the same graph, all edges, positive and negative
    # create the hereto object
    hetero_graph = HeteroData()

    fullGraph = from_networkx(graph, group_node_attrs=node_features, group_edge_attrs=None)
    hetero_graph['person'].x = fullGraph.x.float()

    try:
        posGraph = from_networkx(pos_sub, group_node_attrs=None, group_edge_attrs=edge_features)
        hetero_graph['person', 'likes', 'person'].edge_index = posGraph.edge_index.long()
        hetero_graph['person', 'likes', 'person'].edge_attr = posGraph.edge_attr.float()
    except:
        hetero_graph['person', 'likes', 'person'].edge_index = None
        hetero_graph['person', 'likes', 'person'].edge_attr = None
    try:
        negGraph = from_networkx(neg_sub, group_node_attrs=None, group_edge_attrs=edge_features)
        hetero_graph['person', 'dislikes', 'person'].edge_index = negGraph.edge_index.long()
        hetero_graph['person', 'dislikes', 'person'].edge_attr = negGraph.edge_attr.float()
    except:
        hetero_graph['person', 'dislikes', 'person'].edge_index = None
        hetero_graph['person', 'dislikes', 'person'].edge_attr = None

    return hetero_graph


def make_networkx_to_pyg_graph(graph, turn_to_hetero=False):
    for node in graph.nodes(data=True):
        node_features = [node_feature for node_feature in node[1].keys()]
        break

    for edge in graph.edges(data=True):
        edge_features = [edge_features for edge_features in edge[2].keys()]
        break

    if len(node_features) == 0:
        node_features = None

    if len(edge_features) == 0:
        edge_features = None

    if (turn_to_hetero):
        return turn_to_hetero_graph(graph, node_features, edge_features)
    else:
        fullGraph = from_networkx(graph, group_node_attrs=node_features, group_edge_attrs=edge_features)
        fullGraph.edge_index = fullGraph.edge_index.long()
        fullGraph.x = fullGraph.x.float()
        fullGraph.edge_attr = fullGraph.edge_attr.float()

        return fullGraph


def make_networkx_to_pyg_df(dataset):
    df= copy_df(dataset.df)
    new_graphs = list()
    for index, row in df.iterrows():
        new_graphs.append(make_networkx_to_pyg_graph(row[dataset.graph_column_name]))
    df[dataset.graph_column_name] = new_graphs
    dataset = PyGDataset(df)
    return dataset