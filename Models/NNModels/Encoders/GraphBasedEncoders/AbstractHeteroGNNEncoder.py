from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder, copy_dict
from torch_geometric.data import HeteroData


class HeterogeneousDict(dict):

    def __init__(self, pyg_data):
        assert (isinstance(pyg_data, HeteroData))
        object_dict = copy_dict(pyg_data.to_dict())
        object_dict["edge_types"] = pyg_data.edge_types
        object_dict["node_types"] = pyg_data.node_types
        super(HeterogeneousDict, self).__init__(object_dict)

    def get_x_dict(self):
        x_dict = dict()
        for node_type in self["node_types"]:
            x_dict[node_type] = self[node_type]["x"]
        return x_dict

    def get_edge_index_dict(self):
        edge_index_dict = dict()
        for edge_type in self["edge_types"]:
            edge_index_dict[edge_type] = self[edge_type]["edge_index"]
        return edge_index_dict

    def get_edge_attr_dict(self):
        edge_attr_dict = dict()
        for edge_type in self["edge_types"]:
            #             edge_attr_dict[edge_type]=None
            if "edge_attr" in self[edge_type].keys():
                edge_attr_dict[edge_type] = self[edge_type]["edge_attr"]

        if len(edge_attr_dict) == 0:
            return None

        return edge_attr_dict

    def set_new_x_dict(self, new_x_dict):
        for node_type, new_x in new_x_dict.items():
            self[node_type]["x"] = new_x


class AbstractHeteroGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def extract_useful_data_from_input(self, pyg_data):
        useful_data = HeterogeneousDict(pyg_data)
        return useful_data

    def activation_forward(self, useful_data):
        x_dict = useful_data.get_x_dict()
        for key, value in x_dict.items():
            x_dict[key] = self.activation_function(value)
        useful_data["x_dict"] = x_dict
        return useful_data
