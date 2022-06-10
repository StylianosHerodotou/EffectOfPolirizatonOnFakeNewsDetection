from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder, copy_dict
from torch_geometric.data import HeteroData


class HeterogeneousDict(dict):
    def __init__(self, pyg_data):
        assert (isinstance(pyg_data, HeteroData))
        object_dict = copy_dict(pyg_data.to_dict())
        super(HeterogeneousDict, self).__init__(object_dict)


class AbstractHeteroGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def extract_useful_data_from_input(self, pyg_data):
        useful_data = HeterogeneousDict(pyg_data)
        return useful_data

    def activation_forward(self, useful_data):
        x_dict = useful_data["x_dict"]
        for key, value in x_dict.items():
            x_dict[key] = self.activation_function(value)
        useful_data["x_dict"] = x_dict
        return useful_data
