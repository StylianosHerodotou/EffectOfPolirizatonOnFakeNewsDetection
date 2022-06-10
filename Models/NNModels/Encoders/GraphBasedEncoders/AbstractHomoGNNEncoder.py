from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder, copy_dict
from torch_geometric.data import Data


class HomogeneousDict(dict):
    def __init__(self, pyg_data):
        assert (isinstance(pyg_data, Data))
        object_dict = copy_dict(pyg_data.to_dict())
        super(HomogeneousDict, self).__init__(object_dict)


class AbstractHomogeneousGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def extract_useful_data_from_input(self, pyg_data):
        useful_data = HomogeneousDict(pyg_data)
        return useful_data

    def activation_forward(self, useful_data):
        x = useful_data["x"]
        self.activation_function(x)
        useful_data["x"] = x
        return useful_data
