from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractHeteroGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.is_homogeneous_data=False

    def activation_forward(self, useful_data):
        x_dict = useful_data.x_dict
        for key, value in x_dict.items():
            x_dict[key] = self.activation_function(value)
        # useful_data.x_dict = x_dict
        return useful_data
