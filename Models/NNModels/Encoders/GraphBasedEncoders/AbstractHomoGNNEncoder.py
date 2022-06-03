from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractHomogeneousGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.is_homogeneous_data=True

    def activation_forward(self, useful_data):
        x = useful_data.x
        self.activation_function(x)
        useful_data.x = x
        return useful_data
