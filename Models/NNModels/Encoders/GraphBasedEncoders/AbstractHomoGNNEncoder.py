from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractHomogeneousGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    # def conv_forward(self, useful_data, conv_layer):
    #     x = self.object_specifci_conv_pass(useful_data, conv_layer)
    #     useful_data.x = x
    #     return useful_data

    def activation_forward(self, useful_data):
        x = useful_data.x
        self.activation_function(x)
        useful_data.x = x
        return useful_data
