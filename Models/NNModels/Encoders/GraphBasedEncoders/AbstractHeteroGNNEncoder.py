from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractHeteroGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    # def conv_forward(self, useful_data, conv_layer):
    #     x_dict = self.object_specifci_conv_pass(useful_data, conv_layer)
    #     x_dict = {key: x for key, x in x_dict.items()}
    #     useful_data.x_dict = x_dict
    #     return useful_data

    def activation_forward(self, useful_data):
        x_dict = useful_data.x_dict
        print(x_dict)
        for key, value in x_dict.items():
            x_dict[key] = -1 * self.activation_function(value)
        print("something else \n\n", x_dict)
        useful_data.x_dict = x_dict
        return useful_data
