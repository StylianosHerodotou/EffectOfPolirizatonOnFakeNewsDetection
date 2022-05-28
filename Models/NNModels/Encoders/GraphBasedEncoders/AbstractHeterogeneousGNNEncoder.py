from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractHeterogeneousGNNEncoder(AbstractGNNEncoder):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def conv_forward(self, useful_data, conv_layer):
        conv_input = self.get_conv_input(useful_data)
        x_dict = conv_layer(conv_input)
        x_dict = {key: x for key, x in x_dict.items()}
        useful_data.x_dict = x_dict
        return useful_data

    def activation_forward(self, useful_data):
        x_dict = useful_data.x_dict
        for key, value in x_dict.items():
            x_dict[key] = self.activation_function(value)
        useful_data.x_dict = x_dict
        return useful_data
