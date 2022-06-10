from abc import ABC, abstractmethod


class AbstractConvolution(ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        pass

    @abstractmethod
    def conv_forward(self, useful_data, conv_layer):
        pass