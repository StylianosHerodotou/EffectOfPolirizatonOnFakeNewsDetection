import torch
from abc import ABC, abstractmethod

class AbstractGNNEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        pass

    def add_conv_layers(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            new_conv_layer = self.generate_conv_layer(pyg_data, layer_hyperparameters)
            self.convs.append(new_conv_layer)

    @abstractmethod
    def forward(self, pyg_data):
        pass

    def __init__(self, in_channels, pyg_data,model_parameters):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.add_conv_layers(in_channels, pyg_data,model_parameters)