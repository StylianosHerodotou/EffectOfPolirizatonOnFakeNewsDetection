import torch
from abc import ABC, abstractmethod

class AbstractGNNEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def generate_conv_layer(self, pyg_data,layer_hyperparameters):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        pass

    @abstractmethod
    def add_conv_layers(self,in_channels, pyg_data,model_parameters):
        pass

    @abstractmethod
    def forward(self, pyg_data):
        pass

    def __init__(self, in_channels, pyg_data,model_parameters):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.add_conv_layers(in_channels, pyg_data,model_parameters)
