import torch
from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractGraphGNNEncoder(AbstractGNNEncoder):
    @abstractmethod
    def get_pool_layer(self, pyg_data, layer_hyperparameters):
        pass

    @abstractmethod
    def add_pool_layers(self, in_channels, pyg_data, model_parameters):
        pass

    def __init__(self, in_channels, pyg_data,model_parameters):
        super().__init__(in_channels, pyg_data,model_parameters)
        self.pools = torch.nn.ModuleList()
        self.add_pool_layers(in_channels, pyg_data, model_parameters)
