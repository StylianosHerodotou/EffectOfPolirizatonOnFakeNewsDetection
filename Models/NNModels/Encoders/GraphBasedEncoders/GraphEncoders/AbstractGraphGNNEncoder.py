import torch
from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder

class AbstractGraphGNNEncoder(AbstractGNNEncoder):

    @abstractmethod
    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     hyperparameters_for_each_layer=None ):
        pass

    def add_pool_layers(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            new_pool_layer = self.generate_pool_layer(pyg_data, layer_hyperparameters)
            self.pools.append(new_pool_layer)


    @abstractmethod
    def pool_forward(self, useful_data, pool_layer):
        pass


    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.pools = torch.nn.ModuleList()
        self.add_pool_layers(in_channels, pyg_data, model_parameters)

