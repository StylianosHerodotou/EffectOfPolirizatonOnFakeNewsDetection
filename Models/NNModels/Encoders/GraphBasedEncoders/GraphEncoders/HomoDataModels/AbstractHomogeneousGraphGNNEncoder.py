from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHomoGNNEncoder import AbstractHomogeneousGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractHomogeneousGraphGNNEncoder(AbstractGraphGNNEncoder, AbstractHomogeneousGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        return self.generate_single_pool_layer(pyg_data, layer_hyperparameters)

    def pool_forward(self, useful_data, pool_layer):
        useful_data = self.single_pool_layer_pass(useful_data, pool_layer)
        return useful_data
