from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHomoGNNEncoder import AbstractHomogeneousGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch


class AbstractHomogeneousGraphGNNEncoder(AbstractGraphGNNEncoder, AbstractHomogeneousGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        return self.generate_single_pool_layer(pyg_data, layer_hyperparameters)

    def pool_forward(self, useful_data, pool_layer):
        useful_data= self.single_pool_layer_pass(useful_data, pool_layer)
        return useful_data

    def update_vector_representation(self, useful_data, vector_representation):
        x, batch = useful_data.x, useful_data.batch

        current_layer_vector_representation = torch.cat([gmp(x, batch), gap(x, batch)], dim=-1)
        if vector_representation is None:
            return current_layer_vector_representation
        else:
            vector_representation += current_layer_vector_representation
            return vector_representation

    def final_update_vector_representation(self, vector_representation):
        vector_representation /= len(self.convs)
        return vector_representation
