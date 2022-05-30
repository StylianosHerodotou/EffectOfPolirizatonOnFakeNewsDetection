from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeteroGNNEncoder import AbstractHeteroGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch


class AbstractHeterogeneousGraphGNNEncoder(AbstractGraphGNNEncoder, AbstractHeteroGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        current_layer_pooling_dict = torch.nn.ModuleDict()

        for edge_type in  pyg_data.edge_types:
            current_layer_pooling_dict[edge_type] = self.generate_single_pool_layer(pyg_data, layer_hyperparameters[edge_type])
        return current_layer_pooling_dict

    def pool_forward(self, useful_data, pool_layer_dict):
        for key, pool_layer in pool_layer_dict.items():
            current_useful_data = useful_data[key]
            current_useful_data = self.single_pool_layer_pass(current_useful_data, pool_layer)
            useful_data[key] = current_useful_data
        return useful_data

    def update_vector_representation(self, useful_data, vector_representation):
        current_layer_vector_representation = dict()
        x_dict = useful_data.x_dict

        for key in x_dict.keys():
            current_useful_data= useful_data[key]
            x, batch = current_useful_data.x, current_useful_data.batch

            current_layer_vector_representation_for_node_type = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            current_layer_vector_representation[key]=current_layer_vector_representation_for_node_type

        if vector_representation is None:
            return current_layer_vector_representation
        else:
            for key in vector_representation.keys():
                vector_representation[key]+=current_layer_vector_representation[key]
            return vector_representation

    def final_update_vector_representation(self, vector_representation):
        for key in vector_representation.keys():
            vector_representation[key] /= len(self.convs)

        all_vector_representations = list()
        for value in vector_representation.values():
            all_vector_representations.append(value)

        vector_representation = torch.stack(all_vector_representations, dim=-1)
        return vector_representation
