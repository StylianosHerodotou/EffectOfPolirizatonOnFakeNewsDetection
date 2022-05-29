from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.AbstractSubgraphPooling import \
    AbstractSubgraphPooling
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class AbstractMeanAggregator(AbstractSubgraphPooling, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def update_single_vector_representation(self, useful_data, vector_representation):
        x, batch = useful_data.x, useful_data.batch

        current_layer_vector_representation = list()
        current_layer_vector_representation.append(global_add_pool(x, batch))
        current_layer_vector_representation.append(global_mean_pool(x, batch))
        current_layer_vector_representation.append(global_max_pool(x, batch))

        current_layer_vector_representation = torch.cat(current_layer_vector_representation, dim=-1)
        if vector_representation is None:
            return current_layer_vector_representation
        else:
            vector_representation += current_layer_vector_representation
            return vector_representation

    def final_single_update_vector_representation(self, vector_representation):
        vector_representation /= len(self.convs)
        return vector_representation
