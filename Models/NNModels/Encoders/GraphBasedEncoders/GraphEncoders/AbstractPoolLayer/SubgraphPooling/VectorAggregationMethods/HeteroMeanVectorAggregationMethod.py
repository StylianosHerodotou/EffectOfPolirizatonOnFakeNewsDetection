from abc import ABC
import torch
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.\
    AbstractPoolLayer.SubgraphPooling.VectorAggregationMethods.\
    HomoMeanVectorAggregationMethod import \
    HomoMeanVectorAggregationMethod


class HeteroMeanVectorAggregationMethod(HomoMeanVectorAggregationMethod, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def update_vector_representation(self, useful_data, vector_representation):
        current_layer_vector_representation = dict()
        x_dict = useful_data.get_x_dict()
        for key in x_dict.keys():
            current_layer_vector_representation[key] = None

        for key,value in x_dict.items():
            current_useful_data = useful_data[key]
            current_layer_vector_representation_for_node_type = current_layer_vector_representation[key]
            current_layer_vector_representation_for_node_type = super().update_vector_representation(
                current_useful_data, current_layer_vector_representation_for_node_type)
            current_layer_vector_representation[key] = current_layer_vector_representation_for_node_type

        if vector_representation is None:
            return current_layer_vector_representation
        else:
            for key in vector_representation.keys():
                vector_representation[key] += current_layer_vector_representation[key]
            return vector_representation

    def final_update_vector_representation(self, vector_representation):
        for key, vector_representation_for_node_type in vector_representation.items():
            vector_representation[key] = super().final_update_vector_representation(
                vector_representation_for_node_type)

        all_vector_representations = list()
        for value in vector_representation.values():
            all_vector_representations.append(value)

        vector_representation = torch.cat(all_vector_representations)
        return vector_representation

