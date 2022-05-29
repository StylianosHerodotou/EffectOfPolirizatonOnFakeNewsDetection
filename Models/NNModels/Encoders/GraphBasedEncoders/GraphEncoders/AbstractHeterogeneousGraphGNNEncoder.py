from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeterogeneousGNNEncoder import AbstractHeterogeneousGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch


class AbstractHeterogeneousGraphGNNEncoder(AbstractGraphGNNEncoder, AbstractHeterogeneousGNNEncoder):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def pool_forward(self, useful_data, pool_layer):
        x_dict = useful_data.x_dict
        for key in x_dict.keys():
            current_useful_data= useful_data[key]
            pool_input = self.get_pool_input(current_useful_data)
            all_data = pool_layer(pool_input)
            current_useful_data = self.get_useful_pool_result_data(current_useful_data, all_data)
            useful_data[key]= current_useful_data
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
