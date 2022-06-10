from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeteroGNNEncoder import HeterogeneousDict
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    AbstractPoolingImplementation.AbstractSAGPoolingMethod import \
    AbstractSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.VectorAggregationMethods.AbstractHeteroVectorAggregationMethod import \
    AbstractHeteroVectorAggregationMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    VectorAggregationMethods.AbstractMeanAggregator import \
    AbstractMeanAggregator
import torch
from torch_geometric.data import HeteroData


class HeteroSAGPoolingMeanAggregator(AbstractSAGPoolingMethod, AbstractHeteroVectorAggregationMethod,
                                     AbstractMeanAggregator, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)


    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters, **kwargs):
        conv_hyperparameters_for_each_layer_for_all_edge_type = self.generate_hyperparameters_for_each_conv_layer(
            in_channels, pyg_data, model_parameters)

        pool_hyperparameters_for_each_layer = list()
        for all_edge_layer_hyperparameeters in conv_hyperparameters_for_each_layer_for_all_edge_type:
            input_size = 0
            for edge_type, edge_hyperparameters in all_edge_layer_hyperparameeters.items():
                input_size += edge_hyperparameters["hidden_channels"] * \
                              edge_hyperparameters["heads"]
                break
            ratio = model_parameters["pooling_ratio"]
            layer_hyperparameters = {
                "in_channels": input_size,
                "ratio": ratio
            }
            pool_hyperparameters_for_each_layer.append(layer_hyperparameters)

        return pool_hyperparameters_for_each_layer

    def get_node_mask(self, homo_data):
        return homo_data.pooling_perm

    def get_edge_mask(self, homo_data, old_edge_index):
        perm = homo_data.pooling_perm
        num_nodes = homo_data.node_type_labels.size(0)
        mask = perm.new_full((num_nodes,), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i
        row, col = old_edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        return mask

    def extend_pooling_to_remaining_attributes(self, homo_data, old_edge_index, old_x):
        node_mask = self.get_node_mask(homo_data)
        edge_mask = self.get_edge_mask(homo_data, old_edge_index)
        number_of_old_nodes = old_x.size(0)
        number_of_old_edges = old_edge_index.t().size(0)

        for key in homo_data.to_dict().keys():
            # print(key)
            tensor_size = homo_data[key].size(0)
            if tensor_size == number_of_old_edges and key.startswith("edge"):
                homo_data[key] = homo_data[key][edge_mask]
            elif tensor_size == number_of_old_nodes:
                homo_data[key] = homo_data[key][node_mask]
        return homo_data

    def pool_forward(self, useful_data, pool_layer):
        print(useful_data)
        hetero_data = HeteroData(useful_data)
        homo_data = hetero_data.to_homogeneous()
        old_edge_index = homo_data.edge_index
        old_x = homo_data.x
        homo_data= super(HeteroSAGPoolingMeanAggregator, self).pool_forward(homo_data, pool_layer)
        homo_data = self.extend_pooling_to_remaining_attributes(homo_data, old_edge_index, old_x)
        hetero_data = homo_data.to_heterogeneous()

        useful_data= HeterogeneousDict(hetero_data)
        return useful_data
