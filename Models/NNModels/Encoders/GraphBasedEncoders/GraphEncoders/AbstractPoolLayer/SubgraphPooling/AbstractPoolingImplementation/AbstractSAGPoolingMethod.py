from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    AbstractSubgraphPooling import AbstractSubgraphPooling
from torch_geometric.nn import SAGPooling


class AbstractSAGPoolingMethod(AbstractSubgraphPooling, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_single_pool_layer(self, pyg_data, layer_hyperparameters):
        pooling_layer = SAGPooling(in_channels=layer_hyperparameters["in_channels"],
                                   ratio=layer_hyperparameters["ratio"])
        return pooling_layer

    def single_pool_layer_pass(self, useful_data, pool_layer):
        x, edge_index = useful_data.x, useful_data.edge_index
        edge_attr, batch = useful_data.edge_attr, useful_data.batch

        x, edge_index, edge_attr, batch, _, _ = pool_layer(x, edge_index, edge_attr, batch)

        useful_data.x = x
        useful_data.edge_index = edge_index
        useful_data.edge_attr = edge_attr
        useful_data.batch = batch
        return useful_data


    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = []
        for current_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
            layer_hyperparameters = dict()
            if len(hyperparameters_for_each_layer) == 0:
                layer_hyperparameters["in_channels"] = in_channels
            else:
                prev_layer = hyperparameters_for_each_layer[-1]
                layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"] * prev_layer["heads"]
            layer_hyperparameters["hidden_channels"] = current_hyperparameters["hidden_channels"]
            layer_hyperparameters["heads"] = current_hyperparameters["heads"]
            layer_hyperparameters["dropout"] = current_hyperparameters["dropout"]

            layer_hyperparameters["edge_dim"] = model_parameters["edge_dim"]
            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer
