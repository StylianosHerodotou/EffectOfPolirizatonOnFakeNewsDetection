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

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     hyperparameters_for_each_layer=None):
        if hyperparameters_for_each_layer is None:
            conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
                                                                                                    pyg_data,
                                                                                                    model_parameters)

        hyperparameters_for_each_layer = []
        for index, current_hyperparameters in enumerate(model_parameters["hyper_parameters_for_each_layer"]):
            layer_hyperparameters = dict()
            current_conv_hyperparameters = conv_hyperparameters_for_each_layer[index]

            layer_hyperparameters["in_channels"] = current_conv_hyperparameters["hidden_channels"] * \
                                                   current_conv_hyperparameters["heads"]
            layer_hyperparameters["ratio"] = model_parameters["pooling_ratio"]
            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer
