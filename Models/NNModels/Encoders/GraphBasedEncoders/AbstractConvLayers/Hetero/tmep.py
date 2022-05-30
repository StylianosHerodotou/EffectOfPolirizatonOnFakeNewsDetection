from abc import ABC

from torch_geometric.nn import GATv2Conv

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HeteroDataModels.NormalToHeteroGATGraphEncoder import \
    NormalToHeteroGATGraphEncoder


class AbstractHomoToHeteroConvolution(AbstractGraphGNNEncoder, ABC):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):

        homoconv_layer = super(AbstractHomoToHeteroConvolution, self). \
            generate_conv_layer(pyg_data, layer_hyperparameters, aggr_type="mean")
        return conv_layer

    # returns a list of dictionaries of dictionaries .
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = []
        for current_layer_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
            layer_hyperparameters_for_each_node_type = dict()
            for node_type in pyg_data.node_types:
                current_hyperparameters = current_layer_hyperparameters[node_type]

                layer_hyperparameters_for_node_type = dict()
                if len(hyperparameters_for_each_layer) == 0:
                    layer_hyperparameters_for_node_type["in_channels"] = in_channels
                else:
                    prev_layer = hyperparameters_for_each_layer[-1][node_type]
                    layer_hyperparameters_for_node_type["in_channels"] = prev_layer["hidden_channels"] \
                                                                         * prev_layer["heads"]
                layer_hyperparameters_for_node_type["hidden_channels"] = current_hyperparameters["hidden_channels"]
                layer_hyperparameters_for_node_type["heads"] = current_hyperparameters["heads"]
                layer_hyperparameters_for_node_type["dropout"] = current_hyperparameters["dropout"]

                layer_hyperparameters_for_node_type["edge_dim"] = model_parameters[node_type]["edge_dim"]
                layer_hyperparameters_for_each_node_type[node_type] = layer_hyperparameters_for_node_type
            hyperparameters_for_each_layer.append(layer_hyperparameters_for_each_node_type)
        return hyperparameters_for_each_layer

    def conv_forward(self, useful_data, conv_layer):
        x, edge_index, edge_attr = useful_data.x, useful_data.edge_index, useful_data.edge_attr
        new_x = conv_layer(x, edge_index, edge_attr)
        useful_data.x = new_x
        return useful_data
