from abc import ABC

from torch_geometric.nn import GATv2Conv

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeteroGNNEncoder import AbstractHeteroGNNEncoder


class HANConvolution(AbstractHeteroGNNEncoder, ABC):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        conv_layer = GATv2Conv(in_channels=layer_hyperparameters["in_channels"],
                               out_channels=layer_hyperparameters["hidden_channels"],
                               heads=layer_hyperparameters["heads"],
                               dropout=layer_hyperparameters["dropout"],
                               edge_dim=layer_hyperparameters["edge_dim"])
        return conv_layer

    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):

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

    def conv_forward(self, useful_data, conv_layer):
        x_dict, edge_index_dict, edge_attr_dict = useful_data["x_dict"],\
                                                  useful_data["edge_index_dict"],\
                                                  useful_data["edge_attr_dict"]
        new_x_dict = conv_layer(x_dict, edge_index_dict)
        for node_type, new_x in new_x_dict.items():
            useful_data[node_type].x=new_x
        return useful_data
