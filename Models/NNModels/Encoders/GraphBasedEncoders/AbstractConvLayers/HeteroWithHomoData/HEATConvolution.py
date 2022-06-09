from abc import ABC
from torch_geometric.nn import HEATConv
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHomoGNNEncoder import AbstractHomogeneousGNNEncoder


class HEATConvolution(AbstractHomogeneousGNNEncoder, ABC):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        conv_layer = HEATConv(in_channels=layer_hyperparameters["in_channels"],
                              out_channels=layer_hyperparameters["hidden_channels"],
                              num_node_types=layer_hyperparameters["num_node_types"],
                              num_edge_types=layer_hyperparameters["num_edge_types"],
                              edge_type_emb_dim=layer_hyperparameters["edge_type_emb_dim"],
                              edge_dim=layer_hyperparameters["edge_dim"],
                              edge_attr_emb_dim=layer_hyperparameters["edge_attr_emb_dim"],
                              heads=layer_hyperparameters["heads"],
                              dropout=layer_hyperparameters["dropout"],
                              )

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
            layer_hyperparameters["edge_type_emb_dim"] = current_hyperparameters["edge_type_emb_dim"]
            layer_hyperparameters["edge_attr_emb_dim"] = current_hyperparameters["edge_attr_emb_dim"]
            layer_hyperparameters["heads"] = current_hyperparameters["heads"]
            layer_hyperparameters["dropout"] = current_hyperparameters["dropout"]

            layer_hyperparameters["num_node_types"] = model_parameters["num_node_types"]
            layer_hyperparameters["num_edge_types"] = model_parameters["num_edge_types"]
            layer_hyperparameters["edge_dim"] = model_parameters["edge_dim"]

            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer

    def conv_forward(self, useful_data, conv_layer):
        x, edge_index, edge_attr, node_type, edge_type = useful_data["x"], \
                                                         useful_data["edge_index"],\
                                                         useful_data["edge_attr"],\
                                                         useful_data["node_type"],\
                                                         useful_data["edge_type"]
        new_x = conv_layer(x, edge_index, node_type, edge_type, edge_attr)
        useful_data["x"] = new_x
        return useful_data
