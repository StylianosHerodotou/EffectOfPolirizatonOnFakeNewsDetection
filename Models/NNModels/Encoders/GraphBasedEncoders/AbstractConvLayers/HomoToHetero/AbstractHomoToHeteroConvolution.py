from abc import ABC
from torch_geometric.nn import HeteroConv

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractHomoToHeteroConvolution(AbstractGraphGNNEncoder, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        # self.homo_convolution_class = None

    def generate_conv_layer(self, pyg_data, layer_hyperparameters_for_all_edge_types, aggr_type="mean"):
        conv_dict = dict()
        for edge_type in pyg_data.edge_types:
            layer_hyperparameters = layer_hyperparameters_for_all_edge_types[edge_type]
            conv_dict[edge_type] = self.homo_convolution_class.generate_conv_layer(self, pyg_data,
                                                                                   layer_hyperparameters,
                                                                                   aggr_type=aggr_type)

        return HeteroConv(conv_dict, aggr=aggr_type)

    # returns a list( layer) of dictionaries(edge_type)  of dictionaries (hyperparameters) .
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        # for each edge type generate another model_parameters and call super
        all_edges_hyperparameters_dict = dict()
        for edge_type in pyg_data.edge_types:
            current_edge_pyg_data = pyg_data[edge_type]
            current_edge_model_parameters = model_parameters[edge_type]
            hyperparameters_for_this_edge_type = self.homo_convolution_class. \
                generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data=current_edge_pyg_data,
                                                             model_parameters=current_edge_model_parameters)
            all_edges_hyperparameters_dict[edge_type] = hyperparameters_for_this_edge_type

        all_edges_hyperparameters_list = list()
        for edge_type, edge_hyperparameters_for_all_layers in all_edges_hyperparameters_dict.items():
            # fill list with empty dictionaries.
            if len(all_edges_hyperparameters_list) == 0:
                for index in range(len(edge_hyperparameters_for_all_layers)):
                    all_edges_hyperparameters_list.append(dict())

            for index, edge_layer_hyperparameters in enumerate(edge_hyperparameters_for_all_layers):
                current_layer_all_edges_dict = all_edges_hyperparameters_list[index]
                current_layer_all_edges_dict[edge_type] = edge_layer_hyperparameters

        return all_edges_hyperparameters_list

    def conv_forward(self, useful_data, conv_layer):
        to_add = {"x_dict": "x", "edge_index_dict": "edge_index",
                  "edge_attr_dict": "edge_attr"}

        for hetero_value_name, homo_value_name in to_add.items():
            if hasattr(useful_data, hetero_value_name):
                useful_data[homo_value_name] = useful_data[hetero_value_name]

        self.homo_convolution_class.conv_forward(self, useful_data, conv_layer)

        for hetero_value_name, homo_value_name in to_add.items():
            if hasattr(useful_data, hetero_value_name):
                del useful_data[homo_value_name]

    # # returns a list of dictionaries of dictionaries .
    # def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
    #     hyperparameters_for_each_layer = list()
    #     for current_layer_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
    #         layer_hyperparameters_for_each_edge_type = dict()
    #         for edge_type in pyg_data.edge_types:
    #             current_hyperparameters = current_layer_hyperparameters[edge_type]
    #
    #             layer_hyperparameters_for_edge_type = dict()
    #             if len(hyperparameters_for_each_layer) == 0:
    #                 layer_hyperparameters_for_edge_type["in_channels"] = in_channels
    #             else:
    #                 prev_layer = hyperparameters_for_each_layer[-1][edge_type]
    #                 layer_hyperparameters_for_edge_type["in_channels"] = prev_layer["hidden_channels"] \
    #                                                                      * prev_layer["heads"]
    #             layer_hyperparameters_for_edge_type["hidden_channels"] = current_hyperparameters["hidden_channels"]
    #             layer_hyperparameters_for_edge_type["heads"] = current_hyperparameters["heads"]
    #             layer_hyperparameters_for_edge_type["dropout"] = current_hyperparameters["dropout"]
    #
    #             layer_hyperparameters_for_edge_type["edge_dim"] = model_parameters[edge_type]["edge_dim"]
    #             layer_hyperparameters_for_each_edge_type[edge_type] = layer_hyperparameters_for_edge_type
    #         hyperparameters_for_each_layer.append(layer_hyperparameters_for_each_edge_type)
    #     return hyperparameters_for_each_layer
