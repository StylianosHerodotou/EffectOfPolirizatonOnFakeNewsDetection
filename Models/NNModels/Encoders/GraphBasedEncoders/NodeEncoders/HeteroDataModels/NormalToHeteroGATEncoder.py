from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import HeteroConv
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.AbstractNodeGNNEncoder import AbstractNodeGNNEncoder


class NormalToHeteroGATEncoder(AbstractNodeGNNEncoder):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        conv_dict = dict()
        for edge_type in pyg_data.edge_types:
            edge_feature_dim=None
            if "edge_attr" in pyg_data[edge_type].keys():
                edge_feature_dim = pyg_data[edge_type]["edge_attr"].size()[1]

            conv_dict[edge_type] = GATv2Conv(in_channels=layer_hyperparameters["in_channels"],
                                             out_channels= layer_hyperparameters["hidden_channels"],
                                             heads=layer_hyperparameters["heads"],
                                             dropout=layer_hyperparameters["dropout"],
                                             edge_dim=edge_feature_dim,
                                             add_self_loops=False)

        return HeteroConv(conv_dict, aggr=aggr_type)

    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = []
        for current_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
            layer_hyperparameters = dict()
            if len(hyperparameters_for_each_layer)== 0:
                layer_hyperparameters["in_channels"] = in_channels
            else:
                prev_layer = hyperparameters_for_each_layer[-1]
                layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"] * prev_layer["heads"]

            layer_hyperparameters["hidden_channels"] = current_hyperparameters["hidden_channels"]
            layer_hyperparameters["heads"] = current_hyperparameters["heads"]
            layer_hyperparameters["dropout"] = current_hyperparameters["dropout"]

            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer


    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def forward(self, pyg_data):
        x_dict, edge_index_dict, edge_attr_dict = pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.edge_attr_dict
        for index, conv_layer in enumerate(self.convs):
            x_dict = conv_layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x for key, x in x_dict.items()}
        return x_dict




#
# class NormalToHeteroGATEncoder(torch.nn.Module):
#     def __init__(self, in_channels, pyg_data,
#                  hidden_channels=128, heads=8, dropout=0.2,
#                  num_layers=1):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#
#         first_conv = get_single_normal_to_hetero_gat_layer(pyg_data, in_channels,
#                                                            hidden_channels, heads, dropout)
#         self.convs.append(first_conv)
#
#         for layer in range(num_layers - 1):
#             current_conv = get_single_normal_to_hetero_gat_layer(pyg_data, in_channels=hidden_channels * heads,
#                                                                  hidden_channels=hidden_channels, heads=heads,
#                                                                  dropout=dropout)
#             self.convs.append(current_conv)
#
#     def forward(self, pyg_data):
#         x_dict, edge_index_dict, edge_attr_dict = pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.edge_attr_dict
#         for index, conv_layer in enumerate(self.convs):
#             x_dict = conv_layer(x_dict, edge_index_dict, edge_attr_dict)
#             x_dict = {key: x for key, x in x_dict.items()}
#         return x_dict