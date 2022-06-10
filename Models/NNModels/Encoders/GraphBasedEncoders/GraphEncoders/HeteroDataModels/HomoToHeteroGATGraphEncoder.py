from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.HomoToHetero.HomoToHeteroGATConvolution import \
    HomoToHeteroGATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.HeteroSAGPoolingMeanAggregator import \
    HeteroSAGPoolingMeanAggregator
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HeteroDataModels. \
    AbstractHeterogeneousGraphGNNEncoder import AbstractHeterogeneousGraphGNNEncoder


class HomoToHeteroGATGraphEncoder(HomoToHeteroGATConvolution, AbstractHeterogeneousGraphGNNEncoder,
                                  HeteroSAGPoolingMeanAggregator):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

# from torch_geometric.nn import GATv2Conv, TopKPooling, HeteroConv
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# import torch.nn.functional as F
# import torch
# from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder
#
#
# class NormalToHeteroGATGraphEncoder(AbstractGraphGNNEncoder):
#
#     def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
#         conv_dict = dict()
#         for edge_type in pyg_data.edge_types:
#             edge_feature_dim = None
#             if "edge_attr" in pyg_data[edge_type].keys():
#                 edge_feature_dim = pyg_data[edge_type]["edge_attr"].size()[1]
#
#             conv_dict[edge_type] = GATv2Conv(in_channels=layer_hyperparameters["in_channels"],
#                                              out_channels=layer_hyperparameters["hidden_channels"],
#                                              heads=layer_hyperparameters["heads"],
#                                              dropout=layer_hyperparameters["dropout"],
#                                              edge_dim=edge_feature_dim,
#                                              add_self_loops=False)
#
#         return HeteroConv(conv_dict, aggr=aggr_type)
#
#     def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
#         hyperparameters_for_each_layer = []
#         for current_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
#             layer_hyperparameters = dict()
#             if len(hyperparameters_for_each_layer) == 0:
#                 layer_hyperparameters["in_channels"] = in_channels
#             else:
#                 prev_layer = hyperparameters_for_each_layer[-1]
#                 layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"] * prev_layer["heads"]
#
#             layer_hyperparameters["hidden_channels"] = current_hyperparameters["hidden_channels"]
#             layer_hyperparameters["heads"] = current_hyperparameters["heads"]
#             layer_hyperparameters["dropout"] = current_hyperparameters["dropout"]
#
#             hyperparameters_for_each_layer.append(layer_hyperparameters)
#         return hyperparameters_for_each_layer
#
#     def generate_pool_layer(self, pyg_data, layer_hyperparameters):
#         pooling_layer = TopKPooling(in_channels=layer_hyperparameters["in_channels"],
#                                     ratio=layer_hyperparameters["ratio"])
#         return pooling_layer
#
#     def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters):
#         conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
#                                                                                                 model_parameters)
#
#         hyperparameters_for_each_layer = []
#         for index, current_hyperparameters in enumerate(model_parameters["hyper_parameters_for_each_layer"]):
#             layer_hyperparameters = dict()
#             current_conv_hyperparameters = conv_hyperparameters_for_each_layer[index]
#
#             layer_hyperparameters["in_channels"] = current_conv_hyperparameters["hidden_channels"] * \
#                                                    current_conv_hyperparameters["heads"]
#             layer_hyperparameters["ratio"] = model_parameters["pooling_ratio"]
#             hyperparameters_for_each_layer.append(layer_hyperparameters)
#         return hyperparameters_for_each_layer
#
#     def __init__(self, in_channels, pyg_data, model_parameters):
#         super().__init__(in_channels, pyg_data, model_parameters)
#
#     def forward(self, pyg_data):
#         x, edge_index, edge_attr, batch = pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, pyg_data.batch
#
#         x = F.relu(self.convs[0](x, edge_index, edge_attr))
#         x, edge_index, edge_attr, batch, _, _ = self.pools[0](x, edge_index, edge_attr, batch)
#         cont = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#
#         for layer_index in range(1, self.num_layers):
#             x = F.relu(self.convs[layer_index](x, edge_index, edge_attr))
#             x, edge_index, edge_attr, batch, _, _ = self.pools[layer_index](x, edge_index, edge_attr, batch)
#             cont += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#
#         cont/= cont.size(0)
#         return cont
