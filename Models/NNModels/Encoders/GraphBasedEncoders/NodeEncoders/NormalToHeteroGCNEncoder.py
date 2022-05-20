from torch_geometric.nn import SAGEConv
from torch_geometric.nn import HeteroConv
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.AbstractNodeGNNEncoder import AbstractNodeGNNEncoder


class NormalToHeteroGCNEncoder(AbstractNodeGNNEncoder):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        conv_dict = dict()
        for edge_type in pyg_data.edge_types:
            conv_dict[edge_type] = SAGEConv(in_channels=layer_hyperparameters["in_channels"],
                                           out_channels=layer_hyperparameters["hidden_channels"],
                                           improved=True)
        return HeteroConv(conv_dict, aggr=aggr_type)

    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):

        hyperparameters_for_each_layer = []
        for current_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
            layer_hyperparameters = dict()
            if len(hyperparameters_for_each_layer) == 0:
                layer_hyperparameters["in_channels"] = in_channels
            else:
                prev_layer = hyperparameters_for_each_layer[-1]
                layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"]
            layer_hyperparameters["hidden_channels"] = current_hyperparameters["hidden_channels"]
            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer



    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def forward(self, pyg_data):
        x_dict, edge_index_dict = pyg_data.x_dict, pyg_data.edge_index_dict
        for index, conv_layer in enumerate(self.convs):
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items()}
        return x_dict
