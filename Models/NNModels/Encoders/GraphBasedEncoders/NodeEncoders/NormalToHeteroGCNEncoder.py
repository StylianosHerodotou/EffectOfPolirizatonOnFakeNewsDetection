from torch_geometric.nn import GCNConv
from torch_geometric.nn import HeteroConv
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.AbstractNodeGNNEncoder import AbstractNodeGNNEncoder


class NormalToHeteroGCNEncoder(AbstractNodeGNNEncoder):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters):
        conv_dict = dict()
        for edge_type in pyg_data.edge_types:
            conv_dict[edge_type] = GCNConv(in_channels=layer_hyperparameters["in_channels"],
                                           out_channels=layer_hyperparameters["out_channels"],
                                           improved=True)
        return HeteroConv(conv_dict)

    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = []
        for index, hidden in enumerate(model_parameters["nodes_per_hidden_layer"]):
            layer_hyperparameters = dict()
            if (index == 0):
                layer_hyperparameters["in_channels"] = in_channels
            else:
                layer_hyperparameters["in_channels"] = hyperparameters_for_each_layer[index - 1]["out_channels"]
            layer_hyperparameters["out_channels"] = hidden
            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer

    def add_conv_layers(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            new_conv_layer = self.generate_conv_layer(pyg_data, layer_hyperparameters)
            self.convs.append(new_conv_layer)

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def forward(self, pyg_data):
        x_dict, edge_index_dict = pyg_data.x_dict, pyg_data.edge_index_dict
        for index, conv_layer in enumerate(self.convs):
            x_dict = conv_layer(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items()}
        return x_dict
