from torch_geometric.nn import GATv2Conv, SAGPooling
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomoDataModels.AbstractHomogeneousGraphGNNEncoder import \
    AbstractHomogeneousGraphGNNEncoder


class HomogeneousGATGraphEncoder(AbstractHomogeneousGraphGNNEncoder):

    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        conv_layer = GATv2Conv(in_channels=layer_hyperparameters["in_channels"],
                               hidden_size=layer_hyperparameters["hidden_channels"],
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

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pooling_layer = SAGPooling(in_channels=layer_hyperparameters["in_channels"],
                                   ratio=layer_hyperparameters["ratio"])
        return pooling_layer

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters):
        conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
                                                                                                model_parameters)

        hyperparameters_for_each_layer = []
        for index, current_hyperparameters in enumerate(model_parameters["hyper_parameters_for_each_layer"]):
            conv_hyperparameters_for_current_layer = conv_hyperparameters_for_each_layer[index]
            layer_hyperparameters = dict()

            layer_hyperparameters["in_channels"] = conv_hyperparameters_for_current_layer["hidden_channels"] * \
                                                   conv_hyperparameters_for_current_layer["heads"]
            layer_hyperparameters["ratio"] = model_parameters["pooling_dropout"]

            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def get_pool_input(self, useful_data):
        pool_input = useful_data.x, useful_data.edge_index, useful_data.edge_attr, useful_data.batch
        return pool_input

    def get_useful_pool_result_data(self, useful_data, all_data):
        x, edge_index, edge_attr, batch, _, _ = all_data

        useful_data.x = x
        useful_data.edge_index = edge_index
        useful_data.edge_attr = edge_attr
        useful_data.batch = batch
        return useful_data

    def get_conv_input(self, useful_data):
        conv_input = useful_data.x, useful_data.edge_index, useful_data.edge_attr
        return conv_input
