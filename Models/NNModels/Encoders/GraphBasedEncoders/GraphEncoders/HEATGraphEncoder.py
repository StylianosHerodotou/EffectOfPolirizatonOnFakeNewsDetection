from torch_geometric.nn import HEATConv, MemPooling, DeepGCNLayer
from torch.nn import BatchNorm1d, LeakyReLU, Linear
import torch.nn.functional as F

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class HEATGraphEncoder(AbstractGraphGNNEncoder):

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
        #         norm_layer = BatchNorm1d(layer_hyperparameters["hidden_channels"])
        #         activation_layer = LeakyReLU()
        #         complete_layer = DeepGCNLayer(conv_layer, norm_layer, activation_layer,
        #                                       block='res+', dropout=layer_hyperparameters["dropout"])
        #         return complete_layer
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

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pooling_layer = MemPooling(in_channels=layer_hyperparameters["in_channels"],
                                   out_channels=layer_hyperparameters["hidden_channels"],
                                   heads=layer_hyperparameters["heads"],
                                   num_clusters=layer_hyperparameters["num_clusters"])
        return pooling_layer

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters):
        conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
                                                                                                model_parameters)
        last_conv_layer_hyperparameters = conv_hyperparameters_for_each_layer[-1]

        hyperparameters_for_each_layer = []
        for current_hyperparameters in model_parameters["hyper_parameters_for_each_layer"]:
            layer_hyperparameters = dict()

            if len(hyperparameters_for_each_layer) == 0:
                layer_hyperparameters["in_channels"] = last_conv_layer_hyperparameters["hidden_channels"] * \
                                                       last_conv_layer_hyperparameters["heads"]
            else:
                prev_layer = hyperparameters_for_each_layer[-1]
                layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"]

            layer_hyperparameters["hidden_channels"] = current_hyperparameters["pooling_hidden_channels"]
            layer_hyperparameters["heads"] = current_hyperparameters["pooling_heads"]
            layer_hyperparameters["num_clusters"] = current_hyperparameters["pooling_num_clusters"]

            layer_hyperparameters["dropout"] = current_hyperparameters["pooling_dropout"]

            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer

    def get_pooling_dropout(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            self.pooling_dropout.append(layer_hyperparameters["dropout"])

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.pooling_dropout = list()
        self.get_pooling_dropout(in_channels, pyg_data, model_parameters)

    def forward(self, pyg_data):

        x, edge_index, edge_attr, node_type, edge_type = pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, pyg_data.node_type, pyg_data.edge_type
        for index, conv_layer in enumerate(self.convs):
            x = conv_layer(x, edge_index, node_type, edge_type, edge_attr)

        conv_out = x
        kl_loss = 0
        for index, pool_layer in enumerate(self.pools[:-1]):
            x, Si = pool_layer(x)  # add the batches later.
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.pooling_dropout[index])
            kl_loss += MemPooling.kl_loss(Si)
            print(x.size())

        x, Si = self.pools[-1](x)  # add the batches later.
        kl_loss += MemPooling.kl_loss(Si)

        x = x.squeeze(0)

        ##TODO CHANGE THE BELOW
        return (
            x,
            kl_loss
        )