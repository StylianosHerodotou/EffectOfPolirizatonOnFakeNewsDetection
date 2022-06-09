from abc import ABC
from torch_geometric.nn import MemPooling
import torch.nn.functional as F
from Models.NNModels.Encoders.GraphBasedEncoders.\
    GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class HomoMEMPooling(AbstractGraphGNNEncoder, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.pooling_dropout = list()
        self.get_pooling_dropout(in_channels, pyg_data, model_parameters)

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     last_conv_layer_hyperparameters=None):
        if last_conv_layer_hyperparameters is None:
            conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
                                                                                                    pyg_data,
                                                                                                    model_parameters)
            last_conv_layer_hyperparameters = conv_hyperparameters_for_each_layer[-1]

        hyperparameters_for_each_layer = []
        for current_hyperparameters in model_parameters["pooling_hyper_parameters_for_each_layer"]:
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

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pooling_layer = MemPooling(in_channels=layer_hyperparameters["in_channels"],
                                   out_channels=layer_hyperparameters["hidden_channels"],
                                   heads=layer_hyperparameters["heads"],
                                   num_clusters=layer_hyperparameters["num_clusters"])
        return pooling_layer

    def get_vector_representation(self, useful_data):
        x = useful_data["x"]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x

    @staticmethod
    def get_pooling_additional_loss(useful_data):
        pooling_loss = useful_data["pooling_loss"]
        return pooling_loss

    def extra_pooling_dropout_forward(self, useful_data, index):
        x = useful_data["x"]
        dropout_value = self.pooling_dropout[index]
        x = F.dropout(x, p=dropout_value)
        useful_data["x"] = x
        return useful_data

    def pool_forward(self, useful_data, pool_layer, find_loss=False):
        x = useful_data["x"]
        x, pooling_loss = pool_layer(x)
        useful_data["x"] = x

        pooling_loss = MemPooling.kl_loss(pooling_loss)
        useful_data["pooling_loss"] = useful_data["pooling_loss"] + pooling_loss
        return useful_data

    def get_pooling_dropout(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            dropout = 0
            if "dropout" in layer_hyperparameters.keys():
                dropout = layer_hyperparameters["dropout"]
            self.pooling_dropout.append(dropout)

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        useful_data["pooling_loss"] = 0
        for conv_layer in self.convs:
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            if conv_layer != self.convs[-1]:
                useful_data = self.extra_dropout_forward(useful_data)

        for index, pool_layer in enumerate(self.pools):
            useful_data = self.pool_forward(useful_data, pool_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.extra_pooling_dropout_forward(useful_data, index)

        return self.get_vector_representation(useful_data)
