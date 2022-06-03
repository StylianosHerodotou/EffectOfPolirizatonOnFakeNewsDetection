from abc import ABC, abstractmethod

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.AbstractAtEndPooling import \
    AbstractAtEndPooling
from torch_geometric.nn import MemPooling
import torch.nn.functional as F
import torch


class AbstractMEMPoolingMethod(AbstractAtEndPooling, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pooling_layer = MemPooling(in_channels=layer_hyperparameters["in_channels"],
                                   out_channels=layer_hyperparameters["hidden_channels"],
                                   heads=layer_hyperparameters["heads"],
                                   num_clusters=layer_hyperparameters["num_clusters"])
        return pooling_layer

    @abstractmethod
    def get_node_features(self, useful_data):
        pass

    @abstractmethod
    def set_node_features(self, x, useful_data):
        pass

    def pool_forward(self, useful_data, pool_layer, find_loss=False):
        x = self.get_node_features(useful_data)
        x, pooling_loss = pool_layer(x)
        useful_data = self.set_node_features(x, useful_data)

        pooling_loss = MemPooling.kl_loss(pooling_loss)
        if "pooling_loss" not in useful_data.to_dict().keys():
            useful_data.pooling_loss = 0
        useful_data.pooling_loss = useful_data.pooling_loss + pooling_loss

        return useful_data

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     hyperparameters_for_each_layer=None):
        if hyperparameters_for_each_layer is None:
            conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
                                                                                                    pyg_data,
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

    def get_single_vector_representation(self, useful_data):
        x = self.get_node_features(useful_data)
        x = x.squeeze()
        return torch.flatten(x)

    def get_single_pooling_additional_loss(self, useful_data):
        pooling_loss = useful_data.pooling_loss
        return pooling_loss

    def single_extra_pooling_dropout_forward(self, useful_data, index):
        x = self.get_node_features(useful_data)
        dropout_value = self.pooling_dropout[index]
        x = F.dropout(x, p=dropout_value)
        useful_data = self.set_node_features(x, useful_data)
        return useful_data
