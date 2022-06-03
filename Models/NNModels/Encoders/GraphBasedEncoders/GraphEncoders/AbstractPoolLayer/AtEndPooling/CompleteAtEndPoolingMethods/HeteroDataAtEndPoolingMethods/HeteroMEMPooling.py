from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.\
    AbstractPoolingImplementation.AbstractMEMPoolingMethod import \
    AbstractMEMPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.\
    CompleteAtEndPoolingMethods.HeteroDataAtEndPoolingMethods.AbstractHeteroAtEndPooling import AbstractHeteroAtEndPooling


class HeteroMEMPooling(AbstractHeteroAtEndPooling, AbstractMEMPoolingMethod, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     conv_hyperparameters_for_each_layer=None):
        if conv_hyperparameters_for_each_layer is None:
            conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
                                                                                                    pyg_data,
                                                                                                    model_parameters)
        last_conv_layer_hyperparameters = conv_hyperparameters_for_each_layer[-1]
        first_layer_input = 0
        for edge_type, edge_conv_hyperparameters in last_conv_layer_hyperparameters.items():
            first_layer_input+= edge_conv_hyperparameters["hidden_channels"] * \
                               edge_conv_hyperparameters["heads"]

        hyperparameters_for_each_layer = list()
        for current_hyperparameters in model_parameters["pooling_hyper_parameters_for_each_layer"]:
            layer_hyperparameters = dict()

            if len(hyperparameters_for_each_layer) == 0:
                layer_hyperparameters["in_channels"] = first_layer_input
            else:
                prev_layer = hyperparameters_for_each_layer[-1]
                layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"]

            layer_hyperparameters["hidden_channels"] = current_hyperparameters["pooling_hidden_channels"]
            layer_hyperparameters["heads"] = current_hyperparameters["pooling_heads"]
            layer_hyperparameters["num_clusters"] = current_hyperparameters["pooling_num_clusters"]

            layer_hyperparameters["dropout"] = current_hyperparameters["pooling_dropout"]

            hyperparameters_for_each_layer.append(layer_hyperparameters)
        return hyperparameters_for_each_layer
