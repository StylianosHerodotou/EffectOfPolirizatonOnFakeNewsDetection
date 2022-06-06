from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling. \
    AbstractPoolingImplementation.AbstractMEMPoolingMethod import \
    AbstractMEMPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling. \
    CompleteAtEndPoolingMethods.HeteroDataAtEndPoolingMethods.AbstractHeteroAtEndPooling import \
    AbstractHeteroAtEndPooling
from torch_geometric.nn import MemPooling


class PerNodeTypeMEMPooling(AbstractHeteroAtEndPooling, AbstractMEMPoolingMethod, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters,
                                                     conv_hyperparameters_for_each_layer=None):
        if conv_hyperparameters_for_each_layer is None:
            conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
                                                                                                    pyg_data,
                                                                                                    model_parameters)
        last_conv_layer_hyperparameters = conv_hyperparameters_for_each_layer[-1]

        for node_type, node_conv_hyperparameters in last_conv_layer_hyperparameters.items():
            first_layer_input = node_conv_hyperparameters["hidden_channels"] * \
                                node_conv_hyperparameters["heads"]
            break

        print("first_layer_input", first_layer_input)

        hyperparameters_for_each_node_type = dict()
        for node_type in pyg_data.node_types:
            pooling_hyper_parameters_for_each_layer = model_parameters[node_type]
            hyperparameters_for_each_layer_for_current_type = list()
            for current_hyperparameters in pooling_hyper_parameters_for_each_layer:
                layer_hyperparameters = dict()

                if len(hyperparameters_for_each_layer_for_current_type) == 0:
                    layer_hyperparameters["in_channels"] = first_layer_input
                else:
                    prev_layer = hyperparameters_for_each_layer_for_current_type[-1]
                    layer_hyperparameters["in_channels"] = prev_layer["hidden_channels"]

                layer_hyperparameters["hidden_channels"] = current_hyperparameters["pooling_hidden_channels"]
                layer_hyperparameters["heads"] = current_hyperparameters["pooling_heads"]
                layer_hyperparameters["num_clusters"] = current_hyperparameters["pooling_num_clusters"]

                layer_hyperparameters["dropout"] = current_hyperparameters["pooling_dropout"]

                hyperparameters_for_each_layer_for_current_type.append(layer_hyperparameters)
            hyperparameters_for_each_node_type[node_type] = hyperparameters_for_each_layer_for_current_type

        hyperparameters_for_each_layer = list()
        for node_type, node_hyperparameters_for_all_layers in hyperparameters_for_each_node_type.items():
            # fill list with empty dictionaries.
            if len(hyperparameters_for_each_layer) == 0:
                for index in range(len(node_hyperparameters_for_all_layers)):
                    hyperparameters_for_each_layer.append(dict())

            for index, node_layer_hyperparameters in enumerate(node_hyperparameters_for_all_layers):
                current_layer_all_nodes_dict = hyperparameters_for_each_layer[index]
                current_layer_all_nodes_dict[node_type] = node_layer_hyperparameters

        print("hyperparameters_for_each_layer\n\n", hyperparameters_for_each_layer)

        return hyperparameters_for_each_layer

    def pool_forward(self, useful_data, pool_layer, find_loss=False):
        x_dict = useful_data.x_dict
        for key, x in x_dict.items():
            x, pooling_loss = pool_layer(x)
            x_dict[key] = x
            pooling_loss = MemPooling.kl_loss(pooling_loss)
            if "pooling_loss" not in useful_data.to_dict().keys():
                useful_data.pooling_loss = 0
            useful_data.pooling_loss = useful_data.pooling_loss + pooling_loss

        # useful_data.x_dict = x_dict
        return useful_data

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
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
