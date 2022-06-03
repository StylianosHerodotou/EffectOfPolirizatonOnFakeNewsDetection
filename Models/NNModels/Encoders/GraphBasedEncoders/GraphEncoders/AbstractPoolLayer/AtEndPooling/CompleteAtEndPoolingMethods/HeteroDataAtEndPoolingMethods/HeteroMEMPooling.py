from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling. \
    AbstractPoolingImplementation.AbstractMEMPoolingMethod import \
    AbstractMEMPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling. \
    CompleteAtEndPoolingMethods.HeteroDataAtEndPoolingMethods.AbstractHeteroAtEndPooling import \
    AbstractHeteroAtEndPooling
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHomoGNNEncoder import AbstractHomogeneousGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeteroGNNEncoder import AbstractHeteroGNNEncoder

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

        print("first_layer_input", first_layer_input)

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

    def activation_forward(self, useful_data):
        if "hetero_x" not in useful_data.to_dict().keys():
            to_return = AbstractHeteroGNNEncoder.activation_forward(self,useful_data)
        else:
            to_return= AbstractHomogeneousGNNEncoder.activation_forward(self,useful_data)

        return to_return

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        for conv_layer in self.convs:
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            if conv_layer != self.convs[-1]:
                useful_data = self.extra_dropout_forward(useful_data)

        if not self.is_homogeneous_data:
            useful_data = useful_data.to_homogeneous()
            useful_data.hetero_x = useful_data.x

        for index, pool_layer in enumerate(self.pools):
            useful_data = self.pool_forward(useful_data, pool_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.extra_pooling_dropout_forward(useful_data, index)

        return self.get_vector_representation(useful_data)










    # def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data,
    #                                                  model_parameters,
    #                                                  conv_hyperparameters_for_each_layer=None):
    #
    #     conv_hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels,
    #                                                                                             pyg_data,
    #                                                                                             model_parameters)
    #     last_conv_layer = conv_hyperparameters_for_each_layer[-1]
    #     for edge_type, last_conv_layer in  last_conv_layer.items():
    #         break
    #
    #     all_nodes_hyperparameters_dict = dict()
    #     for node_type in pyg_data.node_types:
    #         current_node_pyg_data = pyg_data[node_type]
    #         current_node_model_parameters = model_parameters[node_type]
    #         current_node_last_conv_layer_hyperparameters = last_conv_layer[node_type]
    #
    #         hyperparameters_for_this_node_type = super().\
    #             generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data=current_node_pyg_data,
    #                                                          model_parameters=current_node_model_parameters,
    #                                                          last_conv_layer_hyperparameters= current_node_last_conv_layer_hyperparameters)
    #
    #         all_nodes_hyperparameters_dict[node_type] = hyperparameters_for_this_node_type
    #
    #     all_nodes_hyperparameters_list = list()
    #     for node_type, node_hyperparameters_for_all_layers in all_nodes_hyperparameters_dict.items():
    #         # fill list with empty dictionaries.
    #         if len(all_nodes_hyperparameters_list) == 0:
    #             for index in range(len(node_hyperparameters_for_all_layers)):
    #                 all_nodes_hyperparameters_list.append(dict())
    #
    #         for index, node_layer_hyperparameters in enumerate(node_hyperparameters_for_all_layers):
    #             current_layer_all_nodes_dict = all_nodes_hyperparameters_list[index]
    #             current_layer_all_nodes_dict[node_type] = node_layer_hyperparameters
    #
    #     return all_nodes_hyperparameters_list
