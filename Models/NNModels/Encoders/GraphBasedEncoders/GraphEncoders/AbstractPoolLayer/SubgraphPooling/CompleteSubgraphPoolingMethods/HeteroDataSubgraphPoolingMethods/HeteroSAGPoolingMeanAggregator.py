from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    AbstractPoolingImplementation.AbstractSAGPoolingMethod import \
    AbstractSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    CompleteSubgraphPoolingMethods.HeteroDataSubgraphPoolingMethods. \
    AbstractHeteroVectorAggregationMethod import AbstractHeteroVectorAggregationMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    VectorAggregationMethods.AbstractMeanAggregator import \
    AbstractMeanAggregator


class HeteroSAGPoolingMeanAggregator(AbstractSAGPoolingMethod, AbstractHeteroVectorAggregationMethod,
                                     AbstractMeanAggregator, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters, **kwargs):
        conv_hyperparameters_for_each_layer_for_all_edge_type = self.generate_hyperparameters_for_each_conv_layer(in_channels,pyg_data,model_parameters)
        conv_hyperparameters_per_edge_type = dict()
        for edge_type in pyg_data.edge_types:
            conv_hyperparameters_per_edge_type[edge_type] = list()

        for all_edge_layer_hyperparameeters in conv_hyperparameters_for_each_layer_for_all_edge_type:
            for edge_type, edge_hyperparameters in all_edge_layer_hyperparameeters.items():
                conv_hyperparameters_per_edge_type[edge_type].append(edge_hyperparameters)

        # returns a list( layer) of dictionaries(edge_type)  of dictionaries (hyperparameters) .
        all_edges_hyperparameters_dict = dict()
        for edge_type in pyg_data.edge_types:
            current_edge_pyg_data = pyg_data[edge_type]
            current_edge_model_parameters = model_parameters[edge_type]
            hyperparameters_for_each_layer = conv_hyperparameters_per_edge_type[edge_type]

            hyperparameters_for_this_edge_type = super().\
                generate_hyperparameters_for_each_pool_layer(in_channels,
                                                             pyg_data=current_edge_pyg_data,
                                                             model_parameters=current_edge_model_parameters,
                                                             hyperparameters_for_each_layer=hyperparameters_for_each_layer)
            all_edges_hyperparameters_dict[edge_type] = hyperparameters_for_this_edge_type

        all_edges_hyperparameters_list = list()
        for edge_type, edge_hyperparameters_for_all_layers in all_edges_hyperparameters_dict.items():
            # fill list with empty dictionaries.
            if len(all_edges_hyperparameters_list) == 0:
                for index in range(len(edge_hyperparameters_for_all_layers)):
                    all_edges_hyperparameters_list.append(dict())

            for index, edge_layer_hyperparameters in enumerate(edge_hyperparameters_for_all_layers):
                current_layer_all_edges_dict = all_edges_hyperparameters_list[index]
                current_layer_all_edges_dict[edge_type] = edge_layer_hyperparameters

        return all_edges_hyperparameters_list

