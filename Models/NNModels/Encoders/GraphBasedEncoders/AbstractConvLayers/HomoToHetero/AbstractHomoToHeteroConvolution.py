from abc import ABC, abstractmethod
from torch_geometric.nn import HeteroConv
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.AbstractConvolution import AbstractConvolution


# this class shows the template. You need to copy paste the below methods to make it work.
class AbstractHomoToHeteroConvolution(AbstractConvolution, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def generate_conv_layer(self, pyg_data, layer_hyperparameters_for_all_edge_types, aggr_type="mean"):
        conv_dict = dict()
        for edge_type in pyg_data.edge_types:
            layer_hyperparameters = layer_hyperparameters_for_all_edge_types[edge_type]
            conv_dict[edge_type] = super().generate_conv_layer(pyg_data,
                                                               layer_hyperparameters,
                                                               aggr_type)

        return HeteroConv(conv_dict, aggr=aggr_type)

    # returns a list( layer) of dictionaries(edge_type)  of dictionaries (hyperparameters) .
    @abstractmethod
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        # for each edge type generate another model_parameters and call super
        all_edges_hyperparameters_dict = dict()
        for edge_type in pyg_data.edge_types:
            current_edge_pyg_data = pyg_data[edge_type]
            current_edge_model_parameters = model_parameters[edge_type]
            hyperparameters_for_this_edge_type = super(). \
                generate_hyperparameters_for_each_conv_layer(in_channels,
                                                             pyg_data=current_edge_pyg_data,
                                                             model_parameters=current_edge_model_parameters)
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
