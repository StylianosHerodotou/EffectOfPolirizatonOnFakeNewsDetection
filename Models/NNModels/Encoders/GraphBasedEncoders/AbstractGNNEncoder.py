import torch
from abc import ABC, abstractmethod
import copy

def copy_dict(original_dict):
    object_dict = dict()
    for key, value in original_dict.items():
        if isinstance(value, dict):
            object_dict[key] = copy_dict(value)
        if isinstance(value, torch.Tensor):
            object_dict[key] = torch.clone(value)
        else:
            object_dict[key] = copy.copy(value)

    return object_dict


class AbstractGNNEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def generate_conv_layer(self, pyg_data, layer_hyperparameters, aggr_type="mean"):
        pass

    @abstractmethod
    def extract_useful_data_from_input(self, pyg_data):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_conv_layer(self, in_channels, pyg_data, model_parameters):
        pass

    @abstractmethod
    def conv_forward(self, useful_data, conv_layer):
        pass

    def add_conv_layers(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_conv_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            new_conv_layer = self.generate_conv_layer(pyg_data, layer_hyperparameters)
            self.convs.append(new_conv_layer)

    def clone_dictionary(self, dic):
        new_dic = copy.copy(dic)
        for key, item in new_dic:
            if isinstance(item, dict):
                new_dic[key] =  self.clone_dictionary(item)
            elif isinstance(item, torch.Tensor):
                new_dic[key] = torch.clone(item)
        return new_dic


    @abstractmethod
    def activation_forward(self, useful_data):
        pass

    def extra_dropout_forward(self, useful_data):
        return useful_data

    @abstractmethod
    def forward(self, pyg_data):
        pass

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__()
        self.activation_function = model_parameters["activation_function"]
        self.convs = torch.nn.ModuleList()
        self.add_conv_layers(in_channels, pyg_data, model_parameters)
