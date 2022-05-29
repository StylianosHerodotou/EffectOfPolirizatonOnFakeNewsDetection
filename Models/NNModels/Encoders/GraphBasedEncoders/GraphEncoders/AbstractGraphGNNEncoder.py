import torch
from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractGraphGNNEncoder(AbstractGNNEncoder):
    @abstractmethod
    def generate_pool_layer(self, pyg_data, layer_hyperparameters):
        pass

    @abstractmethod
    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters):
        pass

    def add_pool_layers(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            new_pool_layer = self.generate_pool_layer(pyg_data, layer_hyperparameters)
            self.pools.append(new_pool_layer)

    @abstractmethod
    def get_pool_input(self, useful_data):
        pass

    @abstractmethod
    def get_useful_pool_result_data(self, useful_data, all_data):
        pass

    @abstractmethod
    def pool_forward(self, useful_data, pool_layer):
        pass

    @abstractmethod
    def update_vector_representation(self, useful_data, vector_representation):
        pass

    @abstractmethod
    def final_update_vector_representation(self, vector_representation):
        pass

    def __init__(self, in_channels, pyg_data,model_parameters):
        super().__init__(in_channels, pyg_data,model_parameters)
        self.pools = torch.nn.ModuleList()
        self.add_pool_layers(in_channels, pyg_data, model_parameters)


    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        vector_representation=None
        for conv_layer, pool_layer in zip(self.convs, self.pools):
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.pool_forward(useful_data, pool_layer)
            vector_representation = self.update_vector_representation(useful_data,vector_representation)

        vector_representation = self.final_update_vector_representation(useful_data,vector_representation)
        return vector_representation
