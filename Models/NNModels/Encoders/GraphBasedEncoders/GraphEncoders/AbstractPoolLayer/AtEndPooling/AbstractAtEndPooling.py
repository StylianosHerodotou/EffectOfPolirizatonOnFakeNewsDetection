from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractAtEndPooling(AbstractGraphGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.pooling_dropout = list()
        self.get_pooling_dropout(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def get_single_vector_representation(self, useful_data):
        pass

    @abstractmethod
    def get_vector_representation(self, useful_data):
        pass

    @abstractmethod
    def get_single_pooling_additional_loss(self, useful_data):
        pass

    @abstractmethod
    def get_pooling_additional_loss(self, useful_data):
        pass

    @abstractmethod
    def single_extra_pooling_dropout_forward(self, useful_data, index):
        pass

    @abstractmethod
    def extra_pooling_dropout_forward(self, useful_data, index):
        pass

    def get_pooling_dropout(self, in_channels, pyg_data, model_parameters):
        hyperparameters_for_each_layer = self.generate_hyperparameters_for_each_pool_layer(in_channels, pyg_data,
                                                                                           model_parameters)
        for layer_hyperparameters in hyperparameters_for_each_layer:
            dropout=0
            if "dropout" in  layer_hyperparameters.keys():
                dropout= layer_hyperparameters["dropout"]
            self.pooling_dropout.append(dropout)

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
