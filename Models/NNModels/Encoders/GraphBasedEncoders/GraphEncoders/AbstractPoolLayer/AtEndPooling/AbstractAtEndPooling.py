from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractAtEndPooling(AbstractGraphGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def get_vector_representation(self, useful_data):
        pass

    @abstractmethod
    def get_pooling_additional_loss(self, useful_data):
        pass

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        for conv_layer in self.convs:
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.extra_dropout_forward(useful_data)

        for pool_layer in self.pools:
            useful_data = self.pool_forward(useful_data, pool_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.extra_dropout_forward(useful_data)

        return self.get_vector_representation(useful_data)
