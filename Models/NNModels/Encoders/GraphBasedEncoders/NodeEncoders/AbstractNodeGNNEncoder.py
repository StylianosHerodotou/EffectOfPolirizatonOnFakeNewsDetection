from abc import ABC, abstractmethod

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractNodeGNNEncoder(AbstractGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def extract_embeddings_from_useful_data(self, useful_data):
        pass

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        for conv_layer in self.convs:
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            if conv_layer != self.convs[-1]:
                useful_data = self.extra_dropout_forward(useful_data)
        return self.extract_embeddings_from_useful_data(useful_data)
