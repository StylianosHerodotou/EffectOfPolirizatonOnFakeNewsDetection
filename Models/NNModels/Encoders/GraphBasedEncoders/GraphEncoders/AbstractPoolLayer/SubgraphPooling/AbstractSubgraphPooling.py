from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractSubgraphPooling(AbstractGraphGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def update_single_vector_representation(self, useful_data, vector_representation):
        pass

    @abstractmethod
    def update_vector_representation(self, useful_data, vector_representation):
        pass

    @abstractmethod
    def final_update_vector_representation(self, vector_representation):
        pass

    @abstractmethod
    def final_update_vector_representation(self, vector_representation):
        pass

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        vector_representation = None
        for conv_layer, pool_layer in zip(self.convs, self.pools):
            useful_data = self.conv_forward(useful_data, conv_layer)
            useful_data = self.activation_forward(useful_data)
            useful_data = self.pool_forward(useful_data, pool_layer)
            if conv_layer != self.convs[-1]:
                useful_data = self.extra_dropout_forward(useful_data)
            vector_representation = self.update_vector_representation(useful_data, vector_representation)
            # print(vector_representation.size())

        vector_representation = self.final_update_vector_representation(vector_representation)
        # print(vector_representation.size())

        return vector_representation
