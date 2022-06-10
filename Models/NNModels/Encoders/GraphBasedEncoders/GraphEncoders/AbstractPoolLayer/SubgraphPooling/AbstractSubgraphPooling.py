from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractGraphGNNEncoder import AbstractGraphGNNEncoder


class AbstractSubgraphPooling(AbstractGraphGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    @abstractmethod
    def update_vector_representation(self, useful_data, vector_representation):
        pass

    @abstractmethod
    def final_update_vector_representation(self, vector_representation):
        pass

    def forward(self, pyg_data):
        useful_data = self.extract_useful_data_from_input(pyg_data)
        print(useful_data.keys())

        vector_representation = None
        for conv_layer, pool_layer in zip(self.convs, self.pools):
            useful_data = self.conv_forward(useful_data, conv_layer)
            # print("after conv", useful_data.keys())

            useful_data = self.activation_forward(useful_data)
            # print("after activation", useful_data.keys())

            useful_data = self.pool_forward(useful_data, pool_layer)
            # print("after pooling", useful_data.keys())

            if conv_layer != self.convs[-1]:
                useful_data = self.extra_dropout_forward(useful_data)
            vector_representation = self.update_vector_representation(useful_data, vector_representation)
            # print("after vector representation", useful_data.keys())

        vector_representation = self.final_update_vector_representation(vector_representation)

        return vector_representation
