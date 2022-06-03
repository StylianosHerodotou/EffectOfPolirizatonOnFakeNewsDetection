from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.AbstractAtEndPooling import \
    AbstractAtEndPooling


class AbstractHomoAtEndPooling(AbstractAtEndPooling, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.is_homogeneous_data = True

    def get_vector_representation(self, useful_data):
        return self.get_single_vector_representation(useful_data)

    def get_pooling_additional_loss(self, useful_data):
        return self.get_single_pooling_additional_loss(useful_data)

    def extra_pooling_dropout_forward(self, useful_data, index):
        return self.single_extra_pooling_dropout_forward(useful_data, index)

