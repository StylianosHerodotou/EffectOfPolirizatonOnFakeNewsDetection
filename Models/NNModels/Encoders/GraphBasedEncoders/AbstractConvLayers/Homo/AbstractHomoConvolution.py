from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHomoGNNEncoder import AbstractHomogeneousGNNEncoder


class AbstractHomoConvolution(AbstractHomogeneousGNNEncoder, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)