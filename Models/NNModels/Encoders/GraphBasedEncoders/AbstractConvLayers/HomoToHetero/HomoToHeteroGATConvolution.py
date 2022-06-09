from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.Homo.HomogeneousGATConvolution import \
    HomogeneousGATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.HomoToHetero. \
    AbstractHomoToHeteroConvolution import AbstractHomoToHeteroConvolution


# TODO MAKE THIS WORK.
class HomoToHeteroGATConvolution(AbstractHomoToHeteroConvolution,
                                 HomogeneousGATConvolution, ABC):
    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
