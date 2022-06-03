from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.\
    AbstractPoolingImplementation.AbstractMEMPoolingMethod import \
    AbstractMEMPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.\
    CompleteAtEndPoolingMethods.HomoDataAtEndPoolingMethods.AbstractHomoAtEndPooling import AbstractHomoAtEndPooling

class HomoMEMPooling(AbstractHomoAtEndPooling, AbstractMEMPoolingMethod, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
