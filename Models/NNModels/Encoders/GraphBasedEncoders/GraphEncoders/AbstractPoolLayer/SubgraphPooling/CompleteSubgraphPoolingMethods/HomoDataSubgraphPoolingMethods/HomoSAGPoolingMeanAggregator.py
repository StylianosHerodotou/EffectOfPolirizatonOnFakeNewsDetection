from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.\
    AbstractPoolingImplementation.AbstractSAGPoolingMethod import \
    AbstractSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.VectorAggregationMethods.AbstractHomoVectorAggregationMethod import \
    AbstractHomoVectorAggregationMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.\
    VectorAggregationMethods.AbstractMeanAggregator import \
    AbstractMeanAggregator


class HomoSAGPoolingMeanAggregator(AbstractSAGPoolingMethod, AbstractHomoVectorAggregationMethod, AbstractMeanAggregator, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
        self.is_homogeneous_data=True
