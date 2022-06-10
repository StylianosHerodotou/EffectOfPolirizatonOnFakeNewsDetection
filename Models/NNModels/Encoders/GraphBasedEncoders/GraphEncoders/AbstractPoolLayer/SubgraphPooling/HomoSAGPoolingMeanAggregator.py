from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.PoolingMethods.HomoSAGPoolingMethod import \
    HomoSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.VectorAggregationMethods.HomoMeanVectorAggregationMethod import \
    HomoMeanVectorAggregationMethod


class HomoSAGPoolingMeanAggregator(HomoSAGPoolingMethod, HomoMeanVectorAggregationMethod, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
