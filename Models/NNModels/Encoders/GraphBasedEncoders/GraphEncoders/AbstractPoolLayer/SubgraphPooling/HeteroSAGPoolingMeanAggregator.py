from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.\
    PoolingMethods.HeteroSAGPoolingMethod import \
    HeteroSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.\
    VectorAggregationMethods.HeteroMeanVectorAggregationMethod import \
    HeteroMeanVectorAggregationMethod


class HeteroSAGPoolingMeanAggregator(HeteroSAGPoolingMethod, HeteroMeanVectorAggregationMethod,
                                     ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)





