from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.Homo.HomogeneousGATConvolution import \
    HomogeneousGATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.HomoSAGPoolingMeanAggregator import \
    HomoSAGPoolingMeanAggregator
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomoDataModels.AbstractHomogeneousGraphGNNEncoder import \
    AbstractHomogeneousGraphGNNEncoder


class HomogeneousGATGraphEncoder(AbstractHomogeneousGraphGNNEncoder, HomogeneousGATConvolution,
                                 HomoSAGPoolingMeanAggregator):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
