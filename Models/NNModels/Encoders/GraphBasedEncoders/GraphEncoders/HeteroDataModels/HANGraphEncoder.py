from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.HomoToHetero.HomoToHeteroGATConvolution import \
    HomoToHeteroGATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.CompleteSubgraphPoolingMethods.HeteroDataSubgraphPoolingMethods.HeteroSAGPoolingMeanAggregator import \
    HeteroSAGPoolingMeanAggregator
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HeteroDataModels. \
    AbstractHeterogeneousGraphGNNEncoder import AbstractHeterogeneousGraphGNNEncoder


class HANGraphEncoder(HomoToHeteroGATConvolution, AbstractHeterogeneousGraphGNNEncoder,
                      HeteroSAGPoolingMeanAggregator):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)