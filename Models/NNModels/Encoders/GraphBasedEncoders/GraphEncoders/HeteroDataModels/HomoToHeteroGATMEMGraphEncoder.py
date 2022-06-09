from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.HomoToHetero.HomoToHeteroGATConvolution import \
    HomoToHeteroGATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.HeteroMEMPooling import HeteroMEMPooling
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HeteroDataModels. \
    AbstractHeterogeneousGraphGNNEncoder import AbstractHeterogeneousGraphGNNEncoder


class HomoToHeteroGATMEMGraphEncoder(HomoToHeteroGATConvolution, AbstractHeterogeneousGraphGNNEncoder, HeteroMEMPooling):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)
