from Models.NNModels.Encoders.GraphBasedEncoders.AbstractConvLayers.HeteroWithHomoData.HEATConvolution import HEATConvolution
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.HomoMEMPooling import HomoMEMPooling
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomoDataModels.AbstractHomogeneousGraphGNNEncoder import \
    AbstractHomogeneousGraphGNNEncoder


class HEATGraphEncoder(AbstractHomogeneousGraphGNNEncoder,HEATConvolution ,HomoMEMPooling):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)