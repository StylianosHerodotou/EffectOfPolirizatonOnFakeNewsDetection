from abc import ABC

from Models.NNModels.Encoders.GraphBasedEncoders.AbstractHeteroGNNEncoder import AbstractHeteroGNNEncoder
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.AbstractNodeGNNEncoder import AbstractNodeGNNEncoder


class AbstractHeterogeneousNodeGNNEncoder(AbstractNodeGNNEncoder, AbstractHeteroGNNEncoder, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def extract_embeddings_from_useful_data(self, useful_data):
        return useful_data.x_dict
