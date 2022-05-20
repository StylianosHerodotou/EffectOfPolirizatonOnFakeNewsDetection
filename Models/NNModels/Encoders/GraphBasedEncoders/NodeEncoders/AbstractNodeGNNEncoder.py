from Models.NNModels.Encoders.GraphBasedEncoders.AbstractGNNEncoder import AbstractGNNEncoder


class AbstractNodeGNNEncoder(AbstractGNNEncoder):

    def __init__(self, in_channels, pyg_data,model_parameters):
        super().__init__(in_channels, pyg_data,model_parameters)
