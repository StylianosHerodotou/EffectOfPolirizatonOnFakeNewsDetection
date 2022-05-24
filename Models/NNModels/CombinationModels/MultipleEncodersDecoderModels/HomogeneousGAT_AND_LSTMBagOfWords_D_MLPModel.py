from Models.NNModels.Classifiers.MLP import MLP
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomogeneousGAT import HomogeneousGAT
import torch

from Models.NNModels.Encoders.TextEncoders.LSTMBagOfWordsEncoder import LSTMBagOfWordsEncoder


class HomogeneousGAT_AND_LSTMBagOfWords_D_MLPModel(torch.nn.Module):
    def __init__(self, encoders_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        self.encoders["HomogeneousGAT"] = HomogeneousGAT(in_channels=encoders_hyperparameters["HomogeneousGAT"]["in_channels"],
                                      edge_dim=encoders_hyperparameters["HomogeneousGAT"]["edge_dim"],
                                      hidden_size=encoders_hyperparameters["HomogeneousGAT"]["hidden_size"],
                                      heads=encoders_hyperparameters["HomogeneousGAT"]["heads"],
                                      dropout=encoders_hyperparameters["HomogeneousGAT"]["dropout"],
                                      pooling_ratio=encoders_hyperparameters["HomogeneousGAT"]["pooling_ratio"],
                                      num_layers=encoders_hyperparameters["HomogeneousGAT"]["num_layers"])
        self.encoders["LSTMBagOfWordsEncoder"] = LSTMBagOfWordsEncoder(embedding_dim=encoders_hyperparameters["LSTMBagOfWordsEncoder"]["embedding_dim"],
                                                hidden_dim=encoders_hyperparameters["LSTMBagOfWordsEncoder"]["hidden_dim"],
                                                vocab_size=encoders_hyperparameters["LSTMBagOfWordsEncoder"]["vocab_size"],
                                                num_layers=encoders_hyperparameters["LSTMBagOfWordsEncoder"]["num_layers"],
                                                dropout=encoders_hyperparameters["LSTMBagOfWordsEncoder"]["dropout"])

        self.decoder = MLP(in_channels=2*encoders_hyperparameters["HomogeneousGAT"]["heads"] *
                                       encoders_hyperparameters["HomogeneousGAT"]["hidden_size"] +
                                        encoders_hyperparameters["LSTMBagOfWordsEncoder"]["hidden_dim"],
                              output_size=decoder_hyperparameters["output_size"],
                              nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        HomogeneousGAT_output = self.encoders["HomogeneousGAT"].forward(data)
        LSTMBagOfWordsEncoder_output = self.encoders["LSTMBagOfWordsEncoder"].forward(data)
        decoder_input =torch.cat((HomogeneousGAT_output, LSTMBagOfWordsEncoder_output), -1)

        decoder_output = self.decoder.forward(decoder_input)
        return decoder_input, decoder_output
