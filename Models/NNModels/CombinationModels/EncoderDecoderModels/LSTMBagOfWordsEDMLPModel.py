from Models.NNModels.Classifiers.MLP import MLP
import torch

from Models.NNModels.Encoders.TextEncoders.LSTMBagOfWordsEncoder import LSTMBagOfWordsEncoder


class LSTMBagOfWordsEDMLPModel(torch.nn.Module):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoder = LSTMBagOfWordsEncoder(embedding_dim=encoder_hyperparameters["embedding_dim"],
                                                hidden_dim=encoder_hyperparameters["hidden_dim"],
                                                vocab_size=encoder_hyperparameters["vocab_size"],
                                                num_layers=encoder_hyperparameters["num_layers"],
                                                dropout=encoder_hyperparameters["dropout"])

        self.decoder = MLP(in_channels=encoder_hyperparameters["hidden_dim"],
                              output_size=decoder_hyperparameters["output_size"],
                              nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                              number_of_hidden_layers=decoder_hyperparameters["number_of_hidden_layers"],
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(encoder_output)
        return encoder_output, decoder_output
