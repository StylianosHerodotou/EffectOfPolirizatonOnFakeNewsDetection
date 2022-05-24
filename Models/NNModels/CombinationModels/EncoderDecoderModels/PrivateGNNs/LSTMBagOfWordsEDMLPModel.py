from Models.NNModels.Classifiers.MLP import MLP

from Models.NNModels.CombinationModels.EncoderDecoderModels.AbstractEncoderDecoderNNModel import \
    AbstractEncoderDecoderNNModel
from Models.NNModels.Encoders.TextEncoders.LSTMBagOfWordsEncoder import LSTMBagOfWordsEncoder


class LSTMBagOfWordsEDMLPModel(AbstractEncoderDecoderNNModel):
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
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(encoder_output)
        return encoder_output, decoder_output
