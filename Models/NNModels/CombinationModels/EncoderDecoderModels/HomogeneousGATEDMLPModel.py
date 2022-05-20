from Models.NNModels.Classifiers.MLP import MLP
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomogeneousGAT import HomogeneousGAT
import torch


class HomogeneousGATEDMLPModel(torch.nn.Module):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoder = HomogeneousGAT(in_channels=encoder_hyperparameters["in_channels"],
                                      edge_dim=encoder_hyperparameters["edge_dim"],
                                      hidden_size=encoder_hyperparameters["hidden_size"],
                                      heads=encoder_hyperparameters["heads"],
                                      dropout=encoder_hyperparameters["dropout"],
                                      pooling_ratio=encoder_hyperparameters["pooling_ratio"],
                                      num_layers=encoder_hyperparameters["num_layers"])

        self.decoder = MLP(in_channels=2*encoder_hyperparameters["heads"] * encoder_hyperparameters["hidden_size"],
                              output_size=decoder_hyperparameters["output_size"],
                              nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                              number_of_hidden_layers=decoder_hyperparameters["number_of_hidden_layers"],
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(encoder_output)
        return encoder_output, decoder_output
