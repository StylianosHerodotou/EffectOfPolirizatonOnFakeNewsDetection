from DatasetRepresentation.DataPreprocessing.BERTProcessing import roberta_column_name
from Models.NNModels.Classifiers.MLP import MLP
from Models.NNModels.Encoders.GraphEncoders.HomogeneousGAT import HomogeneousGAT
import torch

from Models.NNModels.Encoders.TextEncoders.LSTMBagOfWordsEncoder import LSTMBagOfWordsEncoder

class RobertaEmbeddings_AND_HomogeneousGATModel(torch.nn.Module):
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



        self.decoder = MLP(in_channels=2*encoders_hyperparameters["HomogeneousGAT"]["heads"] *
                                       encoders_hyperparameters["HomogeneousGAT"]["hidden_size"] +
                                       encoders_hyperparameters["RobertaEmbeddings"]["sizeOfEmbeddings"]
                                        ,
                              output_size=decoder_hyperparameters["output_size"],
                              nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                              number_of_hidden_layers=decoder_hyperparameters["number_of_hidden_layers"],
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        HomogeneousGAT_output = self.encoders["HomogeneousGAT"].forward(data)

        batch_size = data.batch.max() + 1
        RobertaEmbeddings = data.extra_inputs[roberta_column_name]
        RobertaEmbeddings = RobertaEmbeddings.view(batch_size, -1)

        decoder_input =torch.cat((HomogeneousGAT_output, RobertaEmbeddings), -1)

        decoder_output = self.decoder.forward(decoder_input)
        return decoder_input, decoder_output
