from DatasetRepresentation.DataPreprocessing.BERTProcessing import roberta_column_name
from Models.NNModels.Classifiers.MLP import MLP
import torch


class RobertaEmbeddingsModel(torch.nn.Module):
    def __init__(self, input_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.decoder = MLP(in_channels=input_hyperparameters["RobertaEmbeddings"]["sizeOfEmbeddings"],
                              output_size=decoder_hyperparameters["output_size"],
                              nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                              dropout=decoder_hyperparameters["dropout"])
    def forward(self, data):
        HomogeneousGAT_output = self.encoders["HomogeneousGAT"].forward(data)

        batch_size = data.batch.max() + 1
        RobertaEmbeddings = data.extra_inputs[roberta_column_name]
        RobertaEmbeddings = RobertaEmbeddings.view(batch_size, -1)

        decoder_input =torch.cat((HomogeneousGAT_output, RobertaEmbeddings), -1)

        decoder_output = self.decoder.forward(decoder_input)
        return decoder_input, decoder_output
