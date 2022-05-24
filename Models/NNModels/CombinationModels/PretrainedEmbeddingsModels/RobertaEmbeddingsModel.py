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
        batch_size = data.batch.max() + 1
        roberta_embeddings = data.extra_inputs[roberta_column_name]
        roberta_embeddings = roberta_embeddings.view(batch_size, -1)

        decoder_output = self.decoder.forward(roberta_embeddings)
        return decoder_output
