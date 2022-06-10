from Models.NNModels.Classifiers.MLP import MLP
import torch

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.HomoDataModels.HEATGraphEncoder import HEATGraphEncoder


class HEAT_SAG_ED_MLP_Model(torch.nn.Module):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoder = HEATGraphEncoder(in_channels=encoder_hyperparameters["in_channels"],
                                        pyg_data=encoder_hyperparameters["pyg_data"],
                                        model_parameters=encoder_hyperparameters["model_parameters"])

        last_layer= encoder_hyperparameters["model_parameters"]["pooling_hyper_parameters_for_each_layer"][-1]
        decoder_input_size= last_layer["pooling_hidden_channels"] * last_layer["pooling_num_clusters"]

        self.decoder = MLP(in_channels=decoder_input_size,
                           output_size=decoder_hyperparameters["output_size"],
                           nodes_per_hidden_layer=decoder_hyperparameters["nodes_per_hidden_layer"],
                           dropout=decoder_hyperparameters["dropout"])

    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(encoder_output)
        return encoder_output, decoder_output
