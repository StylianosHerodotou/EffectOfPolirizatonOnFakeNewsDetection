from Models.NNModels.Decoders.HeterogeneousDataMultiTaskDecoder import HeterogeneousDataMultiTaskDecoder
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.NormalToHeteroGATEncoder import NormalToHeteroGATEncoder
import torch


class NormalToHeteroEDMultiTaskNNModel(torch.nn.Module):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoder = NormalToHeteroGATEncoder(in_channels=encoder_hyperparameters["in_channels"],
                                                pyg_data=encoder_hyperparameters["pyg_data"],
                                                model_parameters= encoder_hyperparameters["model_parameters"])
        self.decoder = HeterogeneousDataMultiTaskDecoder(in_channels=encoder_hyperparameters["hidden_channels"] *
                                                                     encoder_hyperparameters["heads"],
                                                         pyg_data=decoder_hyperparameters["pyg_data"],
                                                         classifier_per_task_arguments=decoder_hyperparameters[
                                            "classifier_per_task_arguments"])

    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(data, encoder_output)
        return encoder_output, decoder_output
