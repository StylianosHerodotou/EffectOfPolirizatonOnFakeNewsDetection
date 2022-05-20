import torch
from abc import ABC
class AbstractEncoderDecoderNNModel(ABC, torch.nn.Module):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(data, encoder_output)
        return encoder_output, decoder_output
