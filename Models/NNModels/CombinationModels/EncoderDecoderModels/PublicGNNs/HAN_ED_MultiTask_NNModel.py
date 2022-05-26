from Models.NNModels.CombinationModels.EncoderDecoderModels.AbstractEncoderDecoderNNModel import \
    AbstractEncoderDecoderNNModel
from Models.NNModels.Decoders.HeterogeneousDataMultiTaskDecoder import HeterogeneousDataMultiTaskDecoder
from Models.NNModels.Encoders.GraphBasedEncoders.NodeEncoders.HANEncoder import HANEncoder


class HAN_ED_MultiTask_NNModel(AbstractEncoderDecoderNNModel):
    def __init__(self, encoder_hyperparameters, decoder_hyperparameters):
        super().__init__(encoder_hyperparameters, decoder_hyperparameters)
        self.encoder = HANEncoder(in_channels=encoder_hyperparameters["in_channels"],
                                                pyg_data=encoder_hyperparameters["pyg_data"],
                                                model_parameters=encoder_hyperparameters["model_parameters"])

        last_layer= encoder_hyperparameters["model_parameters"]["hyper_parameters_for_each_layer"][-1]
        decoder_in_channels = last_layer["hidden_channels"]
        self.decoder = HeterogeneousDataMultiTaskDecoder(in_channels=decoder_in_channels,
                                                         pyg_data=decoder_hyperparameters["pyg_data"],
                                                         classifier_per_task_arguments=decoder_hyperparameters[
                                            "classifier_per_task_arguments"])

    def forward(self, data):
        encoder_output = self.encoder.forward(data)
        decoder_output = self.decoder.forward(data, encoder_output)
        return encoder_output, decoder_output
