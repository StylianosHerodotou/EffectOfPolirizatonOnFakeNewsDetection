import torch
from Models.NNModels.Decoders.HeterogeneousDataSingleTaskDecoder import HeterogeneousDataSingleTaskDecoder


class HeterogeneousDataMultiTaskDecoder(torch.nn.Module):
    def __init__(self, in_channels, pyg_data,classifier_per_task_arguments ):
        super().__init__()

        self.task_decoders = torch.nn.ModuleDict()

        for task_name, MLP_arguments in classifier_per_task_arguments.items():
            self.task_decoders[task_name] = HeterogeneousDataSingleTaskDecoder(in_channels, pyg_data, MLP_arguments)

    def forward(self, data,encoder_output):
        decoder_output = dict()
        for task_name, task_decoder in self.task_decoders.items():
            # print("Now forwarding through ",task_name )
            output = task_decoder.forward(data, encoder_output)
            decoder_output[task_name] = output
        return decoder_output

    def get_optimizers(self):
        optimizers=dict()
        for task_name, task_decoder in self.task_decoders.items():
            optimizers[task_name]=task_decoder.optimizer
        return optimizers


