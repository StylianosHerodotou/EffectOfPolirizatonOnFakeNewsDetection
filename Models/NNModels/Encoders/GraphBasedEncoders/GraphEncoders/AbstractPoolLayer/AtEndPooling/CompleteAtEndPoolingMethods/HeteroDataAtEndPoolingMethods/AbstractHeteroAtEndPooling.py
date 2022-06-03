from abc import ABC, abstractmethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.AtEndPooling.AbstractAtEndPooling import \
    AbstractAtEndPooling
import torch

class AbstractHeteroAtEndPooling(AbstractAtEndPooling, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def get_vector_representation(self, useful_data):
        vector_representation_dict = dict()
        x_dict = useful_data.x_dict
        for key in x_dict.keys():
            current_useful_data = useful_data[key]
            current_vector_representation = self.get_single_vector_representation(
                current_useful_data)
            vector_representation_dict[key] = current_vector_representation

        all_vector_representations = list()
        for value in vector_representation_dict.values():
            all_vector_representations.append(value)

        # vector_representation = torch.stack(all_vector_representations, dim=-1)
        vector_representation = torch.cat(all_vector_representations)
        return vector_representation

    def get_pooling_additional_loss(self, useful_data):
        pooling_additional_loss_dict = dict()
        x_dict = useful_data.x_dict
        for key in x_dict.keys():
            current_useful_data = useful_data[key]
            current_pooling_additional_loss = self.get_single_pooling_additional_loss(
                current_useful_data)
            pooling_additional_loss_dict[key] = current_pooling_additional_loss
        return pooling_additional_loss_dict

    def extra_pooling_dropout_forward(self, useful_data, index):
        x_dict = useful_data.x_dict
        for key in x_dict.keys():
            current_useful_data = useful_data[key]
            current_useful_data = self.single_extra_pooling_dropout_forward(
                current_useful_data)
            useful_data[key] = current_useful_data
        return useful_data


