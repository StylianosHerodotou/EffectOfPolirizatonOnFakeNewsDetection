from abc import ABC, abstractmethod

from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.AbstractSubgraphPooling import \
    AbstractSubgraphPooling


class AbstractHomoVectorAggregationMethod(AbstractSubgraphPooling, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def update_vector_representation(self, useful_data, vector_representation):
        return self.update_single_vector_representation(useful_data, vector_representation, is_homogeneous_data=True)

    def final_update_vector_representation(self, vector_representation):
        return self.final_single_update_vector_representation(vector_representation)
