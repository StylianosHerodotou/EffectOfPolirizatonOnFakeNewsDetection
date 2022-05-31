from abc import ABC
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    AbstractPoolingImplementation.AbstractSAGPoolingMethod import \
    AbstractSAGPoolingMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling.VectorAggregationMethods.AbstractHeteroVectorAggregationMethod import \
    AbstractHeteroVectorAggregationMethod
from Models.NNModels.Encoders.GraphBasedEncoders.GraphEncoders.AbstractPoolLayer.SubgraphPooling. \
    VectorAggregationMethods.AbstractMeanAggregator import \
    AbstractMeanAggregator


class HeteroSAGPoolingMeanAggregator(AbstractSAGPoolingMethod, AbstractHeteroVectorAggregationMethod,
                                     AbstractMeanAggregator, ABC):

    def __init__(self, in_channels, pyg_data, model_parameters):
        super().__init__(in_channels, pyg_data, model_parameters)

    def generate_hyperparameters_for_each_pool_layer(self, in_channels, pyg_data, model_parameters, **kwargs):
        conv_hyperparameters_for_each_layer_for_all_edge_type = self.generate_hyperparameters_for_each_conv_layer(
            in_channels, pyg_data, model_parameters)

        pool_hyperparameters_for_each_layer = list()
        for all_edge_layer_hyperparameeters in conv_hyperparameters_for_each_layer_for_all_edge_type:
            input_size = 0
            for edge_type, edge_hyperparameters in all_edge_layer_hyperparameeters.items():
                input_size += edge_hyperparameters["hidden_channels"] * \
                              edge_hyperparameters["heads"]
                break
            ratio = model_parameters["pooling_ratio"]
            layer_hyperparameters = {
                "in_channels": input_size,
                "ratio": ratio
            }
            pool_hyperparameters_for_each_layer.append(layer_hyperparameters)

        return pool_hyperparameters_for_each_layer

    def pool_forward(self, useful_data, pool_layer):
        homo_data = useful_data.to_homogeneous()
        print(homo_data.x.size())
        homo_data= super(HeteroSAGPoolingMeanAggregator, self).pool_forward(homo_data, pool_layer)
        useful_data = homo_data.to_heterogeneous()
        return useful_data
