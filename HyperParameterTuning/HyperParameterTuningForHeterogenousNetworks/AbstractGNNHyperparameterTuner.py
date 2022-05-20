from HyperParameterTuning.AbstractHyperParameterTuner import AbstractHyperParameterTuner
from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForHeterogeneousNetworks.NormalToHeteroGATMultiTaskKFoldTrainer import NormalToHeteroGATMultiTaskKFoldTrainer


class AbstractGNNHyperparameterTuner(AbstractHyperParameterTuner):

    def adjust_training_hyperparameters(self,config):
        model_hyperparameters=config["model_hyperparameters"]["encoder_hyperparameters"]["model_parameters"]
        number_of_all_layers= len(model_hyperparameters["hyper_parameters_for_each_layer"])
        start_from=number_of_all_layers-model_hyperparameters["number_of_layers"]
        model_hyperparameters["hyper_parameters_for_each_layer"]=model_hyperparameters["hyper_parameters_for_each_layer"][start_from:]






