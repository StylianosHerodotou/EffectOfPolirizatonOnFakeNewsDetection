from HyperParameterTuning.AbstractHyperParameterTuner import AbstractHyperParameterTuner
from Training.KFoldTraining.KFoldTrainersForPrivateModels.KFoldTrainersForGNNBasedModels.KFoldTrainerForHomogeneousGATModel import KFoldTrainerForHomogeneousGATModel


class HomogeneousGATTuner(AbstractHyperParameterTuner):
    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = KFoldTrainerForHomogeneousGATModel(training_hyperparameters["number_of_splits"])





