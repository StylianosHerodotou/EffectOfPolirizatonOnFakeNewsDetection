from HyperParameterTuning.AbstractHyperParameterTuner import AbstractHyperParameterTuner
from Training.KFoldTraining.KFoldTrainersForHeterogeneousNetworks.NormalToHeteroGATMultiTaskKFoldTrainer import NormalToHeteroGATMultiTaskKFoldTrainer


class NormalToHeteroGATMultiTaskTuner(AbstractHyperParameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = NormalToHeteroGATMultiTaskKFoldTrainer(training_hyperparameters["number_of_splits"])





