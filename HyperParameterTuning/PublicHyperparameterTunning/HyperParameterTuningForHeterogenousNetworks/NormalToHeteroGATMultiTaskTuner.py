from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForHeterogenousNetworks import \
    AbstractGNNHyperparameterTuner
from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForHeterogeneousNetworks.NormalToHeteroGATMultiTaskKFoldTrainer import NormalToHeteroGATMultiTaskKFoldTrainer


class NormalToHeteroGATMultiTaskTuner(AbstractGNNHyperparameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = NormalToHeteroGATMultiTaskKFoldTrainer(training_hyperparameters["number_of_splits"])





