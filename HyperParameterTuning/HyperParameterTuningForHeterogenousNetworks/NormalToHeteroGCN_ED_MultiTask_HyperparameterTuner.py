from HyperParameterTuning.HyperParameterTuningForHeterogenousNetworks.AbstractGNNHyperparameterTuner import \
    AbstractGNNHyperparameterTuner
from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForHeterogeneousNetworks.NormalToHeteroGCN_ED_MultiTask_KFoldTrainer import NormalToHeteroGCN_ED_MultiTask_KFoldTrainer


class NormalToHeteroGCN_ED_MultiTask_HyperparameterTuner(AbstractGNNHyperparameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = NormalToHeteroGCN_ED_MultiTask_KFoldTrainer(training_hyperparameters["number_of_splits"])
