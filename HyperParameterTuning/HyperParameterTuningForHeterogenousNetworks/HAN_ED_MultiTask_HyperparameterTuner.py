from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForHeterogeneousNetworks.HAN_ED_MultiTask_KFoldTrainer import \
    HAN_ED_MultiTask_KFoldTrainer
from HyperParameterTuning.HyperParameterTuningForHeterogenousNetworks.AbstractGNNHyperparameterTuner import \
    AbstractGNNHyperparameterTuner


class HAN_ED_MultiTask_HyperparameterTuner(AbstractGNNHyperparameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = HAN_ED_MultiTask_KFoldTrainer(training_hyperparameters["number_of_splits"])
