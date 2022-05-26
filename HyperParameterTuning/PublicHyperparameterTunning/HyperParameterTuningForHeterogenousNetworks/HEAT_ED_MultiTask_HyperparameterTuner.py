from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForHeterogeneousNetworks.\
    HEAT_ED_MultiTask_KFoldTrainer import HEAT_ED_MultiTask_KFoldTrainer
from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForHeterogenousNetworks.AbstractGNNHyperparameterTuner import \
    AbstractGNNHyperparameterTuner


class HEAT_ED_MultiTask_HyperparameterTuner(AbstractGNNHyperparameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = HEAT_ED_MultiTask_KFoldTrainer(training_hyperparameters["number_of_splits"])
