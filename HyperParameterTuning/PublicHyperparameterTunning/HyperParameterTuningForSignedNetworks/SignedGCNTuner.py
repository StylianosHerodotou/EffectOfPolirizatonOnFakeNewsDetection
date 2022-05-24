from HyperParameterTuning.AbstractHyperParameterTuner import AbstractHyperParameterTuner
from Training.KFoldTraining.KFoldTrainersForPublicModels.KFoldTrainersForSignedNetworks import SignedGCNKFoldTrainer

class SignedGCNTuner(AbstractHyperParameterTuner):

    def __init__(self, tuning_hyperparameters,training_hyperparameters):
        super().__init__(tuning_hyperparameters)
        self.trainer = SignedGCNKFoldTrainer(training_hyperparameters["number_of_splits"])





