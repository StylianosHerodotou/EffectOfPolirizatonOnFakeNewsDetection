from HyperParameterTuning.AbstractHyperParameterTuner import AbstractHyperParameterTuner
from KFoldTraining.KFoldTrainersForSignedNetworks.SignedGCNKFoldTrainer import SignedGCNKFoldTrainer


class SignedGCNTuner(AbstractHyperParameterTuner):

    def __init__(self, tuning_hyperparameters):
        super().__init__(tuning_hyperparameters)

    def tuning_function(self, config):
        trainer = SignedGCNKFoldTrainer(config["number_of_splits"])
        model_hyperparameters=config["model_hyperparameters"]
        data = config["data"]
        trainer.train(model_hyperparameters, data)



