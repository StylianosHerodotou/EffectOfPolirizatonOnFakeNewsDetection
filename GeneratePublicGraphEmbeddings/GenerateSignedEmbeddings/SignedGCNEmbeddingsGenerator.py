from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForSignedNetworks.SignedGCNTuner import SignedGCNTuner
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForSignedNetworks import SignedGCNSimpleTrainer

class SignedGCNEmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self,tuning_hyperparameters,training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = SignedGCNTuner(tuning_hyperparameters,training_hyperparameters)
        self.simple_trainer = SignedGCNSimpleTrainer()
