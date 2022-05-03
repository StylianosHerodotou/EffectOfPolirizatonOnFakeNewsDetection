from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.HyperParameterTuningForSignedNetworks.SignedGCNTuner import SignedGCNTuner
from Training.SimpleTraining.SimpleTrainersForSignedNetworks.SignedGCNSimpleTrainer import SignedGCNSimpleTrainer

class SignedGCNEmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self,tuning_hyperparameters,training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = SignedGCNTuner(tuning_hyperparameters,training_hyperparameters)
        self.simple_trainer = SignedGCNSimpleTrainer()
