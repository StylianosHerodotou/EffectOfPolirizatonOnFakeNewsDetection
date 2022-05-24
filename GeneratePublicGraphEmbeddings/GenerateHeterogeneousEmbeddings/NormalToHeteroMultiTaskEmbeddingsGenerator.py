from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForHeterogenousNetworks import \
    NormalToHeteroGATMultiTaskTuner
from Training.SimpleTraining.SimpleTrainersForHeterogeneousNetworks.NormalToHeteroMultiTaskTrainer import \
    NormalToHeteroMultiTaskTrainer


class NormalToHeteroMultiTaskEmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self, tuning_hyperparameters, training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = NormalToHeteroGATMultiTaskTuner(tuning_hyperparameters, training_hyperparameters)
        self.simple_trainer = NormalToHeteroMultiTaskTrainer()
