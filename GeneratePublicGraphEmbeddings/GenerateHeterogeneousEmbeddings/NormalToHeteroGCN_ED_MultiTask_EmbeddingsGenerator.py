from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.HyperParameterTuningForHeterogenousNetworks.NormalToHeteroGCN_ED_MultiTask_HyperparameterTuner import \
    NormalToHeteroGCN_ED_MultiTask_HyperparameterTuner
from Training.SimpleTraining.SimpleTrainersForHeterogeneousNetworks.NormalToHeteroGCN_ED_MultiTask_SimpleTrainer import \
    NormalToHeteroGCN_ED_MultiTask_SimpleTrainer

class NormalToHeteroGCN_ED_MultiTask_EmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self, tuning_hyperparameters, training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = NormalToHeteroGCN_ED_MultiTask_HyperparameterTuner(tuning_hyperparameters, training_hyperparameters)
        self.simple_trainer = NormalToHeteroGCN_ED_MultiTask_SimpleTrainer()
