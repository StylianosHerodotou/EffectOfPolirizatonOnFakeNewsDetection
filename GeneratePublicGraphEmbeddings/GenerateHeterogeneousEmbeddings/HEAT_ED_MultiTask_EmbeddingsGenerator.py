from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForHeterogenousNetworks.HEAT_ED_MultiTask_HyperparameterTuner import HEAT_ED_MultiTask_HyperparameterTuner
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForHeterogeneousNetworks.HEAT_ED_MultiTask_SimpleTrainer import HEAT_ED_MultiTask_SimpleTrainer

class HEAT_ED_MultiTask_EmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self, tuning_hyperparameters, training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = HEAT_ED_MultiTask_HyperparameterTuner(tuning_hyperparameters, training_hyperparameters)
        self.simple_trainer = HEAT_ED_MultiTask_SimpleTrainer()

