from GeneratePublicGraphEmbeddings.AbstractEmbeddingsGenerator import AbstractEmbeddingsGenerator
from HyperParameterTuning.PublicHyperparameterTunning.HyperParameterTuningForHeterogenousNetworks.\
    HAN_ED_MultiTask_HyperparameterTuner import HAN_ED_MultiTask_HyperparameterTuner
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForHeterogeneousNetworks.\
    HAN_ED_MultiTask_SimpleTrainer import HAN_ED_MultiTask_SimpleTrainer

class HAN_ED_MultiTask_EmbeddingsGenerator(AbstractEmbeddingsGenerator):
    def __init__(self, tuning_hyperparameters, training_hyperparameters, model_hyperparameters):
        super().__init__()
        self.tuner = HAN_ED_MultiTask_HyperparameterTuner(tuning_hyperparameters, training_hyperparameters)
        self.simple_trainer = HAN_ED_MultiTask_SimpleTrainer()

