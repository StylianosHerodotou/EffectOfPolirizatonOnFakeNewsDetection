from abc import ABC, abstractmethod

from Training.AbstractTrainer import AbstractTrainer


class AbstractPublicTrainer(AbstractTrainer,ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def set_new_model_parameters(self, model, training_hyperparameters,
                                 model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass