from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    def __init__(self):
        self.model_function = None

    @abstractmethod
    def train(self, training_hyperparameters, model_hyperparameters, data,
              in_hyper_parameter_search=True):
        pass