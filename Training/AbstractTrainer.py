from abc import ABC, abstractmethod


class AbstractTrainer(ABC):
    def __init__(self):
        self.model_function = None

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def set_new_model_parameters(self, model, training_hyperparameters, graph_hyperparameters,
                                 model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
    @abstractmethod
    def train(self, training_hyperparameters, graph_hyperparameters, model_hyperparameters, data,
              in_hyper_parameter_search=True):
        pass