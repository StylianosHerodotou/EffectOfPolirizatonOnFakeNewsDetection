
from abc import ABC, abstractmethod


class AbstractEmbeddingsGenerator(ABC):
    def __init__(self):
        self.tuner=None
        self.simple_trainer=None

    def generate_embeddings(self,config):
        best_model_config = self.tuner.get_best_trial_configurations(config)
        training_hyperparameters = best_model_config["training_hyperparameters"]
        training_hyperparameters["epochs"]= training_hyperparameters["final_training_epochs"]
        model_hyperparameters = best_model_config["model_hyperparameters"]
        data = best_model_config["data"]

        self.simple_trainer.create_model(model_hyperparameters)
        self.simple_trainer.train(training_hyperparameters, model_hyperparameters, data,
                             in_hyper_parameter_search=False)
        embeddings= self.simple_trainer.generate_embeddings(training_hyperparameters,
                                                                  model_hyperparameters, data)
        return embeddings




