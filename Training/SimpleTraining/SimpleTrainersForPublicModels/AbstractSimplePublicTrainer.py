from abc import ABC, abstractmethod

from Training.AbstractPublicTrainer import AbstractPublicTrainer


class AbstractSimplePublicTrainer(AbstractPublicTrainer, ABC):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_model(self, model_hyperparameters):
        self.model = self.model_function(model_hyperparameters)

    @abstractmethod
    def create_data_object(self, pre_processed_data):
        pass

    @abstractmethod
    def create_train_eval_data(self, data_object, pre_processed_data, training_hyperparameters):
        pass

    def train(self, training_hyperparameters, model_hyperparameters, data,
              in_hyper_parameter_search=False):
        pre_processed_data = self.preprocess_data(data)
        data_object = self.create_data_object(pre_processed_data)
        train_data, eval_data = self.create_train_eval_data(data_object, pre_processed_data,training_hyperparameters)
        self.set_new_model_parameters(self.model, training_hyperparameters,
                                      model_hyperparameters,
                                      data, pre_processed_data, train_data, eval_data)
        self.model.train_fold(training_hyperparameters, train_data, eval_data, fold_number=0,
                              in_hyper_parameter_search=in_hyper_parameter_search)

    def generate_embeddings(self, training_hyperparameters, model_hyperparameters, data):
        pre_processed_data = self.preprocess_data(data)
        data_object = self.create_data_object(pre_processed_data)
        temp= training_hyperparameters["test_size"]
        training_hyperparameters["test_size"]=0
        train_data, eval_data = self.create_train_eval_data(data_object, pre_processed_data,training_hyperparameters)
        self.set_new_model_parameters(self.model, training_hyperparameters,
                                      model_hyperparameters,
                                      data, pre_processed_data, train_data, eval_data)
        training_hyperparameters["test_size"]=temp
        return self.model.generate_embeddings(train_data)