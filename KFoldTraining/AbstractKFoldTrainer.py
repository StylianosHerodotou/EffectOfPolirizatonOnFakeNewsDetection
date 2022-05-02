from abc import ABC, abstractmethod
from sklearn.model_selection import KFold


class AbstractKFoldTrainer(ABC):
    def __init__(self, number_of_splits, random_state=42):
        self.number_of_splits = number_of_splits
        self.random_state = random_state
        self.model_function = None

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def create_data_for_each_split(self, splits, data):
        pass

    @abstractmethod
    def create_train_eval_data_for_fold(self, fold_data, pre_processed_data):
        pass

    @abstractmethod
    def set_new_model_parameters(self, model, model_hyperparameters, data, pre_processed_data, train_data, eval_data):
        pass

    def train(self, model_hyperparameters, data, in_hyper_parameter_search=True):
        pre_processed_data = self.preprocess_data(data)
        splits = KFold(n_splits=self.number_of_splits, shuffle=True, random_state=self.random_state)

        data_for_each_split = self.create_data_for_each_split(splits, pre_processed_data)
        for fold_number, data_for_fold in enumerate(data_for_each_split):
            train_data, eval_data = self.create_train_eval_data_for_fold(data_for_fold, pre_processed_data)
            model = self.model_function(model_hyperparameters)
            self.set_new_model_parameters(model, model_hyperparameters, data, pre_processed_data, train_data, eval_data)
            model.train_fold(train_data, eval_data, fold_number=fold_number, epochs=data["epochs"],
                             in_hyper_parameter_search=in_hyper_parameter_search)
