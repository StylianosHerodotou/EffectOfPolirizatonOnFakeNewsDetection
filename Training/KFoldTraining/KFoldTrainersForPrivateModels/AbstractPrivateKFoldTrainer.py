from abc import ABC, abstractmethod
from sklearn.model_selection import KFold

from Training.AbstractTrainer import AbstractTrainer
from abc import ABC
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import os

import numpy as np
from Utilities.InitGlobalVariables import dir_to_base


class AbstractPrivateKFoldTrainer(AbstractTrainer, ABC):
    def __init__(self, number_of_splits, random_state=42, fold_name="example_name"):
        super().__init__()
        self.number_of_splits = number_of_splits
        self.random_state = random_state
        self.fold_score= self.init_performance_metric()
        self.fold_name=fold_name

    def init_performance_metric(self):
        scores = {"accuracy": 0,
                  "precision": 0,
                  "recall": 0,
                  "fbeta_score": 0}
        return scores

    def update_fold_parameters(self, new_scores):
        for key in self.fold_score.keys():
            value= new_scores[key]
            self.fold_score[key] += value

    def fold_score_tostring(self):
        s = self.fold_name + "\n"
        for key, value in self.fold_score.items():
            s += key + ": " + str(value) + ", "
        s += '\n'
        return s


    def get_mean_of_scores(self, training_hyperparameters):
        for key, value in self.fold_score.items():
            self.fold_score[key] = value / training_hyperparameters["number_of_splits"]

    def preprocess_data(self, data):
        return data

    def create_data_object_for_each_split(self, splits, train_set):
        return splits.split(np.arange(len(train_set)))

    def create_train_eval_data_for_fold(self, fold_data, train_set, training_hyperparameters):
        train_idx, val_idx = fold_data

        train_sampler = SubsetRandomSampler(train_idx)
        eval_sampler = SubsetRandomSampler(val_idx)

        train_loader = {"size": len(train_sampler)}
        eval_loader = {"size": len(eval_sampler)}

        train_loader["loader"] = DataLoader(train_set, batch_size=training_hyperparameters["batch_size"],
                                            sampler=train_sampler)
        eval_loader["loader"] = DataLoader(train_set, batch_size=training_hyperparameters["batch_size"],
                                           sampler=eval_sampler)
        return train_loader, eval_loader

    def train(self, training_hyperparameters, model_hyperparameters, data,
              in_hyper_parameter_search=True,
              print_results=True,
              dir_to_save=dir_to_base,
              filename="kfold_results.txt",
              ):

        pre_processed_data = self.preprocess_data(data)
        splits = KFold(n_splits=self.number_of_splits, shuffle=True, random_state=self.random_state)
        data_for_each_split = self.create_data_object_for_each_split(splits, pre_processed_data)
        for fold_number, data_for_fold in enumerate(data_for_each_split):
            train_data, eval_data = self.create_train_eval_data_for_fold(data_for_fold, pre_processed_data,
                                                                         training_hyperparameters)
            model = self.model_function(model_hyperparameters)
            current_fold_scores = model.train_fold(training_hyperparameters, train_data, eval_data, fold_number=fold_number,
                                            in_hyper_parameter_search=in_hyper_parameter_search)

            self.update_fold_parameters(current_fold_scores)

        self.get_mean_of_scores(training_hyperparameters)

        if print_results:
            print("I AM HERE!!!!!!!!!!!!")
            with open(os.path.join(dir_to_save, filename), "a") as file_object:
                file_object.write(self.fold_score_tostring())
        return self.fold_score
