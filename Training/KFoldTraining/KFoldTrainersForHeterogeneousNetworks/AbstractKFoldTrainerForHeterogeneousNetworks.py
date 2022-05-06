from abc import ABC

from Training.AbstractTrainer import AbstractTrainer
from Training.KFoldTraining.AbstractKFoldTrainer import AbstractKFoldTrainer
from Utilities.SignedGraphUtils import get_train_eval_indexes, turn_data_to_positive_and_negative_edges
import numpy as np
import copy
import torch
from sklearn.model_selection import KFold

class AbstractKFoldTrainerForHeterogeneousNetworks(AbstractKFoldTrainer, ABC):
    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)

    def preprocess_data(self, graph):
        return graph

    def get_splits_for_each_edge_type(self,splits, graph):
        splits_dict = dict()
        for edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index
            current_edge_type_split = splits.split(np.arange(edge_index.size()[1]))
            splits_dict[edge_type] = current_edge_type_split
        return splits_dict

    def get_dicts_for_each_split_iteration(self,splits, graph,splits_dict):
        splits_in_iterable_form = list()
        for split_index in range(splits.n_splits):
            current_index_dict = dict()
            for edge_type in graph.edge_types:
                current_generator = splits_dict[edge_type]
                current_split_index_indexes = iter(current_generator)
                current_index_dict[edge_type] = current_split_index_indexes
            splits_in_iterable_form.append(current_index_dict)
        return splits_in_iterable_form

    def create_data_object_for_each_split(self, splits, graph):
        splits_dict= self.get_splits_for_each_edge_type(splits,graph)
        splits_in_iterable_form = self.get_dicts_for_each_split_iteration(splits, graph,splits_dict)
        return splits_in_iterable_form

    def create_train_eval_data_for_fold(self, fold_data, pre_processed_data):
        train_dict = copy.copy(pre_processed_data)
        eval_dict = copy.copy(pre_processed_data)
        for edge_type in pre_processed_data.edge_types:
            train_idx, val_idx = fold_data[edge_type]
            print(train_idx)

            edge_index = pre_processed_data[edge_type].edge_index
            train_index, eval_index = get_train_eval_indexes(edge_index, train_idx, val_idx)

            train_dict[edge_type].edge_index = train_index
            eval_dict[edge_type].edge_index = eval_index

            for edge_feature_name, edge_feature_values in pre_processed_data[edge_type].items():
                if edge_feature_name == "edge_index":
                    continue

                train_list = list()
                eval_list = list()
                for index in train_idx:
                    train_list.append(edge_feature_values[index])

                for index in val_idx:
                    eval_list.append(edge_feature_values[index])

                train_list = torch.stack(train_list)
                eval_list = torch.stack(eval_list)

                train_dict[edge_type][edge_feature_name] = train_list
                eval_dict[edge_type][edge_feature_name] = eval_list

        return train_dict, eval_dict

    def train(self, training_hyperparameters, model_hyperparameters, data,
              in_hyper_parameter_search=True):
        pre_processed_data = self.preprocess_data(data)
        splits = KFold(n_splits=self.number_of_splits, shuffle=True, random_state=self.random_state)
        data_for_each_split = self.create_data_object_for_each_split(splits, pre_processed_data)
        for fold_number, data_for_fold in enumerate(data_for_each_split):
            train_data, eval_data = self.create_train_eval_data_for_fold(data_for_fold, pre_processed_data)
            model = self.model_function(model_hyperparameters)
            self.set_new_model_parameters(model, training_hyperparameters, model_hyperparameters,
                                          data, pre_processed_data, train_data, eval_data)
            model.train_fold(training_hyperparameters, train_data, eval_data, fold_number=fold_number,
                             in_hyper_parameter_search=in_hyper_parameter_search)
