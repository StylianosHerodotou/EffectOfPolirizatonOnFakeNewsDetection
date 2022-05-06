from abc import ABC

from Training.AbstractTrainer import AbstractTrainer
from Training.KFoldTraining.AbstractKFoldTrainer import AbstractKFoldTrainer
from Utilities.SignedGraphUtils import get_train_eval_indexes, turn_data_to_positive_and_negative_edges
import numpy as np
import copy
import torch

class AbstractKFoldTrainerForHeterogeneousNetworks(AbstractKFoldTrainer, ABC):
    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)

    def preprocess_data(self, graph):
        return graph

    def create_data_object_for_each_split(self, splits, graph):
        splits_dict=dict()
        for edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index
            current_edge_type_split= splits.split(np.arange(edge_index.size()[1]))
            splits_dict[edge_type]=current_edge_type_split
        return splits_dict

    def create_train_eval_data_for_fold(self, fold_data, pre_processed_data):
        train_dict = copy.copy(pre_processed_data)
        eval_dict = copy.copy(pre_processed_data)
        for edge_type in pre_processed_data.edge_types:
            train_idx, val_idx = fold_data[edge_type]

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
