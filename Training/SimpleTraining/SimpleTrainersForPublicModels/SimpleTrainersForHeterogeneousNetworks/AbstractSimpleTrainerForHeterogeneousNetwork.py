from abc import ABC

from Training.SimpleTraining.SimpleTrainersForPublicModels.AbstractSimplePublicTrainer import AbstractSimplePublicTrainer
import numpy as np
from Utilities.SignedGraphUtils import get_train_eval_indexes
from sklearn.model_selection import train_test_split
import copy


class AbstractSimpleTrainerForHeterogeneousNetwork(AbstractSimplePublicTrainer, ABC):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        return graph

    def create_data_object(self, pre_processed_data):
        return pre_processed_data

    def create_train_eval_data(self, pyg_data, pre_processed_data, training_hyperparameters):
        train_dict = copy.copy(pyg_data)
        eval_dict = copy.copy(pyg_data)
        if training_hyperparameters["test_size"] != 0:
            for edge_type in pyg_data.edge_types:
                edge_index = pyg_data[edge_type].edge_index

                train_idx, val_idx = train_test_split(np.arange(edge_index.size()[1]),
                                                      test_size=training_hyperparameters["test_size"],
                                                      random_state=42, shuffle=True)
                train_index, eval_index = get_train_eval_indexes(edge_index, train_idx, val_idx)

                train_dict[edge_type].edge_index = train_index
                eval_dict[edge_type].edge_index = eval_index

                for edge_feature_name, edge_feature_values in pyg_data[edge_type].items():
                    if edge_feature_name == "edge_index":
                        continue

                    train_list = edge_feature_values[train_idx]
                    eval_list = edge_feature_values[val_idx]

                    train_dict[edge_type][edge_feature_name] = train_list
                    eval_dict[edge_type][edge_feature_name] = eval_list

        return train_dict, eval_dict
