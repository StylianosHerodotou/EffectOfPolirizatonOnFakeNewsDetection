
from abc import ABC

from Training.KFoldTraining import AbstractKFoldTrainer
import numpy as np
from Utilities.SignedGraphUtils import get_train_eval_indexes, turn_data_to_positive_and_negative_edges


class AbstractKFoldTrainerForSignedNetwork(AbstractKFoldTrainer):
    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)

    def preprocess_data(self, graph):
        positive_index, negative_index = turn_data_to_positive_and_negative_edges(graph.edge_index, graph.edge_attr)
        return positive_index, negative_index

    def create_data_for_each_split(self, splits, pre_processed_data):
        positive_index, negative_index = pre_processed_data
        positive_splits = splits.split(np.arange(positive_index.size()[1]))
        negative_splits = splits.split(np.arange(negative_index.size()[1]))
        return zip(positive_splits, negative_splits)

    def create_train_eval_data_for_fold(self, fold_data, pre_processed_data):
        all_positive_index, all_negative_index = pre_processed_data
        fold_positive_index, fold_negative_index = fold_data

        pos_train_idx, pos_val_idx = fold_positive_index
        neg_train_idx, neg_val_idx = fold_negative_index

        train_pos, eval_pos = get_train_eval_indexes(all_positive_index, pos_train_idx, pos_val_idx)
        train_neg, eval_neg = get_train_eval_indexes(all_negative_index, neg_train_idx, neg_val_idx)

        train_data = {
            "pos_index": train_pos,
            "neg_index": train_neg
        }
        eval_data = {
            "pos_index": eval_pos,
            "neg_index": eval_neg
        }
        return train_data, eval_data
