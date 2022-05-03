
from abc import ABC

from Training.SimpleTraining.AbstractSimpleTrainer import AbstractSimpleTrainer
import numpy as np
from Utilities.SignedGraphUtils import get_train_eval_indexes, turn_data_to_positive_and_negative_edges
from sklearn.model_selection import train_test_split

class AbstractSimpleTrainerForSignedNetwork(AbstractSimpleTrainer, ABC):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        positive_index, negative_index = turn_data_to_positive_and_negative_edges(graph.edge_index, graph.edge_attr)
        return positive_index, negative_index

    def create_data_object(self, pre_processed_data):
        return pre_processed_data

    def create_train_eval_data(self, data_object, pre_processed_data):


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
