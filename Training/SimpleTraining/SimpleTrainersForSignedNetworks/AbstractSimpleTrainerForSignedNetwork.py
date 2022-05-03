
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

    def create_train_eval_data(self, data_object, pre_processed_data,training_hyperparameters):
        positive_index, negative_index = pre_processed_data
        if(training_hyperparameters["test_size"]==0):
            train_pos=positive_index
            eval_pos=positive_index
            train_neg=negative_index
            eval_neg= positive_index
        else:
            pos_train_idx, pos_val_idx = train_test_split(np.arange(positive_index.size()[1]),
                                                          test_size = training_hyperparameters["test_size"],
                                                          random_state=42, shuffle=True)
            neg_train_idx, neg_val_idx = train_test_split(np.arange(negative_index.size()[1]),
                                                          test_size = training_hyperparameters["test_size"],
                                                          random_state=42, shuffle=True)

            train_pos, eval_pos = get_train_eval_indexes(positive_index, pos_train_idx, pos_val_idx)
            train_neg, eval_neg = get_train_eval_indexes(negative_index, neg_train_idx, neg_val_idx)

        train_data = {
            "pos_index": train_pos,
            "neg_index": train_neg
        }
        eval_data = {
            "pos_index": eval_pos,
            "neg_index": eval_neg
        }
        return train_data, eval_data
