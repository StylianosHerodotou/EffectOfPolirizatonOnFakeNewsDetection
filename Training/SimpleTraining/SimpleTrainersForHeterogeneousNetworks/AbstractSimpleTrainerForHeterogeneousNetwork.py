from abc import ABC

from Training.SimpleTraining.AbstractSimpleTrainer import AbstractSimpleTrainer
import numpy as np

from Utilities.HeterogeneousGraphsUtils import find_reverse_edge_types
from Utilities.SignedGraphUtils import get_train_eval_indexes, turn_data_to_positive_and_negative_edges
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomLinkSplit


class AbstractSimpleTrainerForHeterogeneousNetwork(AbstractSimpleTrainer, ABC):
    def __init__(self):
        super().__init__()

    def preprocess_data(self, graph):
        return graph

    def create_data_object(self, pre_processed_data):
        return pre_processed_data

    def create_train_eval_data(self, pyg_data, pre_processed_data, training_hyperparameters):
        transform = RandomLinkSplit(
            num_val=0.0,
            num_test=training_hyperparameters["test_size"],
            neg_sampling_ratio=0.0,
            edge_types=pyg_data.edge_types,
            rev_edge_types=[None] * len(pyg_data.edge_types),
            key="edge_type_labels"
        )

        train_data, _, eval_data = transform(pyg_data)
        print(dict(train_data[('entity', 'dislike', 'entity')]).keys())
        print(dict(eval_data[('entity', 'dislike', 'entity')]).keys())

        train_data.edge_index= train_data.edge_type_labels_index
        eval_data.edge_index = eval_data.edge_type_labels_index

        return train_data, eval_data
