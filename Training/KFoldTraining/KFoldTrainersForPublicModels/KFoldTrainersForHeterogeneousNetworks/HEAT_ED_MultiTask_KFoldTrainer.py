from .AbstractKFoldTrainerForHeterogeneousNetworks import AbstractPublicKFoldTrainerForHeterogeneousNetworks
from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.HEAT_ED_MultiTask_CompleteModel import \
    HEAT_ED_MultiTask_CompleteModel
from Utilities.SignedGraphUtils import get_train_eval_indexes
import numpy as np
import copy


class HEAT_ED_MultiTask_KFoldTrainer(AbstractPublicKFoldTrainerForHeterogeneousNetworks):

    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)
        self.model_function = HEAT_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass

    def create_data_object_for_each_split(self, splits, graph):
        to_return = splits.split(np.arange(graph.edge_index.size(1)))
        return to_return

    def create_train_eval_data_for_fold(self, fold_data, pre_processed_data):
        train_dict = copy.copy(pre_processed_data)
        eval_dict = copy.copy(pre_processed_data)

        train_idx, val_idx = fold_data

        edge_index = pre_processed_data.edge_index
        train_index, eval_index = get_train_eval_indexes(edge_index, train_idx, val_idx)

        train_dict.edge_index = train_index
        eval_dict.edge_index = eval_index

        list_of_attributes_to_split = ["edge_type", "edge_attr"]

        for edge_feature_name in list_of_attributes_to_split:
            edge_feature_values = pre_processed_data[edge_feature_name]
            if edge_feature_name == "edge_index":
                continue

            train_list = edge_feature_values[train_idx]
            eval_list = edge_feature_values[val_idx]

            train_dict[edge_feature_name] = train_list
            eval_dict[edge_feature_name] = eval_list

        return train_dict, eval_dict

