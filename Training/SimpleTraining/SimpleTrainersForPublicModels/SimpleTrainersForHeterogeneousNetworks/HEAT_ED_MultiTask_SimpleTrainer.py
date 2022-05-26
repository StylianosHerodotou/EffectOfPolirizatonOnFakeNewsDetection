from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.\
    HEAT_ED_MultiTask_CompleteModel import HEAT_ED_MultiTask_CompleteModel
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForHeterogeneousNetworks.AbstractSimpleTrainerForHeterogeneousNetwork import \
    AbstractSimpleTrainerForHeterogeneousNetwork
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from Utilities.SignedGraphUtils import get_train_eval_indexes


class HEAT_ED_MultiTask_SimpleTrainer(AbstractSimpleTrainerForHeterogeneousNetwork):

    def __init__(self):
        super().__init__()
        self.model_function = HEAT_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass

    def create_train_eval_data(self, pyg_data, pre_processed_data, training_hyperparameters):
        train_dict = copy.copy(pyg_data)
        eval_dict = copy.copy(pyg_data)
        if training_hyperparameters["test_size"] != 0:

            edge_index = pyg_data.edge_index

            train_idx, val_idx = train_test_split(np.arange(edge_index.size(1)),
                                                  test_size=training_hyperparameters["test_size"],
                                                  random_state=42, shuffle=True)
            train_index, eval_index = get_train_eval_indexes(edge_index, train_idx, val_idx)

            train_dict.edge_index = train_index
            eval_dict.edge_index = eval_index

            # edge type
            # edge attribute
            list_of_attributes_to_split = ["edge_type", "edge_attr"]

            for edge_feature_name in list_of_attributes_to_split:
                edge_feature_values = pyg_data[edge_feature_name]
                if edge_feature_name == "edge_index":
                    continue

                train_list = edge_feature_values[train_idx]
                eval_list = edge_feature_values[val_idx]

                train_dict[edge_feature_name] = train_list
                eval_dict[edge_feature_name] = eval_list

        return train_dict, eval_dict
