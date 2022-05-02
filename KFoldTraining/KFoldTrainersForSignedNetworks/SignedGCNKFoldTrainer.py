from KFoldTraining.KFoldTrainersForSignedNetworks.AbstractKFoldTrainerForSignedNetwork import \
    AbstractKFoldTrainerForSignedNetwork
from LargeGraph.LargeGraphTraining.LargeModels.SignedGCN import SignedGCNModel
import torch


class SignedGCNKFoldTrainer(AbstractKFoldTrainerForSignedNetwork):

    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)
        self.model_function = SignedGCNModel
        self.temp_model = None

    def set_new_model_parameters(self, model, model_hyperparameters, data, pre_processed_data, train_data, eval_data):
        # only spectral information
        train_pos = train_data["pos_index"]
        train_neg = train_data["neg_index"]
        x_features = data.x

        if (model_hyperparameters["spectral_features_type"] == 0):
            train_data["SIGNED_features"] = model.model.create_spectral_features(train_pos, train_neg)
        # only node features
        elif (model_hyperparameters["spectral_features_type"] == 1):
            train_data["SIGNED_features"] = x_features
        # using both spectral and node features
        else:
            spectral_features = self.temp_model.model.create_spectral_features(train_pos, train_neg)
            mixed_features = torch.cat((x_features, spectral_features), dim=1)
            print(spectral_features.shape)
            train_data["SIGNED_features"] = mixed_features
            print("size of input", x_features.size(), "size of model input",
                  model_hyperparameters["size_of_x_features"])



