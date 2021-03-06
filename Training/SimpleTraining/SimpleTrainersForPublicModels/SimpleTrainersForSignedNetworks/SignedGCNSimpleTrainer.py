from Models.CompleteModels.PublicModels.PublicSignedModels.SignedGCNCompleteModel import SignedGCNCompleteModel
import torch

from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForSignedNetworks.AbstractSimpleTrainerForSignedNetwork import \
    AbstractSimpleTrainerForSignedNetwork


class SignedGCNSimpleTrainer(AbstractSimpleTrainerForSignedNetwork):

    def __init__(self):
        super().__init__()
        self.model_function = SignedGCNCompleteModel
        self.temp_model = None


    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
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

        eval_data["SIGNED_features"]=train_data["SIGNED_features"]



