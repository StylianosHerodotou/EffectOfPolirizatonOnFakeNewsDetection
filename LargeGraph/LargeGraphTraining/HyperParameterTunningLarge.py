from LargeGraph.LargeGraphTraining.LargeModels.LargeGraphModel import get_train_eval_indexes
from LargeGraph.Utils import turn_data_to_positive_and_negative_edges
from sklearn.model_selection import KFold
import numpy as np
import torch


global temp_model
global gpus_per_trial
global device

gpus_per_trial=0
device="cpu"
def k_fold_training_large(model_hyperparameters, graph,
                          in_hyper_parameter_search=True):
    x_features = graph.x
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    positive_index, negative_index = turn_data_to_positive_and_negative_edges(graph.edge_index, graph.edge_attr)

    train_data = {
        "GAT_features": x_features,
        "GAT_edge_index": edge_index,
        "GAT_edge_attr": edge_attr,
    }

    splits = KFold(n_splits=model_hyperparameters["number_of_splits"], shuffle=True, random_state=42)
    positive_splits = splits.split(np.arange(positive_index.size()[1]))
    negative_splits = splits.split(np.arange(negative_index.size()[1]))

    for fold, (pos_index, neg_index) in enumerate(zip(positive_splits, negative_splits)):

        pos_train_idx, pos_val_idx = pos_index
        neg_train_idx, neg_val_idx = neg_index

        train_pos, eval_pos = get_train_eval_indexes(positive_index, pos_train_idx, pos_val_idx)
        train_neg, eval_neg = get_train_eval_indexes(negative_index, neg_train_idx, neg_val_idx)

        model = model_hyperparameters["model_function"](model_hyperparameters)

        # define device:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.model.to(device)

        optimizer = torch.optim.Adam(model.model.parameters(), lr=model_hyperparameters["learning_rate"],
                                     weight_decay=model_hyperparameters["weight_decay"])
        model.optimizer= optimizer

        train_data["epochs"]= model_hyperparameters["epochs" ]
        train_data["pos_index"] = train_pos
        train_data["neg_index"] = train_neg
        train_data["test_pos_index"] = eval_pos
        train_data["test_neg_index"] = eval_neg

        # only spectral information
        if (model_hyperparameters["spectral_features_type"] == 0):
            train_data["SIGNED_features"] = model.model.create_spectral_features(train_pos, train_neg)
        # only node features
        elif (model_hyperparameters["spectral_features_type"] == 1):
            train_data["SIGNED_features"] = x_features
        # using both spectral and node features
        else:
            spectral_features = temp_model.model.create_spectral_features(train_pos, train_neg)
            mixed_features = torch.cat((x_features, spectral_features), dim=1)
            print(spectral_features.shape)
            train_data["SIGNED_features"] = mixed_features
            print("size of input", x_features.size(), "size of model input",
                  model_hyperparameters["size_of_x_features"])

        model.train_fold_large(train_data, in_hyper_parameter_search=in_hyper_parameter_search,fold_number=fold)


def hyper_parameter_tuning_large(config, checkpoint_dir=None):
    k_fold_training_large(config, config["graph"])


