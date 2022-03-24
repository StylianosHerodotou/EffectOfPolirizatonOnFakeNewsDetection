import torch
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import RobertaTokenizer


def create_training_set(train_df, hyperparameters=None):
    # san hyperparameters tha exo mia lista apo epipleon features to add.
    # kathe item tis listas ine string to function pair.


    train_set = train_df["graph"].tolist()
    y_train = train_df["label"].tolist()
    y_train = torch.LongTensor(y_train)
    y_train = y_train.reshape(y_train.size(0), 1)

    added_features = {}
    for feature_key in hyperparameters["input_types"]:
        added_features[feature_key] = hyperparameters[feature_key](train_df, hyperparameters)

    for index in range(len(train_set)):
        train_set[index].y = y_train[index]
        train_set[index].edge_index = train_set[index].edge_index.long()
        train_set[index].extra_inputs= {}
        for key in added_features:
            train_set[index].extra_inputs[key]= added_features[key][index]
    return train_set