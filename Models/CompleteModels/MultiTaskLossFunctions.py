from pytorch_forecasting.metrics import MAPE

from torch import nn
import torch
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return  torch.from_numpy(np.eye(num_classes, dtype='float64')[y])

def MAPE(y_pred, y):
    return ((y - y_pred).abs() / y.abs()).mean()

def edge_regression_loss_function(decoder_output, pyg_data, edge_type, feature_name):
    criterion = nn.MSELoss()
    edge_prediction = torch.squeeze(decoder_output[edge_type]).float()
    edge_labels = pyg_data[edge_type][feature_name].float()
    loss = criterion(edge_prediction, edge_labels)
    return loss, MAPE(edge_prediction, edge_labels)


def edge_classification_loss_function(decoder_output, pyg_data):
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    sum = 0.0
    for edge_type, edge_prediction in decoder_output.items():
        edge_labels = pyg_data[edge_type].edge_type_labels
        edge_labels = to_categorical(edge_labels, len(pyg_data.edge_types))

        actual_predictions = torch.argmax(edge_prediction, dim=1)
        actual_label = torch.argmax(edge_labels, dim=1)
        #         print(node_prediction.size(), node_labels.size())
        #         print(actual_predictions.size(), actual_label.size())
        sum += torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)

        loss += criterion(edge_prediction, edge_labels)
    return loss, sum / len(decoder_output.keys())


def node_classification_loss_function(out, pyg_data):
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    sum = 0
    for node_type, node_prediction in out.items():
        node_labels = pyg_data[node_type].node_type_labels
        node_labels = to_categorical(node_labels, len(pyg_data.node_types))
        loss += criterion(node_prediction, node_labels)

        actual_predictions = torch.argmax(node_prediction, dim=1)
        actual_label = torch.argmax(node_labels, dim=1)
        #         print(node_prediction.size(), node_labels.size())
        #         print(actual_predictions.size(), actual_label.size())
        sum += torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)
    return loss, sum / len(out.keys())