from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
from pytorch_forecasting.metrics import MAPE
import torch


def MAPE(y_pred, y):
    return ((y - y_pred).abs() / y.abs()).mean()


class EdgeRegressionPerformanceTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def loss_function(self, output, pyg_data, *args):
        edge_type = args[0]
        feature_name = args[1]
        edge_prediction = torch.squeeze(output[edge_type]).float()
        edge_labels = pyg_data[edge_type][feature_name].float()
        loss = self.criterion(edge_prediction, edge_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        edge_type = args[0]
        feature_name = args[1]
        edge_prediction = torch.squeeze(output[edge_type]).float()
        edge_labels = pyg_data[edge_type][feature_name].float()
        return MAPE(edge_prediction, edge_labels)

    def desired_metric_function(self, output, labels, *args):
        return min
