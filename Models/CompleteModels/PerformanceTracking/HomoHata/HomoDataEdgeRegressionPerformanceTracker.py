from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
import torch
import torch.nn.functional as F

from Models.CompleteModels.PerformanceTracking.RMSELossFunction import RMSELoss


class HomoDataEdgeRegressionPerformanceTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
        self.metric_criterion= torch.nn.L1Loss()

    def loss_function(self, output, pyg_data, *args):
        edge_prediction = torch.squeeze(output).float()
        edge_labels = torch.squeeze(pyg_data.edge_attr).float()
        loss = self.criterion(edge_prediction, edge_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        edge_prediction = torch.squeeze(output).float()
        edge_labels = torch.squeeze(pyg_data.edge_attr).float()
        return self.metric_criterion(edge_prediction, edge_labels).item()

    def desired_metric_function(self, new_value, old_value):
        return min(new_value, old_value)
