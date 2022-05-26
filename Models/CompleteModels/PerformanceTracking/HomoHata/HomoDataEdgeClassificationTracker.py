from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
import torch

from Utilities.PerformanceTrackerUtils import to_categorical

class HomoDataEdgeClassificationTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, output, pyg_data, *args):
        edge_labels = pyg_data.edge_type
        edge_labels = to_categorical(edge_labels, pyg_data.edge_labels)
        loss = self.criterion(output, edge_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        edge_labels = pyg_data.edge_type
        edge_labels = to_categorical(edge_labels, pyg_data.edge_labels)

        actual_predictions = torch.argmax(output, dim=1)
        actual_label = torch.argmax(edge_labels, dim=1)

        return torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)

    def desired_metric_function(self, new_value, old_value):
        return max(new_value, old_value)
