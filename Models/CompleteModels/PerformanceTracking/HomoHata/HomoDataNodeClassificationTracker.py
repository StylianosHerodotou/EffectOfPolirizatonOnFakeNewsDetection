from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
import torch

from Utilities.PerformanceTrackerUtils import to_categorical


class HomoDataNodeClassificationTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, output, pyg_data, *args):
        node_labels = pyg_data.node_type
        node_labels = to_categorical(node_labels, pyg_data.num_node_types)
        loss = self.criterion(output, node_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        node_labels = pyg_data.node_type
        node_labels = to_categorical(node_labels, pyg_data.num_node_types)

        actual_predictions = torch.argmax(output, dim=1)
        actual_label = torch.argmax(node_labels, dim=1)

        return torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)

    def desired_metric_function(self, new_value, old_value):
        return max(new_value, old_value)
