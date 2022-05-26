from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
import torch

from Utilities.PerformanceTrackerUtils import to_categorical


class HeteroDataNodeClassificationTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, output, pyg_data, *args):
        loss = 0.0
        for node_type, node_prediction in output.items():
            node_labels = pyg_data[node_type].node_type_labels
            node_labels = to_categorical(node_labels, len(pyg_data.node_types))
            loss += self.criterion(node_prediction, node_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        sum = 0
        for node_type, node_prediction in output.items():
            node_labels = pyg_data[node_type].node_type_labels
            node_labels = to_categorical(node_labels, len(pyg_data.node_types))

            actual_predictions = torch.argmax(node_prediction, dim=1)
            actual_label = torch.argmax(node_labels, dim=1)

            sum += torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)
        return sum / len(output.keys())

    def desired_metric_function(self, new_value, old_value):
        return max(new_value, old_value)
