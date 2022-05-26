from Models.CompleteModels.PerformanceTracking.AbstractPerformanceTracker import AbstractPerformanceTracker
import torch

from Utilities.PerformanceTrackerUtils import to_categorical


class HeteroDataEdgeClassificationTracker(AbstractPerformanceTracker):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss_function(self, output, pyg_data, *args):
        loss = 0.0
        for edge_type, edge_prediction in output.items():
            edge_labels = pyg_data[edge_type].edge_type_labels
            edge_labels = to_categorical(edge_labels, len(pyg_data.edge_types))

            loss += self.criterion(edge_prediction, edge_labels)
        return loss

    def metric_function(self, output, pyg_data, *args):
        sum = 0.0
        for edge_type, edge_prediction in output.items():
            edge_labels = pyg_data[edge_type].edge_type_labels
            edge_labels = to_categorical(edge_labels, len(pyg_data.edge_types))

            actual_predictions = torch.argmax(edge_prediction, dim=1)
            actual_label = torch.argmax(edge_labels, dim=1)

            sum += torch.sum(actual_predictions == actual_label) / actual_predictions.size(0)

        return sum / len(output.keys())

    def desired_metric_function(self, new_value, old_value):
        return max(new_value, old_value)
