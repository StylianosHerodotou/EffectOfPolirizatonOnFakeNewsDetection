import torch
from abc import ABC, abstractmethod

from Models.CompleteModels.AbstractCompleteModel import AbstractCompleteModel
from Utilities.InitGlobalVariables import device
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class AbstractCompletePrivateModel(AbstractCompleteModel, ABC):
    def __init__(self):
        super().__init__()
        self.prefered_metric_value = "fbeta_score"

    def update_total_loss(self, loss, loss_all, data):
        loss_all += data.num_graphs * loss
        return loss_all

    def get_mean_total_loss(self, loss_all, train_loader):
        return loss_all / train_loader["size"]

    def get_test_scores(self, all_predicted_values, all_true_labels):
        current_confusion_matrix = confusion_matrix(all_true_labels, all_predicted_values)
        current_accuracy = accuracy_score(all_true_labels, all_predicted_values)
        current_precision, current_recall, current_fbeta_score, current_support = precision_recall_fscore_support(
            all_true_labels, all_predicted_values, average='micro')

        scores = {"confusion_matrix": current_confusion_matrix,
                  "accuracy": current_accuracy,
                  "precision": current_precision,
                  "recall": current_recall,
                  "fbeta_score": current_fbeta_score,
                  "support": current_support}

        return scores

    def find_performance(self, output, data):
        encoder_output, decoder_output = output
        prediction = decoder_output.max(dim=1)[1]
        prediction = prediction.detach().numpy().tolist()
        true_labels = data.y.detach().numpy().tolist()
        return prediction, true_labels

    def train_step(self, train_loader):
        self.set_model_parameters_to_training_mode()
        loss_all = 0
        for data in train_loader["loader"]:
            data = data.to(device)
            self.zero_grad_optimizer()
            output = self.forward(data)
            loss = self.find_loss(output, data)
            loss_all = self.update_total_loss(loss, loss_all, data)
            self.loss_backward(loss)
            self.optimizer_step()
        return self.get_mean_total_loss(loss_all, train_loader)

    def test(self, test_loader):
        self.set_model_parameters_to_test_mode()
        all_true_labels = list()
        all_predicted_values = list()
        for data in test_loader["loader"]:
            data = data.to(device)
            output = self.forward(data)
            prediction, true_labels = self.find_performance(output, data)
            all_predicted_values.extend(prediction)
            all_true_labels.extend(true_labels)
        return self.get_test_scores(all_predicted_values, all_true_labels)

    def init_performance_metric(self):
        scores = {"confusion_matrix": [[0, 0], [0, 0]],
                  "accuracy": 0,
                  "precision": 0,
                  "recall": 0,
                  "fbeta_score": 0,
                  "support": 0}
        return scores

    # for each
    def get_best_performance_metric_so_far(self, current_performance_metric_dict, new_performance_doct):

        if new_performance_doct[self.prefered_metric_value] > current_performance_metric_dict[self.prefered_metric_value]:
            current_performance_metric_dict[self.prefered_metric_value] = new_performance_doct[
                self.prefered_metric_value]
            current_performance_metric_dict["confusion_matrix"] = new_performance_doct["confusion_matrix"]

        for key in current_performance_metric_dict.keys():
            if key == "confusion_matrix" or key == self.prefered_metric_value:
                continue

            old_value= current_performance_metric_dict[key]
            new_value= new_performance_doct[key]

            if new_value is None:
                continue

            current_performance_metric_dict[key] = max(old_value,new_value)

        return current_performance_metric_dict

    def loss_to_string(self, loss):
        to_return = str("{:.2f}".format(loss))
        return to_return

    def get_report_score(self, performance_metric_dict):
        return performance_metric_dict[self.prefered_metric_value]

    def best_performance_metric_to_string(self, performance_metric_dict):
        best_performance_metric_string = str("{:.2f}".format(self.get_report_score(performance_metric_dict)))
        return best_performance_metric_string

    def performance_string(self, performance_metric_dict):
        to_return = ""
        for metric_name, current_metric in performance_metric_dict.items():
            if metric_name != "confusion_matrix" and current_metric is not None:
                to_return += str(metric_name) + ": " + str("{:.2f}".format(current_metric)) + " ,"
        return to_return
