from ML_Models.PublicModels.LargeGraphModel import LargeGraphModel
from torch_geometric.nn import SignedGCN
import torch


class SignedGCNCompleteModel(LargeGraphModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        model = SignedGCN(in_channels=model_hyperparameters["size_of_x_features"],
                          hidden_channels=model_hyperparameters["hidden_nodes"],
                          num_layers=model_hyperparameters["num_layers"],
                          lamb=model_hyperparameters["lamb"],
                          )
        self.model = model

        optimizer = torch.optim.Adam(model.model.parameters(), lr=model_hyperparameters["learning_rate"],
                                     weight_decay=model_hyperparameters["weight_decay"])
        self.optimizer = optimizer

    def forward(self, train_dic):
        signed_x = train_dic["SIGNED_features"]
        pos_edge_index = train_dic["pos_index"]
        neg_edge_index = train_dic["neg_index"]
        return self.model(signed_x, pos_edge_index, neg_edge_index)

    def find_loss(self, output, train_dic):
        return self.model.loss(output, train_dic["pos_index"], train_dic["neg_index"])

    def find_performance(self, output, test_dic):
        return self.model.test(output, test_dic["pos_index"], test_dic["neg_index"])

    # the below are for recording purposes.
    def init_performance_metric(self):
        initial_performance_metric = {
            "f1": 0,
            "auc": 0
        }
        return initial_performance_metric

    def get_best_performance_metric_so_far(self, current_performance_metric, new_performance):
        new_auc, new_f1 = new_performance
        current_performance_metric["f1"] = max(new_f1, current_performance_metric["f1"])
        current_performance_metric["f1"] = max(new_auc, current_performance_metric["f1"])
        return current_performance_metric

    def loss_backward(self, loss):
        loss.backward()

    def loss_to_string(self, loss):
        return str(loss.item)

    def performance_metric_to_string(self, performance_metric):
        string = ""
        for key, value in performance_metric.items():
            string += key + ": " + value + "\n"
        return string

    def performance_string(self, performance):
        auc, f1 = performance
        return "auc: " + auc + " f1 score: " + f1

    def get_report_score(self, performance):
        auc, f1 = performance
        return f1
