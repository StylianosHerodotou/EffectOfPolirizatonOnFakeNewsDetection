
from abc import ABC, abstractmethod
import torch.nn.functional as F
import os
import torch
import ray

from Utilities.InitGlobalVariables import device, dir_to_ray_checkpoints
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class SmallGraphModel(ABC):
    def __init__(self):
        self.model=None
        self.optimizer=None
        self.is_part_of_ensemble=False

    @abstractmethod
    def forward(self,train_dic):
        pass

    @abstractmethod
    def find_loss(self, output, data):
        pass

    def train_step_small(self, train_loader):

        self.model.train()
        loss_all = 0
        for data in train_loader["loader"]:
            data = data.to(device)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.find_loss(output, data)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            self.optimizer.step()
        return loss_all / train_loader["size"]

    def find_performance(self, output, data):
        prediction = output.max(dim=1)[1]
        prediction = prediction.detach().numpy().tolist()
        true_labels = data.y.detach().numpy().tolist()
        return prediction, true_labels

    def get_test_scores(self, all_predicted_values, all_true_labels):
        current_accuracy = accuracy_score(all_true_labels, all_predicted_values)
        current_precision, current_recall, current_fbeta_score, current_support = precision_recall_fscore_support(
            all_true_labels, all_predicted_values, average='micro')

        scores = {"accuracy": current_accuracy,
                  "precision": current_precision,
                  "recall": current_recall,
                  "fbeta_score": current_fbeta_score}

        return scores


    def test_small(self, test_loader):

        self.model.eval()
        all_true_labels = list()
        all_predicted_values = list()
        for data in test_loader["loader"]:
            data = data.to(device)
            output = self.forward(data)
            prediction, true_labels = self.find_performance(output, data)
            all_predicted_values.extend(prediction)
            all_true_labels.extend(true_labels)
        return self.get_test_scores(all_predicted_values, all_true_labels)

    # def test_small(self, test_loader):
    #
    #     self.model.eval()
    #     correct = 0
    #     for data in test_loader["loader"]:
    #         data = data.to(device)
    #         pred = self.forward(data).max(dim=1)[1]
    #         correct += pred.eq(data.y).sum().item()
    #     return correct / test_loader["size"]

    def metric_dict_to_string(self, dic):
        to_return = ""
        for key, value in dic.items():
            to_return+= key + ": " +  str(dic[key]) + ", "

        to_return+= "\n"
        return to_return
    def train_fold_small(self, train_loader, eval_loader, fold_number=-1, checkpoint_dir="",
                         epochs=100, print_acc_every=1, in_hyper_parameter_search=True):
        best_dict = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "fbeta_score": 0
        }

        for epoch in range(1, epochs + 1):
            loss = self.train_step_small(train_loader)

            if (epoch % print_acc_every == 0):
                train_acc = self.test_small( train_loader)
                test_acc = self.test_small(eval_loader)

                for key,value in test_acc.items():
                    best_dict[key]=max(best_dict[key],value)


                print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train: {self.metric_dict_to_string(train_acc)}'
                      f'Test Acc: {self.metric_dict_to_string(test_acc)}'
                      f', Best accuracy so far: {self.metric_dict_to_string(best_dict)}')

            if (in_hyper_parameter_search):
                # with ray.tune.checkpoint_dir(os.path.join(dir_to_ray_checkpoints,str((fold_number * epochs) + epoch))) as checkpoint_dir:
                #     path = os.path.join(checkpoint_dir, "checkpoint")
                #     torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

                ray.tune.report(accuracy=test_acc["fbeta_score"])

    #TODO create an deep eval method that is going to do an in depth evaluation of the model,
    #find statistics like recall, f1 score, that table exc.