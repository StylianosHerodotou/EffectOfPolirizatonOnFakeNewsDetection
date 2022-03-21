import torch
from abc import ABC, abstractmethod
import ray
import os
from Utilities.InitGlobalVariables import dir_to_ray_checkpoints

def get_train_eval_indexes(edge_index, train_idx, val_idx):
    train = list()
    eval = list()

    edge_index = torch.transpose(edge_index, 0, 1)

    for index in train_idx:
        train.append(edge_index[index])

    for index in val_idx:
        eval.append(edge_index[index])

    train = torch.stack(train)
    eval = torch.stack(eval)

    train = torch.transpose(train, 0, 1)
    eval = torch.transpose(eval, 0, 1)

    return train, eval

class LargeGraphModel(ABC):
    def __init__(self):
        self.model=None
        self.optimizer=None

    @abstractmethod
    def forward(self,train_dic):
        pass

    def find_loss(self, output,train_dic ):
        return self.model.loss(output, train_dic["pos_index"], train_dic["neg_index"])

    def test(self, output, test_dic):
        return self.model.test(output, test_dic["test_pos_index"], test_dic["test_neg_index"])


    def train_step_large(self, train_dic):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.forward(train_dic)
        loss = self.find_loss(output, train_dic)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_large(self, test_dic):
        self.model.eval()
        with torch.no_grad():
            output = self.forward(test_dic)
        return self.test(output, test_dic)

    def train_fold_large(self,train_dic,in_hyper_parameter_search=True, fold_number=-1):

        if ("test_pos_index" not in train_dic or "test_neg_index" not in train_dic):
            train_dic["test_pos_index"] = train_dic["pos_index"]
            train_dic["test_neg_index"] = train_dic["neg_index"]

        best_f1_so_far = 0.0;
        best_auc_so_far = 0.0;

        for epoch in range(train_dic["epochs"]):

            loss = self.train_step_large(train_dic)
            auc, f1 = self.test_large( train_dic)
            best_f1_so_far = max(best_f1_so_far, f1)
            best_auc_so_far = max(best_auc_so_far, auc)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, '
                  f'F1: {f1:.4f}, best f1 so far: {best_f1_so_far:.4f}, best auc so far {best_auc_so_far:.4f} ')

            if in_hyper_parameter_search:
                with ray.tune.checkpoint_dir(os.path.join(dir_to_ray_checkpoints, (fold_number * train_dic["epochs"]) + epoch)) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
                ray.tune.report(f1=f1)





