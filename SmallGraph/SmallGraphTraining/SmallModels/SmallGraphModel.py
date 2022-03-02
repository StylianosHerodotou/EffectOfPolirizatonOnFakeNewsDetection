
from abc import ABC, abstractmethod
import torch.nn.functional as F
import os
import torch
import ray

global device
class SmallGraphModel(ABC):
    def __init__(self):
        self.model=None
        self.optimizer=None

    @abstractmethod
    def forward(self,train_dic):
        pass

    def find_loss(self, output, data):
        return F.nll_loss(output, data.y)

    def train_step_small(self, train_loader):

        self.model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss =
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            self.optimizer.step()
        return loss_all / train_loader["size"]

    def test_small(self, test_loader):

        self.model.eval()
        correct = 0
        for data in test_loader:
            data = data.to(device)
            pred = self.model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / test_loader["size"]

    def train_fold_small(self, train_loader, eval_loader, fold_number=-1, checkpoint_dir="",
                         epochs=100, print_acc_every=1, in_hyper_parameter_search=True):
        best_acc = 0

        for epoch in range(1, epochs + 1):
            loss = self.train_step_small(train_loader)

            if (epoch % print_acc_every == 0):
                train_acc = self.test_small( train_loader)
                test_acc = self.test_small(eval_loader)
                best_acc = max(best_acc, test_acc)

                print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
                      f'Test Acc: {test_acc:.5f}, Best accuracy so far: {best_acc:.5f}')

            if (in_hyper_parameter_search):
                with ray.tune.checkpoint_dir((fold_number * epochs) + epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.model.state_dict(), self.state_dict()), path)

                ray.tune.report(accuracy=test_acc)