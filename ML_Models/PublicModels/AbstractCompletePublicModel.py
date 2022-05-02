import torch
from abc import ABC

from ML_Models.AbstractCompleteModel import AbstractCompleteModel


class AbstractCompletePublicModel(AbstractCompleteModel, ABC):
    def __init__(self):
        super().__init__()

    def train_step(self, train_dic):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.forward(train_dic)
        loss = self.find_loss(output, train_dic)
        self.loss_backward(loss)
        self.optimizer.step()
        return loss

    def test(self, test_dic):
        self.model.eval()
        with torch.no_grad():
            output = self.forward(test_dic)
        return self.find_performance(output, test_dic)
