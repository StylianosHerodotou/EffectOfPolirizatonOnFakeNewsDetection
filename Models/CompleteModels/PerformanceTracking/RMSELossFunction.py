import torch


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, ypred, y):
        loss = torch.sqrt(self.mse(ypred, y) + self.eps)
        return loss
