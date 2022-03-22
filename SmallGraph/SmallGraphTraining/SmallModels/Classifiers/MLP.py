import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels, output_size, nodes_per_hidden_layer: list, number_of_hidden_layers: int,
                 dropout=0, activation_function=F.relu, final_activation_function=F.log_softmax):
        super().__init__()
        self.dropout = dropout
        self.activation_function = activation_function
        self.final_activation_function = final_activation_function
        self.lin = torch.nn.ModuleList()

        if nodes_per_hidden_layer == 0:
            self.lin.append(torch.nn.Linear(in_channels, output_size))
        else:
            self.lin.append(torch.nn.Linear(in_channels, nodes_per_hidden_layer[0]))
            for index in range(1, number_of_hidden_layers):
                self.lin.append(torch.nn.Linear(nodes_per_hidden_layer[index - 1], nodes_per_hidden_layer[index]))
            self.lin.append(torch.nn.Linear(nodes_per_hidden_layer[number_of_hidden_layers - 1], output_size))

    def forward(self, data):
        number_of_layers = len(self.lin)
        if number_of_layers == 1:
            return self.final_activation_function(data)

        # data=self.activation_function(self.lin[0](data))
        for index in range(len(self.lin) - 1):
            data = self.activation_function(self.lin[index](data))
            data = F.dropout(data, p=self.dropout, training=self.training)

        output = self.final_activation_function(self.lin[number_of_layers - 1](data), dim=-1)
        return output
