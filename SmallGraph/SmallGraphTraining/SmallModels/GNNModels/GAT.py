import torch
from torch_geometric.nn import GATv2Conv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from Models.NNModels.Classifiers.MLP import MLP
from SmallGraph.SmallGraphTraining.SmallModels.SmallGraphModel import SmallGraphModel

class GATModel(torch.nn.Module):

    def __init__(self, in_channels,output_size,edge_dim,
                 hidden_size=128, heads=8, dropout=0.2,pooling_ratio=0.8, num_layers=1,
                 is_part_of_ensemble=False, MLP_arguments=None):
        super().__init__()
        self.num_layers=num_layers
        self.is_part_of_ensemble=is_part_of_ensemble
        self.convs = torch.nn.ModuleList()
        self.pools= torch.nn.ModuleList()

        conv1 = GATv2Conv(in_channels, hidden_size, heads=heads, dropout=dropout,edge_dim=edge_dim)
        pool1 = TopKPooling(hidden_size*heads, ratio=pooling_ratio)

        self.convs.append(conv1)
        self.pools.append(pool1)

        for layer in range(self.num_layers-1):
          self.convs.append(GATv2Conv(hidden_size*heads, hidden_size, heads=heads, dropout=dropout,edge_dim=edge_dim))
          self.pools.append(TopKPooling(hidden_size*heads, ratio=pooling_ratio))

        if(is_part_of_ensemble==False):
            self.classifier =MLP(in_channels=2*hidden_size*heads,output_size=output_size, nodes_per_hidden_layer=MLP_arguments["nodes_per_hidden_layer"],
                 dropout=MLP_arguments["dropout"])


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch


        x = F.relu(self.convs[0](x, edge_index,edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pools[0](x, edge_index, edge_attr, batch)
        cont = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for layer_index in range(1,self.num_layers):
          x = F.relu(self.convs[layer_index](x, edge_index,edge_attr))
          x, edge_index, edge_attr, batch, _, _ = self.pools[layer_index](x, edge_index, edge_attr, batch)
          cont += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = cont
        if(self.is_part_of_ensemble):
            return x

        return self.classifier.forward(x)


class GAT(SmallGraphModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        sample = model_hyperparameters["train_set"][0]
        self.model= GATModel(in_channels=sample.x.size()[1],
                         output_size=model_hyperparameters["num_classes"],
                         edge_dim=sample.edge_attr.size()[1],
                         hidden_size=model_hyperparameters["hidden_size"],
                         heads=model_hyperparameters["heads"],
                         dropout=model_hyperparameters["dropout"],
                         pooling_ratio=model_hyperparameters["pooling_ratio"],
                         num_layers=model_hyperparameters["num_layers"],
                         MLP_arguments=model_hyperparameters["MLP_arguments"])

    def forward(self, data):
        return self.model(data)

    def find_loss(self, output, data):
        return F.nll_loss(output, data.y)
