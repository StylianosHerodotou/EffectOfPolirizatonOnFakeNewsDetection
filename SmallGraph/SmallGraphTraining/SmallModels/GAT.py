import torch
from torch_geometric.nn import GATv2Conv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_mean_pool as gmeanp
import torch.nn.functional as F

from SmallGraph.SmallGraphTraining.SmallModels.SmallGraphModel import SmallGraphModel


class GATGNN(torch.nn.Module):
    def __init__(self,in_channels,output_size,edge_dim,
                 hidden_size=128, heads=8, dropout=0.2,pooling_ratio=0.8, num_layers=1 ):
        super().__init__()
        self.num_layers=num_layers

        self.convs = torch.nn.ModuleList()
        self.pools= torch.nn.ModuleList()

        conv1 = GATv2Conv(in_channels, hidden_size, heads=heads, dropout=dropout,edge_dim=edge_dim)
        pool1 = TopKPooling(hidden_size*heads, ratio=pooling_ratio)

        self.convs.append(conv1)
        self.pools.append(pool1)

        for layer in range(self.num_layers-1):
          self.convs.append(GATv2Conv(hidden_size*heads, hidden_size, heads=heads, dropout=dropout,edge_dim=edge_dim))
          self.pools.append(TopKPooling(hidden_size*heads, ratio=pooling_ratio))


        self.lin1 = torch.nn.Linear(2*hidden_size*heads, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, output_size)

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

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class GATModel(SmallGraphModel):
    def __init__ (self, model_hyperparameters):
        super().__init__()
        sample = model_hyperparameters["train_set"][0]
        model = GATGNN(in_channels=sample.x.size()[1],
                    output_size=model_hyperparameters["num_classes"],
                    edge_dim=sample.edge_attr.size()[1],
                    hidden_size=model_hyperparameters["hidden_size"],
                    heads=model_hyperparameters["heads"],
                    dropout=model_hyperparameters["dropout"],
                    pooling_ratio=model_hyperparameters["pooling_ratio"],
                    num_layers=model_hyperparameters["num_layers"])
        self.model=model

    def forward(self,data):
        return self.model(data)
