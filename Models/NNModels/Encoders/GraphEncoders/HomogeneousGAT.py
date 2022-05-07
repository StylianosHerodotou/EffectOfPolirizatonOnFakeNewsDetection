import torch
from torch_geometric.nn import GATv2Conv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from Models.NNModels.Classifiers.MLP import MLP

class HomogeneousGAT(torch.nn.Module):

    def __init__(self, in_channels,edge_dim,
                 hidden_size=128, heads=8, dropout=0.2,pooling_ratio=0.8, num_layers=1):
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


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch


        x = F.relu(self.convs[0](x, edge_index,edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pools[0](x, edge_index, edge_attr, batch)
        cont = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for layer_index in range(1,self.num_layers):
          x = F.relu(self.convs[layer_index](x, edge_index,edge_attr))
          x, edge_index, edge_attr, batch, _, _ = self.pools[layer_index](x, edge_index, edge_attr, batch)
          cont += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return cont
