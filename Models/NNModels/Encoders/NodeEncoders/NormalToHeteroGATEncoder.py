from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.nn import HeteroConv
import torch


def get_single_normal_to_hetero_gat_layer(pyg_data, in_channels,
                                          hidden_channels, heads, dropout):
    conv_dict = {}
    for edge_type in pyg_data.edge_types:
        if ("edge_attr" in pyg_data[edge_type].keys()):
            edge_feature_dim = pyg_data[edge_type]["edge_attr"].size()[1]
            conv_dict[edge_type] = GATv2Conv(in_channels, hidden_channels, heads=heads,
                                             dropout=dropout, edge_dim=edge_feature_dim,
                                             add_self_loops=False)
        else:
            conv_dict[edge_type] = GATv2Conv(in_channels, hidden_channels, heads=heads,
                                             dropout=dropout, edge_dim=None,
                                             add_self_loops=False)
    return HeteroConv(conv_dict)


class NormalToHeteroGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, pyg_data,
                 hidden_channels=128, heads=8, dropout=0.2, num_layers=1):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        first_conv = get_single_normal_to_hetero_gat_layer(pyg_data, in_channels,
                                                           hidden_channels, heads, dropout)
        self.convs.append(first_conv)

        for layer in range(num_layers - 1):
            current_conv = get_single_normal_to_hetero_gat_layer(pyg_data, in_channels=hidden_channels * heads,
                                                                 hidden_channels=hidden_channels, heads=heads,
                                                                 dropout=dropout)
            self.convs.append(current_conv)

    def forward(self, pyg_data):
        x_dict, edge_index_dict, edge_attr_dict = pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.edge_attr_dict
        for index, conv_layer in enumerate(self.convs):
            x_dict = conv_layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x for key, x in x_dict.items()}
        return x_dict