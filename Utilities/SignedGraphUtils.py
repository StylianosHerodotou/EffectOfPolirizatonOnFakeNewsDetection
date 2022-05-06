import numpy as np
import torch
from torch_geometric.data import HeteroData
def get_train_eval_indexes(edge_index, train_idx, val_idx):
    train = list()
    eval = list()

    edge_index = torch.transpose(edge_index, 0, 1)

    # for index in train_idx:
    #     train.append(edge_index[index])
    #
    # for index in val_idx:
    #     eval.append(edge_index[index])
    #
    # train = torch.stack(train)
    # eval = torch.stack(eval)

    train = edge_index[train_idx]
    eval = edge_index[val_idx]

    train = torch.transpose(train, 0, 1)
    eval = torch.transpose(eval, 0, 1)

    print(train.size())
    print(eval.size())

    return train, eval


def turn_data_to_positive_and_negative_edges(edge_index, edge_attr):
    positive_index_from = list()
    positive_index_to = list()

    negative_index_from = list()
    negative_index_to = list()
    numberOfEdges = edge_attr.size()[0]
    for index in range(0, numberOfEdges):
        if (edge_attr[index][0] == 1.0):
            positive_index_from.append(edge_index[0][index])
            positive_index_to.append(edge_index[1][index])
        else:
            negative_index_from.append(edge_index[0][index])
            negative_index_to.append(edge_index[1][index])

    positive_index = torch.tensor(np.array([positive_index_from, positive_index_to])).long()
    negative_index = torch.tensor(np.array([negative_index_from, negative_index_to])).long()
    return positive_index, negative_index


class DataInBatchesOnlySigned(HeteroData):
    def __init__(self, signed_x, pos_edge_index, neg_edge_index):
        super().__init__()
        self.signed_x = signed_x
        self.pos_edge_index = pos_edge_index
        self.neg_edge_index = neg_edge_index