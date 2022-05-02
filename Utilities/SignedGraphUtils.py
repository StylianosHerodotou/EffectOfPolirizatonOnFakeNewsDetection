import torch

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