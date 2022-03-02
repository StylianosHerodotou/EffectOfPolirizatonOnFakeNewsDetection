import torch
def create_training_set(train_df, hyperparameters=None, add_lstm_rpresentation=False):
    article_representation_train = None

    train_set = train_df["graph"].tolist()

    y_train = train_df["label"].tolist()
    y_train = torch.LongTensor(y_train)
    y_train = y_train.reshape(y_train.size(0), 1)

    # if (add_lstm_rpresentation):
    #     article_representation_train = TextPreprocessing.get_lstm_article_representation(train_df, hyperparameters)

    for index in range(len(train_set)):
        train_set[index].y = y_train[index]
        train_set[index].edge_index = train_set[index].edge_index.long()
        # if (add_lstm_rpresentation):
        #     train_set[index].article_rep = article_representation_train[index]
    return train_set