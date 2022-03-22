import torch
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_bag_of_words_embeddings_for_training_set(train_df, hyperparameters,
                                                 clean_column_name="clean_article"):

    corpus = list(train_df[clean_column_name])
    onehot_rep = [one_hot(words, hyperparameters["vocab_size"]) for words in corpus]
    embedded_docs = pad_sequences(onehot_rep, padding='pre', maxlen=hyperparameters["maxlen"])
    article_representation = torch.from_numpy(embedded_docs).long()
    return article_representation


def create_training_set(train_df, hyperparameters=None):
    article_representation_train = None

    train_set = train_df["graph"].tolist()

    y_train = train_df["label"].tolist()
    y_train = torch.LongTensor(y_train)
    y_train = y_train.reshape(y_train.size(0), 1)

    if("bag_of_words" in hyperparameters["input_type"]):
        article_representation_train= get_bag_of_words_embeddings_for_training_set(train_df, hyperparameters)

    for index in range(len(train_set)):
        train_set[index].y = y_train[index]
        train_set[index].edge_index = train_set[index].edge_index.long()
        if (article_representation_train is not None):
            train_set[index].article_rep = article_representation_train[index]
    return train_set