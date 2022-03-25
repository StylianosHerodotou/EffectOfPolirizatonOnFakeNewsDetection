import torch
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


clean_column_name="clean_column_name"
def generate_bag_of_words_embeddings_for_training_set(train_df, hyperparameters):

    corpus = list(train_df[clean_column_name])
    onehot_rep = [one_hot(words, hyperparameters["vocab_size"]) for words in corpus]
    embedded_docs = pad_sequences(onehot_rep, padding='pre', maxlen=hyperparameters["maxlen"])
    bag_of_words_representation = torch.from_numpy(embedded_docs).long()
    return bag_of_words_representation

def get_bag_of_words_embeddings_for_training_set(train_df, hyperparameters):
    return generate_bag_of_words_embeddings_for_training_set(train_df, hyperparameters,
                                                 clean_column_name)


