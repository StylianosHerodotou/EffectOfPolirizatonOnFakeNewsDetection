import torch
import torch.nn.functional as F
from DatasetRepresentation.DataPreprocessing.RNNProcessing import clean_column_name

class LSTMBagOfWordsEncoder(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 num_layers=2, dropout=0.2):
        super(LSTMBagOfWordsEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)


    def forward(self, data):

        sentence = data.extra_inputs[clean_column_name]
        ###TODO CHECK WHETHER IT works better if you batch like ROberta
        batch_size = data.batch.max() + 1
        single_entry_size = int(len(sentence) / batch_size)

        embeds = self.word_embeddings(sentence)
        embeds = F.dropout(embeds, p=self.dropout, training=self.training)

        lstm_out, _ = self.lstm(embeds.view(batch_size, single_entry_size, -1))
        lstm_out = F.dropout(lstm_out, p=self.dropout, training=self.training)

        # lstm_out= lstm_out.view(len(sentence), -1)
        lstm_out = lstm_out[:, -1, :]
        return lstm_out