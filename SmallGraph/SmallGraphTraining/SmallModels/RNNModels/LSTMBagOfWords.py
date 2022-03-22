from SmallGraph.SmallGraphTraining.SmallModels.SmallGraphModel import SmallGraphModel
import torch
import torch.nn.functional as F

class LSTMBagOfWordsModel(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 output_size, num_layers=2, dropout=0.2,
                 is_part_of_ensemble=False, MLP_arguments=None):
        super(LSTMBagOfWordsModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.is_part_of_ensemble = is_part_of_ensemble

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        if (is_part_of_ensemble == False):
            self.classifier = MLP(in_channels=hidden_dim, output_size=output_size,
                                  nodes_per_hidden_layer=MLP_arguments["nodes_per_hidden_layer"],
                                  number_of_hidden_layers=MLP_arguments["number_of_hidden_layers"],dropout=MLP_arguments["dropout"])

    def forward(self, data):

        sentence = data.article_rep
        batch_size = data.batch.max() + 1
        single_entry_size = int(len(sentence) / batch_size)

        embeds = self.word_embeddings(sentence)
        embeds = F.dropout(embeds, p=self.dropout, training=self.training)

        lstm_out, _ = self.lstm(embeds.view(batch_size, single_entry_size, -1))
        lstm_out = F.dropout(lstm_out, p=self.dropout, training=self.training)

        # lstm_out= lstm_out.view(len(sentence), -1)
        lstm_out = lstm_out[:, -1, :]
        if (self.is_part_of_ensemble):
            return lstm_out

        return self.classifier.forward(lstm_out)

        # tag_space = self.hidden2tag(lstm_out)
        # tag_space = self.drop(tag_space)

        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores


class LSTMBagOfWords(SmallGraphModel):

    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model = LSTMBagOfWordsModel(
            embedding_dim=model_hyperparameters["embedding_dim"],
            hidden_dim=model_hyperparameters["hidden_dim"],
            vocab_size=model_hyperparameters["vocab_size"],
            output_size=model_hyperparameters["output_size"],
            num_layers=model_hyperparameters["num_layers"],
            MLP_arguments=model_hyperparameters["MLP_arguments"])

    def forward(self, data):
        return self.model(data)

    def find_loss(self, output, data):
        return F.nll_loss(output, data.y)