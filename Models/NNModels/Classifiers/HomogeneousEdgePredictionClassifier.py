import torch
import torch.nn.functional as F

from Models.NNModels.Classifiers.MLP import MLP


class HeterogeneousEdgePredictionClassifier(torch.nn.Module):
    def __init__(self, in_channels, output_size, nodes_per_hidden_layer: list,
                 dropout=0, activation_function=F.relu, final_activation_function=F.log_softmax):
        super().__init__()
        self.model = MLP(in_channels, output_size, nodes_per_hidden_layer,
                         dropout, activation_function, final_activation_function)

    def forward(self, encoder_output, pyg_data):
        edge_index = pyg_data.edge_index
        row, col = edge_index

        from_node_embeddings = encoder_output[row];
        to_node_embeddings = encoder_output[col];
        embeddings_to_use = torch.cat([from_node_embeddings, to_node_embeddings], dim=-1)

        output = self.model(embeddings_to_use)
        return output