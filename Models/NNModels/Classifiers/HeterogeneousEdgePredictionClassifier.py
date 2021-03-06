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
        edge_index_dict = pyg_data.edge_index_dict

        decoder_output = {}
        for edge_type, edge_index in edge_index_dict.items():
            row, col = edge_index
            from_node_type = edge_type[0]
            to_node_type = edge_type[2]

            from_node_embeddings = encoder_output[from_node_type][row];
            to_node_embeddings = encoder_output[to_node_type][col];
            embeddings_to_use = torch.cat([from_node_embeddings, to_node_embeddings], dim=-1)

            output = self.model(embeddings_to_use)
            decoder_output[edge_type] = output
        return decoder_output