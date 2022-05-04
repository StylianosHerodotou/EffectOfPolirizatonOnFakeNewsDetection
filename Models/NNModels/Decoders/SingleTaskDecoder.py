from Models.NNModels.Classifiers import EdgePredictionClassifier
from Models.NNModels.Classifiers import MLP
import torch
from torch_geometric.nn import to_hetero


class SingleTaskDecoder(torch.nn.Module):
    def __init__(self, in_channels, pyg_data, classifier_arguments):
        super().__init__()

        if classifier_arguments["classifier_type"] == "edge":
            self.classifier = EdgePredictionClassifier(in_channels=in_channels,
                                                       output_size=classifier_arguments["output_size"],
                                                       nodes_per_hidden_layer=classifier_arguments[
                                                           "nodes_per_hidden_layer"],
                                                       number_of_hidden_layers=classifier_arguments[
                                                           "number_of_hidden_layers"],
                                                       dropout=classifier_arguments["dropout"],
                                                       activation_function=classifier_arguments["activation_function"],
                                                       final_activation_function=classifier_arguments[
                                                           "final_activation_function"]
                                                       )
            self.loss_arguments = {"edge_type": classifier_arguments["edge_type"],
                                   "feature_name": classifier_arguments["feature_name"]}
        else:
            self.classifier = MLP(in_channels=in_channels,
                                  output_size=classifier_arguments["output_size"],
                                  nodes_per_hidden_layer=classifier_arguments["nodes_per_hidden_layer"],
                                  number_of_hidden_layers=classifier_arguments["number_of_hidden_layers"],
                                  dropout=classifier_arguments["dropout"],
                                  activation_function=classifier_arguments["activation_function"],
                                  final_activation_function=classifier_arguments["final_activation_function"])
            self.classifier = to_hetero(self.classifier, pyg_data.metadata(),
                                        aggr=classifier_arguments["aggergation_function"])
            self.loss_arguments=None

        # in both cases this is this tasks optimizer.
        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=classifier_arguments["learning_rate"],
                                          weight_decay=classifier_arguments["weight_decay"])
        self.loss_function = classifier_arguments["loss_function"]
        self.metric_weight = classifier_arguments["metric_weight"]

    def forward(self, data, encoder_output):
        if self.loss_arguments==None:
            output = self.classifier.forward(encoder_output, data)
        else:
            output = self.classifier.forward(encoder_output)
        return output