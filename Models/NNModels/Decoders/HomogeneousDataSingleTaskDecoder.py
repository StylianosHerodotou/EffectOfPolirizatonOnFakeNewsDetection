from Models.NNModels.Classifiers.HomogeneousEdgePredictionClassifier import HomogeneousEdgePredictionClassifier
from Models.NNModels.Classifiers.MLP import MLP
import torch


class HomogeneousDataSingleTaskDecoder(torch.nn.Module):
    def __init__(self, in_channels, pyg_data, classifier_arguments):
        super().__init__()
        self.loss_arguments = None
        self.classifier_type = classifier_arguments["classifier_type"]

        if classifier_arguments["classifier_type"] == "edge":
            self.classifier = HomogeneousEdgePredictionClassifier(in_channels=2 * in_channels,
                                                                  output_size=classifier_arguments["output_size"],
                                                                  nodes_per_hidden_layer=classifier_arguments[
                                                                      "nodes_per_hidden_layer"],
                                                                  dropout=classifier_arguments["dropout"],
                                                                  activation_function=classifier_arguments[
                                                                      "activation_function"],
                                                                  final_activation_function=classifier_arguments[
                                                                      "final_activation_function"]
                                                                  )
            if "edge_type" in classifier_arguments.keys():
                self.loss_arguments = {"edge_type": classifier_arguments["edge_type"],
                                       "feature_name": classifier_arguments["feature_name"]}
        else:
            self.classifier = MLP(in_channels=in_channels,
                                  output_size=classifier_arguments["output_size"],
                                  nodes_per_hidden_layer=classifier_arguments["nodes_per_hidden_layer"],
                                  dropout=classifier_arguments["dropout"],
                                  activation_function=classifier_arguments["activation_function"],
                                  final_activation_function=classifier_arguments["final_activation_function"])

        # in both cases this is this tasks optimizer.
        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=classifier_arguments["learning_rate"])
        self.performance_tracker = classifier_arguments["performance_tracker"]()
        self.metric_weight = classifier_arguments["metric_weight"]

    def forward(self, data, encoder_output):
        if self.classifier_type == "edge":
            output = self.classifier.forward(encoder_output, data)
        else:
            output = self.classifier.forward(encoder_output)
        return output
