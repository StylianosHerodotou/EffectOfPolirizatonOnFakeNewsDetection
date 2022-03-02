from LargeGraph.LargeGraphTraining.LargeModels.LargeGraphModel import LargeGraphModel
from torch_geometric.nn import SignedGCN

class SignedGCNModel(LargeGraphModel):
    def __init__ (self, model_hyperparameters):
        super().__init__()
        model = SignedGCN(in_channels=model_hyperparameters["size_of_x_features"],
                          hidden_channels=model_hyperparameters["hidden_nodes"],
                          num_layers=model_hyperparameters["num_layers"],
                          lamb=model_hyperparameters["lamb"],
                          )
        self.model=model

    def forward(self,train_dic):
        signed_x = train_dic["SIGNED_features"]
        pos_edge_index = train_dic["pos_index"]
        neg_edge_index = train_dic["neg_index"]
        return self.model(signed_x, pos_edge_index, neg_edge_index)
