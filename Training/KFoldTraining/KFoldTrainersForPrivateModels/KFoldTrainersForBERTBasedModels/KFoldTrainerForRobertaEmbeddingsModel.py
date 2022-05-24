from Training.KFoldTraining.KFoldTrainersForPrivateModels.AbstractPrivateKFoldTrainer import AbstractPrivateKFoldTrainer
from Models.CompleteModels.PrivateModels.BertBasedModels.RobertaEmbeddingsCompleteModel import RobertaEmbeddingsCompleteModel


class KFoldTrainerForRobertaEmbeddingsModel(AbstractPrivateKFoldTrainer):
    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)
        self.model_function = RobertaEmbeddingsCompleteModel
