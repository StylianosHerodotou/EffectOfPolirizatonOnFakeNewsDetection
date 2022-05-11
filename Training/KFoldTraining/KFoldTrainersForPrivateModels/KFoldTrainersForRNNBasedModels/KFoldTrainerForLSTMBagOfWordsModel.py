from abc import ABC
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader

from Training.KFoldTraining.KFoldTrainersForPrivateModels.AbstractPrivateKFoldTrainer import AbstractPrivateKFoldTrainer
from Training.KFoldTraining.KFoldTrainersForPublicModels.AbstractPublicKFoldTrainer import AbstractPublicKFoldTrainer
import numpy as np
from Models.CompleteModels.PrivateModels.RNNModels.LSTMBagOfWordsCompleteModel import LSTMBagOfWordsCompleteModel
from Utilities.InitGlobalVariables import dir_to_base


class KFoldTrainerForLSTMBagOfWordsModel(AbstractPrivateKFoldTrainer, ABC):
    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)
        self.model_function = LSTMBagOfWordsCompleteModel



