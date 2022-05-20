from .AbstractKFoldTrainerForHeterogeneousNetworks import AbstractPublicKFoldTrainerForHeterogeneousNetworks
from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.HAN_ED_MultiTask_CompleteModel import \
    HAN_ED_MultiTask_CompleteModel

class HAN_ED_MultiTask_KFoldTrainer(AbstractPublicKFoldTrainerForHeterogeneousNetworks):

    def __init__(self, number_of_splits, random_state=42):
        super().__init__(number_of_splits, random_state)
        self.model_function = HAN_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
