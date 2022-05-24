from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.NormalToHeteroGCN_ED_MultiTask_CompleteModel import NormalToHeteroGCN_ED_MultiTask_CompleteModel
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForHeterogeneousNetworks import \
    AbstractSimpleTrainerForHeterogeneousNetwork


class NormalToHeteroGCN_ED_MultiTask_SimpleTrainer(AbstractSimpleTrainerForHeterogeneousNetwork):

    def __init__(self):
        super().__init__()
        self.model_function =NormalToHeteroGCN_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
