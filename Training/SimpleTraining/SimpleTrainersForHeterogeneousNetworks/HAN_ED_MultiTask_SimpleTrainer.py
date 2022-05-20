from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.HAN_ED_MultiTask_CompleteModel import HAN_ED_MultiTask_CompleteModel
from Training.SimpleTraining.SimpleTrainersForHeterogeneousNetworks.AbstractSimpleTrainerForHeterogeneousNetwork import \
    AbstractSimpleTrainerForHeterogeneousNetwork


class HAN_ED_MultiTask_SimpleTrainer(AbstractSimpleTrainerForHeterogeneousNetwork):

    def __init__(self):
        super().__init__()
        self.model_function =HAN_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
