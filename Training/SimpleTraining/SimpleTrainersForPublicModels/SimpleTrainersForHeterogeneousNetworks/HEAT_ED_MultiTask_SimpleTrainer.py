from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.\
    HEAT_ED_MultiTask_CompleteModel import HEAT_ED_MultiTask_CompleteModel
from Training.SimpleTraining.SimpleTrainersForPublicModels.SimpleTrainersForHeterogeneousNetworks.AbstractSimpleTrainerForHeterogeneousNetwork import \
    AbstractSimpleTrainerForHeterogeneousNetwork


class HEAT_ED_MultiTask_SimpleTrainer(AbstractSimpleTrainerForHeterogeneousNetwork):

    def __init__(self):
        super().__init__()
        self.model_function = HEAT_ED_MultiTask_CompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
