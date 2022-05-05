from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.NormalToHeteroMultiTaskCompleteModel import \
    NormalToHeteroMultiTaskCompleteModel
from Training.SimpleTraining.SimpleTrainersForHeterogeneousNetworks.AbstractSimpleTrainerForHeterogeneousNetwork import \
    AbstractSimpleTrainerForHeterogeneousNetwork


class NormalToHeteroMultiTaskTrainer(AbstractSimpleTrainerForHeterogeneousNetwork):

    def __init__(self):
        super().__init__()
        self.model_function = NormalToHeteroMultiTaskCompleteModel

    def set_new_model_parameters(self, model, training_hyperparameters, model_hyperparameters,
                                 data, pre_processed_data, train_data, eval_data):
        pass
