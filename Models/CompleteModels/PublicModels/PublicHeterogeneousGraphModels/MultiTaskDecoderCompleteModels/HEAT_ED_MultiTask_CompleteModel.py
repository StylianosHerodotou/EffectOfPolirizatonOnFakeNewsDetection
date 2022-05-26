import torch
from Models.CompleteModels.PublicModels.PublicHeterogeneousGraphModels.MultiTaskDecoderCompleteModels.AbstractMultiTaskDecoderCompleteModel import \
    AbstractMultiTaskDecoderCompleteModel
from Models.NNModels.CombinationModels.EncoderDecoderModels.PublicGNNs.HEAT_ED_MultiTask_NNModel import HEAT_ED_MultiTask_NNModel


class HEAT_ED_MultiTask_CompleteModel(AbstractMultiTaskDecoderCompleteModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        model = HEAT_ED_MultiTask_NNModel(model_hyperparameters["encoder_hyperparameters"],
                                         model_hyperparameters["decoder_hyperparameters"])
        self.model = model

        self.optimizer = self.model.decoder.get_optimizers()
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(),
                                             lr=model_hyperparameters["encoder_hyperparameters"]["learning_rate"])
        self.optimizer["encoder_optimizer"] = encoder_optimizer
