from Models.CompleteModels.PrivateModels.AbstractCompletePrivateModel import AbstractCompletePrivateModel
import torch
import torch.nn.functional as F

from Models.NNModels.CombinationModels.EncoderDecoderModels.PrivateGNNs.LSTMBagOfWordsEDMLPModel import LSTMBagOfWordsEDMLPModel


class LSTMBagOfWordsCompleteModel(AbstractCompletePrivateModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        model = LSTMBagOfWordsEDMLPModel(model_hyperparameters["encoder_hyperparameters"],
                                         model_hyperparameters["decoder_hyperparameters"])
        self.model = model
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=model_hyperparameters["learning_rate"])
        self.optimizer = optimizer

        self.loss_function = F.nll_loss

    def forward(self, train_data):
        return self.model.forward(train_data)

    def generate_embeddings(self, train_data):
        encoder_output, decoder_output = self.forward(train_data)
        return encoder_output

    def find_loss(self, output, train_data):
        encoder_output, decoder_output = output
        return self.loss_function(decoder_output, train_data.y)

    def loss_backward(self, loss):
        loss.backward()

    def zero_grad_optimizer(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def set_model_parameters_to_training_mode(self):
        self.model.train()

    def set_model_parameters_to_test_mode(self):
        self.model.eval()
