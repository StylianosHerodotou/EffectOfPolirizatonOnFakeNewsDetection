from Models.CompleteModels.PrivateModels.AbstractCompletePrivateModel import AbstractCompletePrivateModel
import torch
from Models.NNModels.CombinationModels.EncoderDecoderModels.HomogeneousGATEDMLPModel import HomogeneousGATEDMLPModel
import torch.nn.functional as F

class HomogeneousGATCompleteModel(AbstractCompletePrivateModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model=model = HomogeneousGATEDMLPModel(model_hyperparameters["encoder_hyperparameters"],
                                         model_hyperparameters["decoder_hyperparameters"])
        self.optimizer = dict()
        self.optimizer["encoder"]=torch.optim.Adam(self.model.encoder.parameters(),
                                     lr=model_hyperparameters["learning_rate"],
                                     weight_decay=model_hyperparameters["weight_decay"])
        self.optimizer["decoder"] = torch.optim.Adam(self.model.decoder.parameters(),
                                                lr=model_hyperparameters["learning_rate"],
                                                weight_decay=model_hyperparameters["weight_decay"])
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
        for optimizer_name, optimizer in self.optimizer.items():
            optimizer.zero_grad()

    def optimizer_step(self):
        for optimizer_name, optimizer in self.optimizer.items():
            optimizer.step()

    def set_model_parameters_to_training_mode(self):
        self.model.encoder.train()
        self.model.decoder.train()

    def set_model_parameters_to_test_mode(self):
        self.model.encoder.eval()
        self.model.decoder.eval()