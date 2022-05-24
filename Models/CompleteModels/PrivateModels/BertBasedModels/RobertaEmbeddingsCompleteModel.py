from DatasetRepresentation.DataPreprocessing.BERTProcessing import roberta_column_name
from Models.CompleteModels.PrivateModels.AbstractCompletePrivateModel import AbstractCompletePrivateModel
import torch
import torch.nn.functional as F
from Models.NNModels.CombinationModels.PretrainedEmbeddingsModels.RobertaEmbeddingsModel import RobertaEmbeddingsModel


class RobertaEmbeddingsCompleteModel(AbstractCompletePrivateModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        model = RobertaEmbeddingsModel(model_hyperparameters["input_hyperparameters"],
                                       model_hyperparameters["decoder_hyperparameters"])
        self.model = model
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=model_hyperparameters["learning_rate"])
        self.optimizer = optimizer

        self.loss_function = F.nll_loss

    def forward(self, train_data):
        return self.model.forward(train_data)

    def generate_embeddings(self, train_data):
        batch_size = train_data.batch.max() + 1
        roberta_embeddings = train_data.extra_inputs[roberta_column_name]
        roberta_embeddings = roberta_embeddings.view(batch_size, -1)
        return roberta_embeddings

    def find_loss(self, output, train_data):
        # decoder_input, decoder_output = output
        return self.loss_function(output, train_data.y)

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

    def find_performance(self, output, data):
        prediction = output.max(dim=1)[1]
        prediction = prediction.detach().numpy().tolist()
        true_labels = data.y.detach().numpy().tolist()
        return prediction, true_labels
