from Models.CompleteModels.PublicModels.AbstractCompletePublicModel import AbstractCompletePublicModel
import torch
from Models.NNModels.CombinationModels.EncoderDecoderModels.NormalToHeteroEDMultiTaskNNModel import \
    NormalToHeteroEDMultiTaskNNModel


class NormalToHeteroMultiTaskCompleteModel(AbstractCompletePublicModel):
    def __init__(self, model_hyperparameters):
        super().__init__()
        model = NormalToHeteroEDMultiTaskNNModel(model_hyperparameters["encoder_hyperparameters"],
                                                 model_hyperparameters["decoder_hyperparameters"])
        self.model = model

        self.optimizer = self.model.decoder.get_optimizers()
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(),
                                             lr=model_hyperparameters["encoder_hyperparameters"]["learning_rate"],
                                             weight_decay=model_hyperparameters["encoder_hyperparameters"][
                                                 "weight_decay"])
        self.optimizer["encoder_optimizer"] = encoder_optimizer

    def forward(self, train_data):
        return self.model.forward(train_data)

    def generate_embeddings(self, train_data):
        encoder_output, decoder_output = self.forward(train_data)
        return encoder_output

    def find_loss(self, output, train_data, regularize_loss=True):
        encoder_output, decoder_output = output
        task_decoders = self.model.decoder.task_decoders
        loss_dict = dict()

        for task_name, current_output in decoder_output.items():
            current_decoder = task_decoders[task_name]
            if current_decoder.loss_arguments is not None:
                loss = current_decoder.performance_tracker.loss_function(current_output, train_data,
                                                                         current_decoder.loss_arguments["edge_type"],
                                                                         current_decoder.loss_arguments["feature_name"])
            else:
                loss = current_decoder.performance_tracker.loss_function(current_output, train_data)
            loss_dict[task_name] = loss

        return loss_dict

    def find_performance(self, output, test_data):
        encoder_output, decoder_output = output
        task_decoders = self.model.decoder.task_decoders
        performance_dict = dict()

        for task_name, current_output in decoder_output.items():
            current_decoder = task_decoders[task_name]
            if current_decoder.loss_arguments is not None:
                metric = current_decoder.performance_tracker.metric_function(current_output, test_data,
                                                                     current_decoder.loss_arguments["edge_type"],
                                                                     current_decoder.loss_arguments["feature_name"])
            else:
                metric = current_decoder.performance_tracker.metric_function(current_output, test_data)
            performance_dict[task_name] = metric
        return performance_dict

    def loss_backward(self, loss_dict):
        for i, (task_name, loss) in enumerate(loss_dict.items()):
            if i != len(loss_dict) - 1:
                loss_dict[task_name].backward(retain_graph=True)
            else:
                loss_dict[task_name].backward(retain_graph=False)

    def zero_grad_optimizer(self):
        for key, optimizer in self.optimizer.items():
            optimizer.zero_grad()

    def optimizer_step(self):
        for key, optimizer in self.optimizer.items():
            optimizer.step()

    def set_model_parameters_to_training_mode(self):
        self.model.encoder.train()
        self.model.decoder.train()

    def set_model_parameters_to_test_mode(self):
        self.model.encoder.eval()
        self.model.decoder.eval()

    # the below are for recording purposes.
    def init_performance_metric(self):
        initial_performance_metric = dict()
        for task_name in self.model.decoder.task_decoders.keys():
            initial_performance_metric[task_name] = 0
        return initial_performance_metric

    def get_best_performance_metric_so_far(self, current_performance_metric, new_performance):
        new_auc, new_f1 = new_performance
        current_performance_metric["f1"] = max(new_f1, current_performance_metric["f1"])
        current_performance_metric["f1"] = max(new_auc, current_performance_metric["f1"])
        return current_performance_metric

    def loss_to_string(self, loss):
        return str(loss.item())

    def performance_metric_to_string(self, performance_metric):
        string = ""
        for key, value in performance_metric.items():
            string += key + ": " + "{:.2f}".format(value) + "\n"
        return string

    def performance_string(self, performance):
        auc, f1 = performance
        return "auc: " + "{:.2f}".format(auc) + " f1 score: " + "{:.2f}".format(f1)

    def get_report_score(self, performance):
        auc, f1 = performance
        return f1
