from Models.CompleteModels.PublicModels.AbstractCompletePublicModel import AbstractCompletePublicModel
import torch

class AbstractMultiTaskDecoderCompleteModel(AbstractCompletePublicModel):
    def __init__(self):
        super().__init__()

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

            loss *= current_decoder.metric_weight
            loss_dict[task_name] = loss

        return loss_dict

    #this is
    def find_performance(self, output, test_data):
        encoder_output, decoder_output = output
        task_decoders = self.model.decoder.task_decoders
        performance_dict = dict()

        for task_name, current_output in decoder_output.items():
            current_decoder = task_decoders[task_name]
            if current_decoder.loss_arguments is not None:
                metric = current_decoder.performance_tracker.metric_function(current_output, test_data,
                                                                             current_decoder.loss_arguments[
                                                                                 "edge_type"],
                                                                             current_decoder.loss_arguments[
                                                                                 "feature_name"])
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

    #for each
    def get_best_performance_metric_so_far(self, current_performance_metric_dict, new_performance_doct):
        best_performance = {}
        for task_name, new_performance in new_performance_doct.items():
            current_performance_tracker = self.model.decoder.task_decoders[task_name].performance_tracker
            old_performance = current_performance_metric_dict[task_name]
            best_performance[task_name] = current_performance_tracker.desired_metric_function(new_performance,
                                                                                              old_performance)
        return best_performance

    def loss_to_string(self, loss_dict):
        to_return = ""
        for task_name, current_task_loss in loss_dict.items():
            to_return += str(task_name) + ": " + str("{:.2f}".format(current_task_loss.item())) + " ,"
        return to_return

    def best_performance_metric_to_string(self, performance_metric_dict):
        best_performance_metric_string = str("{:.2f}".format(self.get_report_score(performance_metric_dict)))
        return best_performance_metric_string

    def performance_string(self, performance_metric_dict):
        to_return = ""
        for task_name, current_task_metric in performance_metric_dict.items():
            to_return += str(task_name) + ": " + str("{:.2f}".format(current_task_metric)) + " ,"
        return to_return

    def get_report_score(self, performance_metric_dict):
        combined_performance = 0
        for task_name, task_decoder in self.model.decoder.task_decoders.items():
            combined_performance += performance_metric_dict[task_name] * task_decoder.metric_weight
        return combined_performance.item()
