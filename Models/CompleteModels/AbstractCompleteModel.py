import torch
from abc import ABC, abstractmethod
import ray


class AbstractCompleteModel(ABC):
    def __init__(self):
        self.model = None
        self.optimizer = None

    @abstractmethod
    def forward(self, train_dic):
        pass

    @abstractmethod
    def generate_embeddings(self, train_dic):
        pass

    @abstractmethod
    def find_loss(self, output, train_dic):
        pass

    @abstractmethod
    def find_performance(self, output, test_dic):
        pass

    @abstractmethod
    def loss_backward(self, loss):
        pass

    @abstractmethod
    def zero_grad_optimizer(self):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    @abstractmethod
    def set_model_parameters_to_training_mode(self):
        pass

    @abstractmethod
    def set_model_parameters_to_test_mode(self):
        pass

    @abstractmethod
    def train_step(self, train_dic):
        pass

    @abstractmethod
    def test(self, test_dic):
        pass

    @abstractmethod
    def init_performance_metric(self):
        pass

    @abstractmethod
    def get_best_performance_metric_so_far(self, current_performance_metric, new_performance):
        pass

    @abstractmethod
    def loss_to_string(self, loss):
        pass

    @abstractmethod
    def best_performance_metric_to_string(self, performance_metric):
        pass

    @abstractmethod
    def performance_string(self, performance):
        pass

    @abstractmethod
    def get_report_score(self, performance):
        pass

    def train_fold(self, training_hyperparameters, train_data, eval_data, fold_number,
                   in_hyper_parameter_search):

        performance_metric = self.init_performance_metric()
        for epoch in range(training_hyperparameters["epochs"]):

            loss = self.train_step(train_data)
            if "print_every" in training_hyperparameters.keys() and epoch % training_hyperparameters[
                "print_every"] == 0:
                performance_test = self.test(eval_data)
                performance_train = self.test(train_data)
                performance_metric = self.get_best_performance_metric_so_far(performance_metric, performance_test)

                print(f'Fold number: {fold_number:02d}, Epoch: {epoch:03d}, Loss: {self.loss_to_string(loss)}\n'
                      f'Train performance: {self.performance_string(performance_train)}\n'
                      f'Test performance: {self.performance_string(performance_test)}\n'
                      f'best metric so far: {self.best_performance_metric_to_string(performance_metric)}')

                if in_hyper_parameter_search:
                    # with ray.tune.checkpoint_dir(os.path.join(dir_to_ray_checkpoints, str((fold_number * train_dic["epochs"]) + epoch))) as checkpoint_dir:
                    #     path = os.path.join(checkpoint_dir, "checkpoint")
                    #     torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
                    ray.tune.report(score=self.get_report_score(performance_test))
