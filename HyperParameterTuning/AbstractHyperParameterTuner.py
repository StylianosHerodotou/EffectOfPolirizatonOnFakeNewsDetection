from abc import ABC, abstractmethod
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch
from Utilities.InitGlobalVariables import dir_to_ray_results

class AbstractHyperParameterTuner(ABC):
    def __init__(self, tuning_hyperparameters):
        self.scheduler = ASHAScheduler(metric=tuning_hyperparameters["asha_metric"],
                                        mode=tuning_hyperparameters["asha_mode"],
                                        max_t=tuning_hyperparameters["max_num_epochs"],
                                        grace_period=tuning_hyperparameters["be_nice_until"],
                                        reduction_factor=tuning_hyperparameters["reduction_factor"])

        self.reporter = CLIReporter(parameter_columns=tuning_hyperparameters["reporter_parameter_columns"],
                                    metric_columns=tuning_hyperparameters["reporter_metric_columns"])
        self.num_samples = tuning_hyperparameters["num_samples"]
        self.metric = tuning_hyperparameters["asha_metric"]
        self.mode = tuning_hyperparameters["asha_mode"]
        self.trainer = None

    def tuning_function(self, config):
        trainer = self.trainer(config["number_of_splits"])
        training_hyperparameters=config["training_hyperparameters"]
        graph_hyperparameters=config["graph_hyperparameters"]
        model_hyperparameters=config["model_hyperparameters"]
        data = config["data"]
        trainer.train(training_hyperparameters, graph_hyperparameters,  model_hyperparameters, data)

    def get_best_trial_configurations(self, config, print_results=False):
        result = ray.tune.run(
            partial(self.tuning_function),
            # resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            progress_reporter=self.reporter,
            local_dir=dir_to_ray_results
        )

        best_trial = result.get_best_trial(self.metric, self.mode,"last")
        if print_results:
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation f1: {}".format(
                best_trial.last_result[self.metric]))

        # return the best model config:
        return best_trial.config


