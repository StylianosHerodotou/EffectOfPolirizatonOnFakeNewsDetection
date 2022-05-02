import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch

from Utilities.InitGlobalVariables import gpus_per_trial, dir_to_ray_results
from Utilities.InitGlobalVariables import device

def run_hyper_parameter_tuning(hyperparameters, tuning_hyperparameters):
    scheduler = ASHAScheduler(
        metric=tuning_hyperparameters["asha_metric"],
        mode=tuning_hyperparameters["asha_mode"],
        max_t=tuning_hyperparameters["max_num_epochs"],
        grace_period=tuning_hyperparameters["be_nice_until"],
        reduction_factor=tuning_hyperparameters["reduction_factor"])

    reporter = CLIReporter(
        parameter_columns=tuning_hyperparameters["reporter_parameter_columns"],
        metric_columns=tuning_hyperparameters["reporter_metric_columns"])

    result = ray.tune.run(
        partial(tuning_hyperparameters["tuning_function"]),
        # resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=hyperparameters,
        num_samples=tuning_hyperparameters["num_samples"],
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir =dir_to_ray_results
    )

    best_trial = result.get_best_trial(tuning_hyperparameters["asha_metric"], tuning_hyperparameters["asha_mode"],
                                       "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation f1: {}".format(
    #     best_trial.last_result[tuning_hyperparameters["asha_metric"]]))

    # return the best model config:
    return best_trial.config

