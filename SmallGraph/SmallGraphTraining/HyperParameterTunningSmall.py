from sklearn.model_selection import KFold
import os
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from Utilities.InitGlobalVariables import device
from Utilities.InitGlobalVariables import gpus_per_trial
from Utilities.InitGlobalVariables import dir_to_base


def k_fold_training_small(hyperparameters, train_set, in_hyper_parameter_search=True):
    sum_dict = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "fbeta_score": 0
    }

    splits = KFold(n_splits=hyperparameters["number_of_splits"], shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_set)))):
        print('Fold {}\n\n'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        eval_sampler = SubsetRandomSampler(val_idx)

        train_loader = {"size":len(train_sampler)}
        eval_loader = {"size": len(eval_sampler)}

        train_loader["loader"] = DataLoader(train_set, batch_size=hyperparameters["batch_size"], sampler=train_sampler)
        eval_loader ["loader"] = DataLoader(train_set, batch_size=hyperparameters["batch_size"], sampler=eval_sampler)

        model = hyperparameters["model_function"](hyperparameters)

        # define device:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.model = torch.nn.DataParallel(model.model)
        model.model.to(device)

        optimizer = hyperparameters["optimizer"](model.model.parameters(),lr=hyperparameters["learning_rate"])
        model.optimizer=optimizer

        results_dict = model.train_fold_small(train_loader, eval_loader, epochs=hyperparameters["epochs"],
                         in_hyper_parameter_search=in_hyper_parameter_search)

        for key, value in results_dict.items():
            sum_dict[key] += value

        for key, value in sum_dict.items():
            sum_dict[key] = value/hyperparameters["number_of_splits"]

        s=hyperparameters["large_emb_names"] + "\n"
        for key, value in sum_dict.items():
            s+= key +": "+ str(value) + ", "
        s+='\n'

        with open(os.path.join(dir_to_base, "best from kfold"), "a") as file_object:
            file_object.write(s)
def hyper_parameter_tuning_small(config, checkpoint_dir=None):
    k_fold_training_small(config, config["train_set"])

def train_and_write_best_model(best_config, train_set, hyperparameters,
                               path_to_save=None, name_of_file="sample.txt"):
        if(path_to_save==None):
            path_to_save=dir_to_base

        splits = KFold(n_splits=best_config["number_of_splits"], shuffle=True, random_state=42)
        sum_dict ={
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "fbeta_score": 0
        }
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_set)))):
            print('Fold {}\n\n'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            eval_sampler = SubsetRandomSampler(val_idx)

            train_loader = {"size": len(train_sampler)}
            eval_loader = {"size": len(eval_sampler)}

            train_loader["loader"] = DataLoader(train_set, batch_size=hyperparameters["batch_size"],
                                                sampler=train_sampler)
            eval_loader["loader"] = DataLoader(train_set, batch_size=hyperparameters["batch_size"],
                                               sampler=eval_sampler)

            best_trained_model = hyperparameters["model_function"](best_config)
            best_optimizer = hyperparameters["optimizer"](best_trained_model.model.parameters(), lr=best_config["learning_rate"])
            best_trained_model.optimizer=best_optimizer

            if torch.cuda.is_available():
                if gpus_per_trial > 1:
                    best_trained_model.model = torch.nn.DataParallel(best_trained_model,model)
            best_trained_model.model.to(device)

            best_trained_model.train_fold_small(train_loader, eval_loader,
                             epochs=hyperparameters["final_training_epochs"],
                             in_hyper_parameter_search=False)

            test_dict = best_trained_model.test_small(eval_loader)
            for key, value in test_dict.items():
                sum_dict[key]+=value

        for key, value in sum_dict.items():
            sum_dict[key] = value/best_config["number_of_splits"]

        s=hyperparameters["large_emb_names"] + "\n"
        for key, value in sum_dict.items():
            s+= key +": "+ str(value) + ", "
        s+='\n'

        with open(os.path.join(path_to_save, name_of_file), "a") as file_object:
            file_object.write(s)