from DatasetRepresentation.NetworkXRepresentation.NetworkXDataset import read_networkx_dataset
from SmallGraph.SmallGraphTraining.SmallModels.RNNModels.LSTMBagOfWords import LSTMBagOfWords
from SmallGraph.SmallGraphTraining.TrainSmallModel import generate_and_save_results_for_small_models


def trainRNN(dir_to_dataset = "/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/Datasets",
             base_filename = "joined_dataset_no_empty_graphs.csv"):

    dataset = read_networkx_dataset(dir_to_dataset=dir_to_dataset, name=base_filename)
    # show how many items of each "score" class there are in the dataset.
    print(min(dataset.df.groupby('label').size()))
    dataset.df = dataset.df.groupby('label').apply(
        lambda x: x.sample(n=min(dataset.df.groupby('label').size()))).reset_index(
        drop=True)

    hyperparameters_small = {
        "num_classes": 2,
        #     "number_of_splits": 2,
        "number_of_splits": 4,
        "batch_size": 32,
        "epochs": 50,
        #     "epochs": 1,
        "learning_rate": tune.loguniform(1e-6, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "train_size": 0.8,
        "number_of_overal_models": 2
        #     "number_of_overal_models": 1
        "large_emb_names":[],
        "input_type": ["bag_of_words"]
    }
    mlp_nodes_per_layer=[]
    for i in range(tune.choice([0,1,2])):
        mlp_nodes_per_layer.append(tune.choice([32,64,128,256,512]))

    MLP_arguments = {
        "nodes_per_hidden_layer": mlp_nodes_per_layer,
        "dropout": tune.uniform(0.01, 0.4)
    }

    model_hyperparameters_small = dict(model_function=LSTMBagOfWords,
                                       embedding_dim=tune.choice([32, 64, 128, 256, 512]),
                                       hidden_dim=tune.choice([32, 64, 128, 256, 512]),
                                       vocab_size=tune.choice([2500, 5000, 7500, 10000]), output_size=2,
                                       num_layers=tune.choice([1, 2])) = MLP_arguments

    }

    asha_metric_small = "accuracy"
    tuning_hyperparameters_small = {
        "asha_metric": asha_metric_small,
        "asha_mode": "max",
        "max_num_epochs": 20,
        #     "max_num_epochs": 1,

        "be_nice_until": 10,
        #     "be_nice_until": 1,
        "reduction_factor": 2,
        "reporter_metric_columns": [asha_metric_small, "training_iteration"],
        "num_samples": 2,
        "tunning_function": hyper_parameter_tuning_small
    }

    generate_and_save_results_for_small_models(dataset,
                                               hyperparameters=hyperparameters_small,
                                               model_hyperparameters=model_hyperparameters_small,
                                               tuning_hyperparameters=tuning_hyperparameters_small)
