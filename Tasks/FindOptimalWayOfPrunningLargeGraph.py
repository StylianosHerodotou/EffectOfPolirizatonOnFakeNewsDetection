from DatasetRepresentation.BaseDataset import copy_df
from DatasetRepresentation.NetworkXRepresentation.NetworkXDataset import read_networkx_dataset, NetworkXDataset
from LargeGraph.GenerateEmbeddings.GenerateAllLargeEmbeddings import generate_all_large_embeddings
from LargeGraph.LargeGraphTraining.HyperParameterTunningLarge import hyper_parameter_tuning_large
from LargeGraph.LargeGraphTraining.LargeModels.SignedGCN import SignedGCNModel
from SmallGraph.SmallGraphTraining.SmallModels.GAT import GAT
from SmallGraph.SmallGraphTraining.TrainSmallModel import generate_and_save_results_for_small_models
from Utilities.JoinRawDatasetUtils import read_int_to_node_mapping, read_graph_file
import networkx as nx
from ray import tune
import torch



from Utilities.InitGlobalVariables import device
from Utilities.InitGlobalVariables import gpus_per_trial
from Utilities.InitGlobalVariables import dir_to_large


def remove_nodes_based_on_frequency(large_graph,int_to_node_mapping_lagre,threshold):
  to_remove_ids=set()
  to_remove_names=set()
  for node in large_graph.nodes(data=True):
    if(node[1]["coverage"]<threshold):
      to_remove_names.add(int_to_node_mapping_lagre[node[0]])
      to_remove_ids.add(node[0])

  print(len(to_remove_ids))
  print(large_graph.number_of_nodes())
  large_graph.remove_nodes_from(to_remove_ids)
  large_graph.remove_nodes_from(list(nx.isolates(large_graph)))
  print(large_graph.number_of_nodes())
  return to_remove_names

def findOptimalWayOfPrunningLargeOne():
    # large one one time stuff
    # dir_to_large = "/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/Datasets/sag"

    base_filename = "joined_dataset_no_empty_graphs.csv"
    base_newtorkx_dataset = read_networkx_dataset(base_filename)
    int_to_node_mapping_lagre = read_int_to_node_mapping(dir_to_large)

    # #small one, one time stuff
    # filename= "joined_centrality.csv"
    # base_df=CutDownLargeOne.read_networkx_dataset(filename,dir_to_dataset="/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset")
    # show how many items of each "score" class there are in the dataset.
    print(min(base_newtorkx_dataset.df.groupby('label').size()))
    base_newtorkx_dataset.df = base_newtorkx_dataset.df.groupby('label').apply(lambda x: x.sample(n=min(base_newtorkx_dataset.df.groupby('label').size()))).reset_index(
        drop=True)


    #######these are the hyper for the large one:
    hyperparameters_large = {
        "num_classes": 2,
        "number_of_splits": 4,
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": tune.loguniform(1e-6, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "spectral_features_type": 0,
        # "add_centrality_node_features_to_large":tune.choice([True,False]),
        "add_centrality_node_features_to_large": tune.choice([False]),
        "add_self_identidy_to_large": tune.choice([True, False]),
        "final_training_epochs": 100,
        "tunning_function":hyper_parameter_tuning_large
    }

    model_hyperparameters_large = {
        "model_function": SignedGCNModel,
        # "size_of_x_features": graph.x.size()[1],
        "hidden_nodes": tune.choice([64, 128, 172, 256]),
        "num_layers": tune.choice([1, 2, 3]),
        "lamb": 5,
        "bias": True
    }

    asha_metric_large = "f1"
    tuning_hyperparameters_large = {
        "asha_metric": asha_metric_large,
        "asha_mode": "max",
        "max_num_epochs": 20,
        "be_nice_until": 10,
        "reduction_factor": 2,
        # "reporter_parameter_columns":reporter_parameter_columns,
        "reporter_metric_columns": [asha_metric_large, "training_iteration"],
        "num_samples": 2,

    }
    # global device
    # device = "cpu"
    #
    # global gpus_per_trial
    # gpus_per_trial = 0
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     gpus_per_trial = 1

    hyperparameters_small = {
        "num_classes": 2,
        "number_of_splits": 4,
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": tune.loguniform(1e-6, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        # "eval_set":eval_set,
        # "large_emb_names":['signed_spectral_features', 'signed_node_features', 'signed_node_and_spectral_features', 'POLE'],
        "large_emb_names": ['signed_spectral_features', 'POLE'],

        "train_size": 0.8,
        "number_of_overal_models": 10
    }

    model_hyperparameters_small = {
        'model_function': GAT,
        "hidden_size": tune.choice([64, 128, 172, 256]),
        "heads": tune.choice([1, 2, 4, 8, 16]),
        "dropout": tune.uniform(0.01, 0.4),
        "pooling_ratio": 0.8,
        "num_layers": tune.choice([1, 2]),
    }

    asha_metric_small = "accuracy"
    tuning_hyperparameters_small = {
        "asha_metric": asha_metric_small,
        "asha_mode": "max",
        "max_num_epochs": 20,
        "be_nice_until": 10,
        "reduction_factor": 2,
        "reporter_metric_columns": [asha_metric_small, "training_iteration"],
        "num_samples": 1,

    }

    # caverage_values_to_test=[5,10,15,20,25]
    # caverage_values_to_test=[0, 5,7,10,15,20,25,30,35,40,45,50,55,60,65,70]
    caverage_values_to_test = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    for threshold in caverage_values_to_test:
        # cut down large graph
        # aka remove nodes with less than threshold
        large_graph = read_graph_file(dir_to_large)
        to_remove_names = remove_nodes_based_on_frequency(large_graph, threshold)

        # train embeddings on this large graph
        # --> find best hyperparameters for each model
        # --> train using best hyperparameters
        # --> write the embeddings to the appropriate file
        generate_all_large_embeddings(large_graph, int_to_node_mapping_lagre,
                      hyperparameters=hyperparameters_large,
                      model_hyperparameters=model_hyperparameters_large
                      , tuning_hyperparameters=tuning_hyperparameters_large)

        current_df = copy_df(base_newtorkx_dataset.df)
        current_networkx_dataset = NetworkXDataset(current_df)
        current_networkx_dataset.remove_nodes_not_in_large(to_remove_names)
        generate_and_save_results_for_small_models(current_networkx_dataset,
                      hyperparameters=hyperparameters_small,
                      model_hyperparameters=model_hyperparameters_small,
                      tuning_hyperparameters=tuning_hyperparameters_small)
