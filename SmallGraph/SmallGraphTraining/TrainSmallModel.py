from DatasetRepresentation.DataPreprocessing.CombinationOfProsessing import create_training_set
from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import read_embedings_file
from DatasetRepresentation.Utils import make_networkx_to_pyg_df
from sklearn.model_selection import train_test_split

from Utilities.InitGlobalVariables import  dir_to_large

from SmallGraph.SmallGraphTraining.HyperParameterTunningSmall import train_and_write_best_model
from Utilities.HyperParameterTunning import run_hyper_parameter_tuning


def generate_and_save_results_for_small_models(networkx_dataset, hyperparameters, model_hyperparameters, tuning_hyperparameters):
    # create graph that contains node embeddings from the above.
    large_embedings = read_embedings_file(dir_to_large)

    networkx_dataset.add_large_node_embeddings_to_df(large_embedings,large_emb_names=hyperparameters["large_emb_names"])

    # turn graphs to pyg
    pyg_dataset= make_networkx_to_pyg_df(networkx_dataset)

    train_df, test_df = train_test_split(pyg_dataset.df, test_size=1 - hyperparameters["train_size"], random_state=42,
                                         shuffle=True, stratify=None)
    # eval_df, test_df= train_test_split(test_df, test_size=eval_test_split, random_state=42, shuffle=True, stratify=None)
    train_set = create_training_set(train_df, hyperparameters, False)
    # eval_set=create_training_set(eval_df,hyperparameters,add_lstm_representation)
    test_set = create_training_set(test_df, hyperparameters, False)

    hyperparameters["train_set"] = train_set

    tuning_hyperparameters["reporter_parameter_columns"] = list(model_hyperparameters.keys())

    hyperparameters.update(model_hyperparameters)

    for time in range(hyperparameters["number_of_overal_models"]):
        best_config = run_hyper_parameter_tuning(hyperparameters, tuning_hyperparameters)
        train_and_write_best_model(best_config, train_set, hyperparameters)