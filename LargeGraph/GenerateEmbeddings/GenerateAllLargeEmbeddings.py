from DatasetRepresentation.Utils import make_networkx_to_pyg_graph
from LargeGraph.GenerateEmbeddings import GeneratePOLEEmbeddings
from LargeGraph.GenerateEmbeddings.GenerateGNNEmbeddings import get_best_model_embeddings_large, write_emd_to_file
from LargeGraph.Utils import add_centrality_node_features_to_graph_to_large, add_identidy_node_feature_to_graph_large
import copy

global temp_model


def generate_all_large_embeddings(large_graph, int_to_node_mapping, hyperparameters, model_hyperparameters, tuning_hyperparameters):
    # train embeddings on this large graph
    # --> find best hyperparameters for each model

    # add features if desired.
    if (hyperparameters["add_centrality_node_features_to_large"]):
        add_centrality_node_features_to_graph_to_large(large_graph)
    if (hyperparameters["add_self_identidy_to_large"]):
        add_identidy_node_feature_to_graph_large(large_graph)

    # turn graph to pyg format.
    graph = make_networkx_to_pyg_graph(large_graph)

    # set the appropriate data in the hyperparameter dicct to be used later.
    hyperparameters["graph"] = graph
    model_hyperparameters["size_of_x_features"] = graph.x.size()[1]

    # set the appropriate report columns and set them in the dict.
    reporter_parameter_columns = list(model_hyperparameters.keys())
    for item in reporter_parameter_columns:
        if (item.endswith("features")):
            reporter_parameter_columns.remove(item)

    if ("GAT_edge_attr" in reporter_parameter_columns):
        reporter_parameter_columns.remove("GAT_edge_attr")
    reporter_parameter_columns

    tuning_hyperparameters["reporter_parameter_columns"] = reporter_parameter_columns

    # merge them into one dict.
    hyperparameters.update(model_hyperparameters)

    # create three different dicts in order to make three different call.
    # Todo : maybe use parallel method to do the follwing parallel calls.
    signed_spectral_features_hyperparameters = copy.deepcopy(hyperparameters)
    signed_spectral_features_hyperparameters["spectral_features_type"] = 0

    signed_node_features_hyperparameters = copy.deepcopy(hyperparameters)
    signed_node_features_hyperparameters["spectral_features_type"] = 1

    signed_node_and_spectral_features_hyperparameters = copy.deepcopy(hyperparameters)
    signed_node_and_spectral_features_hyperparameters["spectral_features_type"] = 2

    global temp_model
    temp_dic = {"size_of_x_features": signed_node_and_spectral_features_hyperparameters["size_of_x_features"],
                "hidden_nodes": 2,
                "num_layers": 1,
                "lamb": 5
                }
    temp_model = signed_node_and_spectral_features_hyperparameters["model_function"](temp_dic)
    print(signed_node_and_spectral_features_hyperparameters["size_of_x_features"])
    signed_node_and_spectral_features_hyperparameters["size_of_x_features"] *= 2

    # --> train using best hyperparameters
    embeddings = {
        "signed_spectral_features": get_best_model_embeddings_large(graph, signed_spectral_features_hyperparameters,
                                                                    tuning_hyperparameters),
        # "signed_node_features":get_best_model_embeddings_large(graph,signed_node_features_hyperparameters,tuning_hyperparameters  ),
        # "signed_node_and_spectral_features":get_best_model_embeddings_large(graph,signed_node_and_spectral_features_hyperparameters,tuning_hyperparameters  )
    }

    # --> write the embeddings to the appropriate file
    write_emd_to_file(large_graph, emb_dic=embeddings)

    # --> write the POLE embeddings into the embeddings file created above.
    GeneratePOLEEmbeddings.write_POLE_embeddings(large_graph, int_to_node_mapping)