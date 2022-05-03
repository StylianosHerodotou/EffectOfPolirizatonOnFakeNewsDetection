from Utilities.SignedGraphUtils import turn_data_to_positive_and_negative_edges
import torch
import os

from Utilities.HyperParameterTunning import run_hyper_parameter_tuning
from Utilities.JoinRawDatasetUtils import read_int_to_node_mapping, read_node_to_int_mapping

global temp_model
from Utilities.InitGlobalVariables import gpus_per_trial

from Utilities.InitGlobalVariables import device
from Utilities.InitGlobalVariables import dir_to_large


# device="cpu"


def get_best_model_embeddings_large(graph,hyperparameters,tuning_hyperparameters  ):
  best_model_config = run_hyper_parameter_tuning(hyperparameters, tuning_hyperparameters)

  best_trained_model = hyperparameters["model_function"](best_model_config)
  best_optimizer = torch.optim.Adam(best_trained_model.model.parameters(), lr=best_model_config["learning_rate"],
                                    weight_decay=best_model_config["weight_decay"])

  best_trained_model.optimizer=best_optimizer

  if torch.cuda.is_available():
      if gpus_per_trial > 1:
          best_trained_model.model = torch.nn.DataParallel(best_trained_model.model)
  # device="cpu"
  best_trained_model.model.to(device)

  x_features=graph.x
  edge_index= graph.edge_index
  edge_attr= graph.edge_attr
  positive_index, negative_index= turn_data_to_positive_and_negative_edges(graph.edge_index, graph.edge_attr)

  train_data={
      "GAT_features": x_features,
      "GAT_edge_index":edge_index,
      "GAT_edge_attr":edge_attr,
  }

  train_data["pos_index"]=positive_index
  train_data["neg_index"]=negative_index
  train_data["test_pos_index"]=positive_index
  train_data["test_neg_index"]=negative_index

  #only spectral information
  if(hyperparameters["spectral_features_type"]==0):
    train_data["SIGNED_features"]= best_trained_model.model.create_spectral_features(positive_index, negative_index)
  #only node features
  elif(hyperparameters["spectral_features_type"]==1):
    train_data["SIGNED_features"]= x_features
  #using both spectral and node features
  else:
    spectral_features= temp_model.model.create_spectral_features(positive_index, negative_index)
    print("spectral_features size", spectral_features.size())
    x_features= torch.cat((x_features,spectral_features),dim=1)
    print("size of input", x_features.size(), "size of model input", hyperparameters["size_of_x_features"])
    train_data["SIGNED_features"]= x_features

  #swap these to make the training function work.
  train_data["epochs"]=hyperparameters["final_training_epochs"]

  best_trained_model.train_fold_large(train_data,in_hyper_parameter_search=False)

  embeddings= best_trained_model.forward(train_data)
  return embeddings

def get_nodes_wiki_id_using_path(dir_path):
  nodes = read_node_to_int_mapping(dir_path)
  nodes= set(nodes.keys())
  return nodes

def create_embs_dic(graph, emb_dic, map):
  to_write={}
  for emb_name in emb_dic.keys():
    to_write[emb_name]={}

  emb_index=0;
  for node in graph.nodes():
    key= map[node]
    for emb_name, emb_value in emb_dic.items():
      to_write[emb_name][key]= emb_dic[emb_name][emb_index].detach().numpy().tolist()
    emb_index+=1
  return to_write

import json

def write_emd_to_file(graph,emb_dic,  to_write_path=None):
  if(to_write_path==None):
    to_write_path=dir_to_large
  map= read_int_to_node_mapping(dir_to_large)
  to_write= create_embs_dic(graph, emb_dic, map)
  file_name=  os.path.join(to_write_path,"emb.json")
  f = open(file_name, "w")
  json.dump(to_write, f)
  f.close()