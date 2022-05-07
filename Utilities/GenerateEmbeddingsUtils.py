import json
import os

from DatasetRepresentation.BaseDataset import switch_values_and_keys
from Utilities.InitGlobalVariables import dir_to_large
from Utilities.JoinRawDatasetUtils import read_node_to_int_mapping, read_int_to_node_mapping

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


def write_emd_to_file(graph,emb_dic,  to_write_path=None, overwrite_prev_embeddings=False):
  if(to_write_path==None):
    to_write_path=dir_to_large
  map= read_int_to_node_mapping(dir_to_large)
  to_write= create_embs_dic(graph, emb_dic, map)
  file_name=  os.path.join(to_write_path,"emb.json")
  if overwrite_prev_embeddings:
    f = open(file_name, "w")
  else:
    f = open(file_name, "a")
  json.dump(to_write, f)
  f.close()

def removeNodesFromGraph(setOfNodeNamesToRemove, graph, int_to_node_mapping):
    node_to_int_mapping = switch_values_and_keys(int_to_node_mapping)
    setOfNodeIdsToRemove = set();

    for node_name in setOfNodeNamesToRemove:
      node_id = node_to_int_mapping[node_name]
      setOfNodeIdsToRemove.add(node_id)
    graph.remove_nodes_from(setOfNodeIdsToRemove)