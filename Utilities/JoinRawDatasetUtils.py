# -*- coding: utf-8 -*-
"""JoinRawDataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sPCzXt5euAzq3U-4DNN-A9TN91IRZaCQ
"""

# from google.colab import drive
# drive.mount("/content/drive",force_remount=True)

# !grep -r "[:20]" "/content/drive/MyDrive/Colab Notebooks"

# !unzip -q /content/drive/MyDrive/ThesisProject/fake_news_in_time/micro_sags.zip -d /content/drive/MyDrive/ThesisProject/fake_news_in_time/

import os.path as osp
from math import ceil

import pandas as pd
import pickle
# Import the NetworkX package
import networkx as nx
import os
import json

def read_file(dir_path:str, base_file_name:str):
  path= os.path.join(dir_path, base_file_name)
  to_return=None;
  with (open(path, "rb")) as openfile:
      try:
          to_return=pickle.load(openfile)
      except EOFError:
          print("could not load ", path)
          pass
  return to_return 

def read_graph_file(dir_path):
  base_file_name="sag.pckl"
  return read_file(dir_path,base_file_name )
def read_node_to_int_mapping(dir_path):
  base_file_name="node_to_int.pckl"
  return read_file(dir_path,base_file_name )
def read_int_to_node_mapping(dir_path):
  base_file_name="int_to_node.pckl"
  return read_file(dir_path,base_file_name )

# #count number of items
# count=0
# for first_add_dir in os.listdir(dir_to_small):
#   first_layer_path = os.path.join(dir_to_small, first_add_dir)
#   for second_add_dir in os.listdir(first_layer_path):
#       count=count+1
# print(count)

def join_dataset(dir_to_articles= "/content/drive/MyDrive/ThesisProject/fake_news_in_time/articles",
                 dir_to_small= "/content/drive/MyDrive/ThesisProject/fake_news_in_time/micro_sags_small",
                 dir_to_save="/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset",
                 save_file_name= "joined_dataset_no_preprosessing.csv",
                 overwrite=False
                 ):
    #initi the list of data
    graphs= list()
    labels= list()
    articles= list()
    paths = list()
    int_to_node_mappings=list()
    
    #used to get up to dte quickly. 
    found_first=True
    file_done_so_far=set()


    #init the file to be written
    path_to_save=os.path.join(dir_to_save, save_file_name)

    list_of_columns_names=["path","label", "graph", "article", "int_to_node_mapping"]

    if overwrite or not os.path.isfile(path_to_save):
      temp_df = pd.DataFrame(columns =list_of_columns_names)
      temp_df.to_csv(path_to_save, index=False)
      found_first=True
    else:
      found_first=False
      temp_df=pd.read_csv(path_to_save)
      file_done_so_far=set(temp_df["path"])


    count=0
    update_file_after=100

    for first_add_dir in os.listdir(dir_to_small):
      first_layer_path = os.path.join(dir_to_small, first_add_dir)
      for second_add_dir in os.listdir(first_layer_path):
          current_dir_path = os.path.join(first_layer_path, second_add_dir)
          if(found_first==False):
            if(current_dir_path in file_done_so_far):
              count+=1
              continue
            else:
              found_first=True

          path_to_article= os.path.join(dir_to_articles, first_add_dir)
          path_to_article= os.path.join(path_to_article, second_add_dir)
          path_to_article=path_to_article+".json"
          
          #get graph
          current_graph= read_graph_file(current_dir_path)
          #turn to dictionary. 
          current_graph=nx.to_dict_of_dicts(current_graph)
          current_graph=json.dumps(current_graph)
          graphs.append(current_graph)

          #get int to node mapping
          int_to_node_small = read_int_to_node_mapping(current_dir_path)
          int_to_node_small =json.dumps(int_to_node_small)
          int_to_node_mappings.append(int_to_node_small)



          f = open(path_to_article)
          data = json.load(f)
          articles.append( data["text"])
          #get label
          if data["label"]=="reliable":
            labels.append(1)
          else:
            labels.append(0)
          
          paths.append(current_dir_path)
          count+=1

          if count%update_file_after==0:
            temp_df = pd.DataFrame(list(zip( paths, labels, graphs, articles,int_to_node_mappings)), columns =list_of_columns_names)
            # append data frame to CSV file
            temp_df.to_csv(path_to_save, mode='a', index=False, header=False)
            graphs.clear()
            labels.clear()
            articles.clear()
            paths.clear()
            int_to_node_mappings.clear()
            print(count)

    temp_df = pd.DataFrame(list(zip( paths, labels, graphs, articles,int_to_node_mappings)), columns =list_of_columns_names)
    # append data frame to CSV file
    temp_df.to_csv(path_to_save, mode='a', index=False, header=False)
    graphs.clear()
    labels.clear()
    articles.clear()
    paths.clear()
    int_to_node_mappings.clear()
    print(count)

# dir_to_articles= "/content/drive/MyDrive/ThesisProject/fake_news_in_time/articles"
# dir_to_small= "/content/drive/MyDrive/ThesisProject/fake_news_in_time/micro_sags"
# dir_to_large="/content/drive/MyDrive/ThesisProject/fake_news_in_time/sag"
# dir_to_save="/content/drive/MyDrive/ThesisProject/fake_news_in_time/compact_dataset"
# save_file_name= "joined_dataset_no_preprosessing.csv"

# join_dataset(dir_to_articles=dir_to_articles,dir_to_small=dir_to_small,
#              dir_to_save=dir_to_save, save_file_name=save_file_name,overwrite=False)

# add_node_to_int_columns(no_pre_df)

# no_pre_df

# path_to_raw_file=os.path.join(dir_to_save, save_file_name)
# no_pre_df= pd.read_csv(path_to_raw_file)
# try:
#   turn_json_to_mapping(no_pre_df)
# except:
#   pass

# try: 
#   turn_mappings_to_int(no_pre_df, in_json=True)
# except:
#   turn_mappings_to_int(no_pre_df, in_json=False)
# add_node_to_int_columns(no_pre_df)
# no_pre_df
# no_pre_df.to_csv(path_to_raw_file, index=False)