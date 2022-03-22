from abc import ABC
import json
import pandas as pd

def turn_str_keys_to_int(dic):
    dic = {int(k): v for k, v in dic.items()}
    return dic


def get_nodes_wiki_id_using_mapping(int_to_node_mapping):
    nodes = set()
    for key, value in int_to_node_mapping.items():
        nodes.add(value)
    return nodes


def switch_values_and_keys(dic: dict):
    new_dict = {}
    for key, value in dic.items():
        new_dict[value] = key
    return new_dict


def copy_df(df):
    columns = df.columns.tolist()
    dic = {}
    for column in columns:
        dic[column] = []

    toAdd = None
    for index, row in df.iterrows():
        for column in columns:
            #two special cases:
            if column == "graph":
                try:
                    toAdd = row[column].copy()
                except:
                    toAdd = row[column].clone()
            elif column == "int_to_node_mapping":
                toAdd = row[column].copy()
            else:
                toAdd = row[column]

            dic[column].append(toAdd)

    to_return = df = pd.DataFrame.from_dict(dic, orient='index').transpose()
    return to_return


class BaseDataset(ABC):

    def __init__(self, df):
        self.df = df
        self.mapping_column_name = "int_to_node_mapping"
        self.graph_column_name = "graph"
        self.article_column_name="article"


    def turn_json_to_mapping(self,mapping_column_name):
        mappings = self.df[mapping_column_name].tolist()
        new_mappings = list()

        for mapping in mappings:
            mapping = json.loads(mapping)
            new_mappings.append(mapping)
        self.df[mapping_column_name] = new_mappings

    def turn_mapping_to_json(self):
        mappings = self.df[self.mapping_column_name].tolist()
        new_mappings = list()

        for mapping in mappings:
            mapping = json.dumps(mapping)
            new_mappings.append(mapping)
        self.df[self.mapping_column_name] = new_mappings

    def turn_mappings_to_int(self, in_json=True):
        mappings = self.df[self.mapping_column_name].tolist()
        new_mappings = list()

        for mapping in mappings:
            if in_json:
                mapping = json.loads(mapping)
            # turn keys back to int
            mapping = turn_str_keys_to_int(mapping)

            new_mappings.append(mapping)
        self.df[self.mapping_column_name] = new_mappings

    def add_node_to_int_columns(self):
        node_to_int_mappings = list();
        for index, row in self.df.iterrows():
            node_to_int_mappings.append(switch_values_and_keys(row[self.int_to_node_column]))
        self.df["node_to_int_mapping"] = node_to_int_mappings



    def add_roberta_features(self):
        pass


