hetero_to_homo_name_mapping = {"x_dict": "x", "edge_index_dict": "edge_index",
          "edge_attr_dict": "edge_attr"}
def add_homo_attributes(useful_data):

    for hetero_value_name, homo_value_name in hetero_to_homo_name_mapping.items():
        if hasattr(useful_data, hetero_value_name):
            useful_data[homo_value_name] = useful_data[hetero_value_name]


def remove_homo_attributes(useful_data):

    for hetero_value_name, homo_value_name in hetero_to_homo_name_mapping.items():
        if hasattr(useful_data, hetero_value_name):
            del useful_data[homo_value_name]

    return useful_data