def find_reverse_edge_types(pyg_data):
    reverse_edge_types=list()
    for edge_type in pyg_data.edge_types:
        possible_reverse_edge = (edge_type[2],edge_type[1],edge_type[0])
        if possible_reverse_edge in pyg_data.edge_types:
            reverse_edge_types.append(edge_type)
    return reverse_edge_types

