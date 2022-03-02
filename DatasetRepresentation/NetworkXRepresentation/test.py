from DatasetRepresentation.NetworkXRepresentation.NetworkXDataset import read_networkx_dataset

dataset= read_networkx_dataset(dir_to_dataset=r"C:\Users\35796\Desktop\university\Iliko mathimaton\4 Etos\Thesis\CodePart", name= r"joined_dataset_no_preprosessing.csv")
print(dataset.df.shape)
dataset.remove_empty_graphs()
print(dataset.df.shape)
dataset.add_centrality_node_features_to_df()