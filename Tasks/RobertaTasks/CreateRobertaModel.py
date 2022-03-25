from DatasetRepresentation.NetworkXRepresentation.NetworkXDataset import read_networkx_dataset, NetworkXDataset
from Utilities.JoinRawDatasetUtils import read_int_to_node_mapping
from Utilities.InitGlobalVariables import dir_to_large
from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph
from DatasetRepresentation.Utils import make_networkx_to_pyg_df

dir_to_dataset = "/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/Datasets"
base_filename = "joined_dataset_no_empty_graphs_old.csv"

dataset = read_networkx_dataset(dir_to_dataset=dir_to_dataset, name=base_filename)
# show how many items of each "score" class there are in the dataset.
print(min(dataset.df.groupby('label').size()))
dataset.df = dataset.df.groupby('label').apply(
    lambda x: x.sample(n=min(dataset.df.groupby('label').size()))).reset_index(
    drop=True)

from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import add_centrality_node_features_to_graph
dataset.add_centrality_node_features_to_df(features=["degree_centrality"])
from DatasetRepresentation.Utils import make_networkx_to_pyg_df
# turn graphs to pyg
pyg_dataset = make_networkx_to_pyg_df(dataset)
from transformers import RobertaTokenizer
import torch
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

df=pyg_dataset.df[:100]
df.columns
roberta_input= []
for sentense in df["article"].values:
    current = tokenizer.encode_plus(
                sentense,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                truncation=True,
                max_length = 10,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
           )
    roberta_input.append(current)

from SmallGraph.SmallGraphTraining.SmallModels.BERTBasedModels.RobertaModel import Roberta
model= Roberta()

train_set= df["graph"].tolist()
y_train = df["label"].tolist()
y_train = torch.LongTensor(y_train)
y_train = y_train.reshape(y_train.size(0), 1)

for index in range(len(train_set)):
    train_set[index].y=y_train[index]
    train_set[index].edge_index=train_set[index].edge_index.long()
    train_set[index].robert_rep=roberta_input[index]

model.optimizer= torch.optim.Adam(model.model.parameters(), lr=0.0005)
from torch_geometric.loader import DataLoader

loader=DataLoader(train_set, batch_size=32)
for i,data in enumerate(loader):
    print(i)
    output= model.forward(data)
    loss=model.find_loss(output, data)
    loss.backward()