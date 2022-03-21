import os
import json

from DatasetRepresentation.NetworkXRepresentation.NetworkXGraphProcessing import read_embedings_file

from Utilities.InitGlobalVariables import dir_to_large, dir_to_base

def create_POLE_input_file(graph, filename="somepath.txt"):
    s = ""
    for edge in graph.edges(data=True):
        s += str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]["weight"]) + "\n"
    # Writing to file
    with open(os.path.join(dir_to_large,filename), "w") as f:
        # Writing data to a file
        f.write(s)

def write_POLE_embeddings(graph, int_to_node_mapping,
                          input_filename="POLE_input_ile.txt",
                          path_to_POLE="/POLE/src/embedding.py"):
    create_POLE_input_file(graph, input_filename)
    path_to_POLE=os.path.join(dir_to_base, path_to_POLE)

    print(os.path.join(dir_to_large,input_filename))
    print(os.getcwd())
    command = f"python {path_to_POLE} --graph { os.path.join(dir_to_large,input_filename)} --embedding {os.path.join(dir_to_large,'POLE.emb')}"
    # os.system(command)

    stream = os.popen(command)
    output = stream.read()
    print(output)

    # read the POLE emmbeddings
    file1 = open(os.path.join(dir_to_large,'POLE.emb'), 'r')
    Lines = file1.readlines()
    file1.close()

    embedings = list()
    # Strips the newline character
    for line in Lines:
        current_emb = list()
        for single_emb in line.split(" "):
            current_emb.append(float(single_emb))
        embedings.append(current_emb)

    # print(len(embedings), len(embedings[0]))

    # match the right embeddings to the right node
    large_embedings = read_embedings_file(dir_to_large)
    POLE_dic = dict()

    #######
    emb_index = 0;
    for node in graph.nodes():
        key = int_to_node_mapping[node]
        POLE_dic[key] = embedings[emb_index]
        emb_index += 1
    #######

    # for node in range(len(embedings)):
    #   key= int_to_node_mapping[node]
    #   POLE_dic[key]=embedings[node]

    # put the POLE embeddings in the emb file
    large_embedings["POLE"] = POLE_dic
    file_name = os.path.join(dir_to_large, "emb.json")
    f = open(file_name, "w")
    json.dump(large_embedings, f)
    f.close()