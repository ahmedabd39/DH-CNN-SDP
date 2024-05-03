import pandas as pd 
import os
import re
import warnings
import javalang

import networkx as nx
import matplotlib.pyplot as plt

from PARRR import PythonParser
from node2vec import Node2Vec


warnings.filterwarnings('ignore')
data = pd.read_csv("ant_1.7_with_nods.csv")
code = " "
file_paths = data['name']

cfg_nodes = []

j=1
G = nx.DiGraph()
EMBEDDING_FILENAME = './ant_1.7_node2vec_embeddings.txt'

for i in range(0, len(file_paths)):
	file_name = "cleaned_dataset/"+file_paths[i]+".java"
	#EMBEDDING_FILENAME = file_paths[i] + ".txt"
	#print(file_name)

	graph_obj = PythonParser(file_name).construct_graph()
	#G = nx.DiGraph()
	G.add_nodes_from(graph_obj.nodes)
	G.add_edges_from(graph_obj.edges)
	print(graph_obj.nodes)
	print(len(graph_obj.nodes), len(graph_obj.edges))
	print(graph_obj.edges)

	cfg_nodes.append(graph_obj.nodes)
	#nx.draw(G, with_labels=True)
	# plt.show()
	#CFG = CFG+G

df = pd.DataFrame(cfg_nodes)
data['cfgnodes']=cfg_nodes
# df.to_csv("declaration2.csv",index = False)
# data.to_csv("with_nods.csv",index = False)
data.to_csv("ant_1.7_with_cfgnods.csv",index = False)
print("saved")
#nx.draw(G, with_labels=True)
#plt.show()
node2vec = Node2Vec(G, dimensions=100, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

	# Look for most similar nodes
	#model.wv.most_similar('2')  # Output node names are always strings

	# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)