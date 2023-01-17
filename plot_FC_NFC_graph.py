#%% This file just plots the graphs of an FC and NFC network for slides
#%% Import Libraries and functions
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from pca_data_functions import *
#%% Define Parameters:
N =  10 # Set No of Nodes
#%% Plot the graph NFC
p =  0.4 # Set Parameter for Erdos-Reyni Convergence
A = gen_graph_adjacency(N,p) # Generate Adjacency Matrix of a connected graph (Erdos-Renyi Method)
rows, cols = np.where(A == 1) # find edges from Adjacency matrix
edges = zip(rows.tolist(), cols.tolist()) # convert rows and cols (edges) to list
G = nx.Graph()
G.add_edges_from(edges)
labelmap = dict(zip(G.nodes(), range(1,N+1))) # add labels to graph
nx.draw(G, labels=labelmap, with_labels=True)
plt.savefig('images/NFC_Graph.eps')
plt.show()
#%% Plot the graph FC
p =  1
A = gen_graph_adjacency(N,p) # Generate Adjacency Matrix of a connected graph (Erdos-Renyi Method)
rows, cols = np.where(A == 1) # find edges from Adjacency matrix
edges = zip(rows.tolist(), cols.tolist()) # convert rows and cols (edges) to list
G = nx.Graph()
G.add_edges_from(edges)
labelmap = dict(zip(G.nodes(), range(1,N+1))) # add labels to graph
nx.draw(G, labels=labelmap, with_labels=True)
plt.savefig('images/FC_Graph.eps')
plt.show()



