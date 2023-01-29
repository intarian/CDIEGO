#%% This file is an extension to original 3a.effect of different N on sythetic
# data under different topologies. We generate topologies here for different N using
# Erdos-Reyni graph and save these topologies in sim_data folder. The adjacency matrix is
# then used by W_gen_M function to generate weight matrix. The original 3a experiment
# runs 1 consensus rounds to see the effect of performance effect of different N.
# We generate 3 topologies for different values of N
#%% Import Libraries and functions
import numpy as np
from pca_data_functions import *
import multiprocessing as mp
import time
from algorithms import *
#%% Define Parameters:
param = np.zeros((4,1)) # Store the values of parameters
Na = param[0,0] = 10 # Set No of Nodes a
Nb = param[1,0] = 20 # Set No of Nodes b
Nc = param[2,0] = 30 # Set No of Nodes c
#%% Generate Erdos - Reyni Graph for different Nodes
p = param[3,0] = 0.1
A_Na = gen_graph_adjacency(Na, p)
A_Nb = gen_graph_adjacency(Nb, p)
A_Nc = gen_graph_adjacency(Nc, p)
## Save the data of arrays
np.save('sim_data/3a.topo_gen_Na.npy', A_Na)
np.save('sim_data/3a.topo_gen_Nb.npy', A_Nb)
np.save('sim_data/3a.topo_gen_Nc.npy', A_Nc)
#np.save('sim_data/3a.eff_N_synthetic_param_mp.npy', param)