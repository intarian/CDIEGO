#%% This file uses multi-processing capabilities to run the code on multiple cores.
# Only Monte-Carlo simulations are executed on multiple cores
# We use only single core in this scratch code
#%% This file computes the effects of convergence under different topologies:
# 1. Ring Graph  (Not implemented yet)
# 2. FC graph with one drop node
# The convergence is tested under two scenario (Tc = 1. and Tc = optimal)
# under Synthetic data.
# The file just runs the algorithm and save the data in sim_data folder.
#%% Import Libraries and functions
import numpy as np
from pca_data_functions import *
import multiprocessing as mp
import time
from algorithms import *
#%% Define Parameters:
param = np.zeros((7,1)) # Store the values of parameters
d = param[0,0] = 20 # Set dimensionality of data
tot_iter = param[1,0] = 50*1000 # Set no. of iterations
monte_carlo = param[2,0] = 50 # Set no. of monte carlo runs
N = param[3,0] = 40 # Set no of nodes
step_size = param[4,0] = 0.5 # Set step size
#%% Set Eigengap
eig_gap_fac = 2 #Controls the factor of eigengap.
siga = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac) + 0.1
eigen_gap = param[5,0] = np.round(siga[0] - siga[1], 3) # Set Eigengap
#%% Initialize empty array to store error for monte carlo simulations
# Save monte-carlo runs for C-DIEGO algorithm under different setting for each N
cdiego_m_fgd = np.zeros((2,monte_carlo,tot_iter)) # 2 cases: Tc=1 and Tc = opt
#%% Generate Covariance Matrix and ground truth eigenvector
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
#%% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate a one connection drop graph using Adjacency matrix
import networkx as nx
from matplotlib import pyplot as plt
A = np.ones((N,N)) - np.eye(N)
## Drop the edge between node 1 and node N
A[1,N-1] = A[N-1,1] = 0
#%% Verify using Plot the Graph
# rows, cols = np.where(A == 1) # find edges from Adjacency matrix
# edges = zip(rows.tolist(), cols.tolist()) # convert rows and cols (edges) to list
# G = nx.Graph()
# G.add_edges_from(edges)
# labelmap = dict(zip(G.nodes(), range(1,N+1))) # add labels to graph
# nx.draw(G, labels=labelmap, with_labels=True)
# plt.show()
W_f_fgd = W_gen_M(N, A)
# We generate a FC graph and adjust the weight matrix to drop
# the connection from node 1 to node N
# W_f_fgd = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
# W_f_fgd[1,N-1] = W_f_fgd[N-1,1] = 0 # Drop a connection btw node 1 and node N
#%% Optimal Tc calculator
Tmix_fgd = Tmix_calc(W_f_fgd, N)
T_opt_gd = param[6,0] = int(np.ceil(Tmix_fgd*(3/2)*(np.log(N * tot_iter))))

#%% Parallelization Begins here #%%

#%% Generate samples for all monte-carlo runs # Consumes alot of memory.
x_samples_m = np.zeros((monte_carlo,N*tot_iter,d))
for sam in range(0,monte_carlo):
    # %% Generate N*tot_iter data samples using Cov Mat for N nodes per iteration.
    print('Generating Data samples for monte carlo run index: ', sam, '\n')
    x_samples_m[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma, N * tot_iter)

#%% Create functions to be run on each worker.
# CDIEGO Function (FGD graph) Tc = 1
def monte_carlo_mp_CDIEGO_FGD_Tc_1(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FGD) with Tc = 1 is: ', r,'\n',' with N = ',N)
    return CDIEGO(W_f_fgd, N, d, vti, tot_iter, x_samples_m[r, :, :], pca_vect, step_size,1)
# CDIEGO Function (FGD graph) Tc = Opt
def monte_carlo_mp_CDIEGO_FGD_Tc_opt(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FGD) with Tc = Opt is: ', r,'\n',' with N = ',N)
    return CDIEGO(W_f_fgd, N, d, vti, tot_iter, x_samples_m[r, :, :], pca_vect, step_size,T_opt_gd)


## Start Parallelization on Multiple workers
start_time = time.time()
if __name__ == '__main__':
    print('Starting MP process')
    mon_ranges = np.arange(0,monte_carlo).tolist()
    pool = mp.Pool() # no of parallel workers
    cdiego_m_fgd[0, :, :] = pool.map(monte_carlo_mp_CDIEGO_FGD_Tc_1, mon_ranges)
    cdiego_m_fgd[1, :, :] = pool.map(monte_carlo_mp_CDIEGO_FGD_Tc_opt, mon_ranges)
    pool.close()
    pool.join()
    ## Save the data of arrays
    np.save('sim_data/4.eff_topologies_synthetic_data_mp.npy', cdiego_m_fgd)
    np.save('sim_data/4.eff_topologies_synthetic_param_mp.npy', param)
print("--- %s seconds ---" % (time.time() - start_time))
