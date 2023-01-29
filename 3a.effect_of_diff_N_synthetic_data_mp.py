#%% This file uses multi-processing capabilities to run the code on multiple cores.
# Only Monte-Carlo simulations are executed on multiple cores
#%% This file computes the effects of different value of N (no of nodes) to compare the
# performance of C-DIEGO algorithm under NFC graph using CDIEGO using Synthetic Data.
# The file runs CDIEGO algorithm under different values of N using the same covariance matrix.
# The graphs are generated using Erdos-Reyni method with fixed parameter p.
# The file just runs the algorithm and save the data in sim_data folder.
#%% Import Libraries and functions
import numpy as np
from pca_data_functions import *
import multiprocessing as mp
import time
from algorithms import *
#%% Define Parameters:
param = np.zeros((10,1)) # Store the values of parameters
d = param[0,0] = 20 # Set dimensionality of data
tot_iter = param[1,0] = 10*1000 # Set no. of iterations
monte_carlo = param[2,0] = 20 # Set no. of monte carlo runs
#%% Set Eigengap
eig_gap_fac = 2 #Controls the factor of eigengap.
siga = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac) + 0.1
eigen_gap = param[3,0] = np.round(siga[0] - siga[1], 3) # Set Eigengap
# Set step size for each topology size with diff no of Node
step_size_Na = param[4,0] = 0.5
step_size_Nb = param[5,0] = 1
step_size_Nc = param[6,0] = 1.5
## (use multiple factor of tot_iter for even distribution of data at N nodes)
Na = param[7,0] = 10 # Set No of Nodes a
Nb = param[8,0] = 10 # Set No of Nodes b
Nc = param[9,0] = 10 # Set No of Nodes c
#%% Initialize empty array to store error for monte carlo simulations
# Save monte-carlo runs for C-DIEGO algorithm under different setting for each N
cdiego_m_Na = np.zeros((monte_carlo,tot_iter))
cdiego_m_Nb = np.zeros((monte_carlo,tot_iter))
cdiego_m_Nc = np.zeros((monte_carlo,tot_iter))
#%% Generate Covariance Matrix and ground truth eigenvector
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
#%% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Load Topologies generated using file 3a.effect_of_diff_N_topologies_gen.py
A_Na = np.load('sim_data/3a.topo_gen_Na.npy')
A_Nb = np.load('sim_data/3a.topo_gen_Nb.npy')
A_Nc = np.load('sim_data/3a.topo_gen_Na.cnpy')
W_nf_Na = W_gen_M(Na, A_Na)
W_nf_Nb = W_gen_M(Nb, A_Nb)
W_nf_Nc = W_gen_M(Nc, A_Nc)
# #%% Ensure that node 1 is connnected to all nodes. (This should be changed later)
# while (W_nf_Na[:, 0] == 0).any():
#     print('\nW_nf_Na: ', ' with N = ', Na)
#     W_nf_Na = W_gen_M(Na, A_Na)
# while (W_nf_Nb[:, 0] == 0).any():
#     print('\nW_nf_Na: ', ' with N = ', Nb)
#     W_nf_Nb = W_gen_M(Nb, A_Nb)
# while (W_nf_Nc[:, 0] == 0).any():
#     print('\nW_nf_Na: ', ' with N = ', Nc)
#     W_nf_Nc = W_gen_M(Nc, A_Nc)
#%% Parallelization Begins here #%%

#%% Generate samples for all monte-carlo runs # Consumes alot of memory.
x_samples_m_Na = np.zeros((monte_carlo,Na*tot_iter,d))
x_samples_m_Nb = np.zeros((monte_carlo,Nb*tot_iter,d))
x_samples_m_Nc = np.zeros((monte_carlo,Nc*tot_iter,d))
for sam in range(0,monte_carlo):
    # %% Generate N*tot_iter data samples using Cov Mat for N nodes per iteration.
    x_samples_m_Na[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma, Na * tot_iter)
    x_samples_m_Nb[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma, Nb * tot_iter)
    x_samples_m_Nc[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma, Nc * tot_iter)

#%% Create functions to be run on each worker.
# CDIEGO Function (FC graph) Nodes a
def monte_carlo_mp_CDIEGO_Na(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) is: ', r,'\n',' with N = ',Na)
    return CDIEGO(W_nf_Na, Na, d, vti, tot_iter, x_samples_m_Na[r, :, :], pca_vect, step_size_Na,1)
# CDIEGO Function (FC graph) Nodes b
def monte_carlo_mp_CDIEGO_Nb(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) is: ', r,'\n',' with N = ',Nb)
    return CDIEGO(W_nf_Nb, Nb, d, vti, tot_iter, x_samples_m_Nb[r, :, :], pca_vect, step_size_Nb,1)
# CDIEGO Function (FC graph) Nodes c
def monte_carlo_mp_CDIEGO_Nc(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) is: ', r,'\n',' with N = ',Nc)
    return CDIEGO(W_nf_Nc, Nc, d, vti, tot_iter, x_samples_m_Nc[r, :, :], pca_vect, step_size_Nc,1)

## Start Parallelization on Multiple workers
start_time = time.time()
if __name__ == '__main__':
    print('Starting MP process')
    mon_ranges = np.arange(0,monte_carlo).tolist()
    pool = mp.Pool() # no of parallel workers
    cdiego_m_Na = pool.map(monte_carlo_mp_CDIEGO_Na, mon_ranges)
    cdiego_m_Nb = pool.map(monte_carlo_mp_CDIEGO_Nb, mon_ranges)
    cdiego_m_Nc = pool.map(monte_carlo_mp_CDIEGO_Nc, mon_ranges)
    pool.close()
    pool.join()
    ## Save the data of arrays
    np.save('sim_data/3a.eff_Na_synthetic_data_mp.npy', cdiego_m_Na)
    np.save('sim_data/3a.eff_Nb_synthetic_data_mp.npy', cdiego_m_Nb)
    np.save('sim_data/3a.eff_Nc_synthetic_data_mp.npy', cdiego_m_Nc)
    np.save('sim_data/3a.eff_N_synthetic_param_mp.npy', param)
print("--- %s seconds ---" % (time.time() - start_time))