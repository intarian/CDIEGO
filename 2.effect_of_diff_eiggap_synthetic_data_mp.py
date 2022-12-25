#%% This file uses multi-processing capabilities to run the code on multiple cores.
# Only Monte-Carlo simulations are executed on multiple cores
#%% This file computes the effects of different value of eigengap $\Lambda$ to compare the
# performance of C-DIEGO algorithm under FC vs. NFC graph with optimal Tc using Synthetic data under different $\lambda$
# The file runs C-DIEGO algorithm for different values of $\Lambda$ under a FC graph and a fixed NFC graph.
# The idea is to show the convergence is better with large $\lambda$ and no effect with using optimal Tc. i.e.
# the performance of C-DIEGO(T_opt) is same as with C-DIEGO under FC.
# We supply Tc = 1 for C-DIEGO under FC graph
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
tot_iter = param[1,0] = 50*1000 # Set no. of iterations
N = param[2,0] = 40 # Set No of Nodes
monte_carlo = param[3,0] = 50 # Set no. of monte carlo runs
p = param[4,0] = 0.1 # Set Parameter for Erdos-Reyni Convergence
step_size = param[5,0] = 0.05 # Set step size
#%% Set different Eigengaps
# Set Eigengap a
eig_gap_fac_a = 1 #Controls the factor of eigengap.
siga = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac_a) + 0.1
eigen_gap_a = param[6,0] = np.round(siga[0] - siga[1], 3)
# Set Eigengap b
eig_gap_fac_b = 2 #Controls the factor of eigengap.
sigb = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac_b) + 0.1
eigen_gap_b = param[7,0] = np.round(sigb[0] - sigb[1], 3)
# Set Eigengap c
eig_gap_fac_c = 4 #Controls the factor of eigengap.
sigc = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac_c) + 0.1
eigen_gap_c = param[8,0] = np.round(sigc[0] - sigc[1], 3)
#%% Initialize empty array to store error for monte carlo simulations
# Save monte-carlo runs for C-DIEGO algorithm under FC vs NFC under each eigengap
cdiego_m_ega = np.zeros((2,monte_carlo,tot_iter))
cdiego_m_egb = np.zeros((2,monte_carlo,tot_iter))
cdiego_m_egc = np.zeros((2,monte_carlo,tot_iter))
#%% Generate Covariance Matrix and ground truth eigenvector for different Eigengap
[pca_vect_a, Sigma_a, eig_gap_a] = data_cov_mat_gen(d,eig_gap_fac_a)
[pca_vect_b, Sigma_b, eig_gap_b] = data_cov_mat_gen(d,eig_gap_fac_b)
[pca_vect_c, Sigma_c, eig_gap_c] = data_cov_mat_gen(d,eig_gap_fac_c)
#%% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
#%% Generate Erdos-Renyi Graph and weight matrix using Metropolis-Hastling Algorithm
A = gen_graph_adjacency(N, p)
W_nf = W_gen_M(N, A)
#%% Compute Tmix and Maximum No. Of Rounds for different values of CDIEGO
Tmix = Tmix_calc(W_nf, N)
T_opt = param[9,0] = int(np.ceil(Tmix*(3/2)*(np.log(N * tot_iter))))

#%% Parallelization Begins here #%%

#%% Generate samples for all monte-carlo runs # Consumes alot of memory.
x_samples_m_ega = np.zeros((monte_carlo,N*tot_iter,d))
x_samples_m_egb = np.zeros((monte_carlo,N*tot_iter,d))
x_samples_m_egc = np.zeros((monte_carlo,N*tot_iter,d))
for sam in range(0,monte_carlo):
    # %% Generate N*tot_iter data samples using Cov Mat for N nodes per iteration.
    x_samples_m_ega[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma_a, N * tot_iter)
    x_samples_m_egb[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma_b, N * tot_iter)
    x_samples_m_egc[sam, :, :] = np.random.multivariate_normal(np.zeros(d), Sigma_c, N * tot_iter)

#%% Create functions to be run on each worker.
## For Eigengap a
# C-DIEGO Function under FC
def monte_carlo_mp_CDIEGO_FC_ega(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) under eigengap a is: ', r,'\n')
    return CDIEGO(W_f, N, d, vti, tot_iter, x_samples_m_ega[r, :, :], pca_vect_a, step_size,1)
# CDIEGO Function under T_opt
def monte_carlo_mp_CDIEGO_Topt_ega(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (NFC) T_opt under eigengap a is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m_ega[r,:,:], pca_vect_a, step_size, T_opt)

## For Eigengap b
# C-DIEGO Function under FC
def monte_carlo_mp_CDIEGO_FC_egb(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) under eigengap b is: ', r,'\n')
    return CDIEGO(W_f, N, d, vti, tot_iter, x_samples_m_egb[r, :, :], pca_vect_b, step_size,1)
# CDIEGO Function under T_opt
def monte_carlo_mp_CDIEGO_Topt_egb(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (NFC) T_opt under eigengap b is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m_egb[r,:,:], pca_vect_b, step_size, T_opt)

## For Eigengap c
# C-DIEGO Function under FC
def monte_carlo_mp_CDIEGO_FC_egc(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (FC) under eigengap c is: ', r,'\n')
    return CDIEGO(W_f, N, d, vti, tot_iter, x_samples_m_egc[r, :, :], pca_vect_c, step_size,1)
# CDIEGO Function under T_opt
def monte_carlo_mp_CDIEGO_Topt_egc(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for C-DIEGO (NFC) T_opt under eigengap c is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m_egc[r,:,:], pca_vect_c, step_size, T_opt)


## Start Parallelization on Multiple workers
start_time = time.time()
if __name__ == '__main__':
    print('Starting MP process')
    mon_ranges = np.arange(0,monte_carlo).tolist()
    pool = mp.Pool() # no of parallel workers
    cdiego_m_ega[0, :, :] = pool.map(monte_carlo_mp_CDIEGO_FC_ega, mon_ranges)
    cdiego_m_ega[1, :, :] = pool.map(monte_carlo_mp_CDIEGO_Topt_ega, mon_ranges)
    cdiego_m_egb[0, :, :] = pool.map(monte_carlo_mp_CDIEGO_FC_egb, mon_ranges)
    cdiego_m_egb[1, :, :] = pool.map(monte_carlo_mp_CDIEGO_Topt_egb, mon_ranges)
    cdiego_m_egc[0, :, :] = pool.map(monte_carlo_mp_CDIEGO_FC_egc, mon_ranges)
    cdiego_m_egc[1, :, :] = pool.map(monte_carlo_mp_CDIEGO_Topt_egc, mon_ranges)
    pool.close()
    pool.join()
    ## Save the data of arrays
    np.save('sim_data/2.eff_eiggap_ega_synthetic_data_mp.npy', cdiego_m_ega)
    np.save('sim_data/2.eff_eiggap_egb_synthetic_data_mp.npy', cdiego_m_egb)
    np.save('sim_data/2.eff_eiggap_egc_synthetic_data_mp.npy', cdiego_m_egc)
    np.save('sim_data/2.eff_eiggap_synthetic_params_mp.npy', param)
print("--- %s seconds ---" % (time.time() - start_time))



