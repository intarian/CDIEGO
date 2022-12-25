#%% This file uses multi-processing capabilities to run the code on multiple cores.
# Only Monte-Carlo simulations are executed on multiple cores
#%% This file computes the effects of different value of T_c to compare the
# performance of DIEGO algorithm with CDIEGO using MNIST Data.
# The file runs CDIEGO algorithm for different values of T_c and shows that for optimal T_c,
# the performance of CDIEGO(T_opt) is same as DIEGO.
# The file just runs the algorithm and save the data in sim_data folder.
#%% Import Libraries and functions
import numpy as np
from pca_data_functions import *
import multiprocessing as mp
import time
from algorithms import *
#%% Load MNIST Dataset
import pickle
import gzip
# Load the dataset
(train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(
    gzip.open('MINST_dataset/mnist_py3k.pkl.gz', 'rb'))
data = np.concatenate((train_inputs, valid_inputs))
#%% Define Parameters:
param = np.zeros((9,1)) # Store the values of parameters
d = param[0,0] = data.shape[1] # Set dimensionality of data
N = param[1,0] = 10 # Set No of Nodes
tot_iter = param[2,0] = int(data.shape[0]/N) # Set no. of iterations
monte_carlo = param[3,0] = 50 # Set no. of monte carlo runs
p = param[4,0] = 0.1 # Set Parameter for Erdos-Reyni Convergence
step_size = param[5,0] = 0.05 # Set step size
#%% Compute True eigenvectors for MINST (Using Sample Covariance method)
A = (1/data.shape[0])*np.matrix(data.T@data) # Find covariance matrix Sigma using \Sigma = \tilde{A}^T \tilde{A}
eigv = np.linalg.eigh(A) # Find eigenvalue decomposition of eigv
ev = np.matrix(eigv[-1])  ## Fetch eigen vectors
pca_vect = ev[:, -1]
#%% Initialize empty array to store error for monte carlo simulations
# Save for DIEGO algorithm
diego_cdiego_m = np.zeros((4,monte_carlo,tot_iter))
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
T_a = int(np.ceil(Tmix))
T_b = int(np.ceil((np.log(N * (tot_iter)))))
T_opt = int(np.ceil(Tmix*(3/2)*(np.log(N * tot_iter))))
## Store the values above to param array
param[6,0] = T_a; param[7,0] = T_b; param[8,0] = T_opt;
#%% Parallelization Begins here #%%

#%% Generate samples for all monte-carlo runs # Consumes alot of memory.
x_samples_m = np.zeros((monte_carlo,N*tot_iter,d))
for sam in range(0,monte_carlo):
    # %% Generate N*tot_iter data samples using Cov Mat for N nodes per iteration.
    np.random.seed(1)
    np.random.shuffle(data)
    x_samples_m[sam,:,:] = data
#%% Create functions to be run on each worker.
# DIEGO Function
def monte_carlo_mp_DIEGO(r): # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for DIEGO is: ', r,'\n')
    return DIEGO(W_f, N, d, vti, tot_iter, x_samples_m[r, :, :], pca_vect, step_size)
# CDIEGO Function for T_a
def monte_carlo_mp_CDIEGO_Ta(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for CDIEGO T_a is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m[r,:,:], pca_vect, step_size, T_a)

# CDIEGO Function for T_b
def monte_carlo_mp_CDIEGO_Tb(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for CDIEGO T_b is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m[r, :, :], pca_vect, step_size, T_b)

# CDIEGO Function for T_max
def monte_carlo_mp_CDIEGO_Topt(r):  # r is the index of monte-carlo simulations being processed by a core
    print('Current Monte Carlo for CDIEGO T_opt is: ', r, '\n')
    return CDIEGO(W_nf, N, d, vti, tot_iter, x_samples_m[r, :, :], pca_vect, step_size, T_opt)

## Start Parallelization on Multiple workers
start_time = time.time()
if __name__ == '__main__':
    print('Starting MP process')
    mon_ranges = np.arange(0,monte_carlo).tolist()
    pool = mp.Pool() # no of parallel workers
    diego_cdiego_m[0, :, :] = pool.map(monte_carlo_mp_DIEGO, mon_ranges)
    diego_cdiego_m[1, :, :] = pool.map(monte_carlo_mp_CDIEGO_Ta, mon_ranges)
    diego_cdiego_m[2, :, :] = pool.map(monte_carlo_mp_CDIEGO_Tb, mon_ranges)
    diego_cdiego_m[3, :, :] = pool.map(monte_carlo_mp_CDIEGO_Topt, mon_ranges)
    pool.close()
    pool.join()
    ## Save the data of arrays
    np.save('sim_data/2.eff_Tc_DvsCD_mnist_data_mp.npy', diego_cdiego_m)
    np.save('sim_data/2.eff_Tc_DvsCD_mnist_params_mp.npy', param)

print("--- %s seconds ---" % (time.time() - start_time))


