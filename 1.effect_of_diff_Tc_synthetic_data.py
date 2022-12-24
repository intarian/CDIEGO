#%% This file computes the effects of different value of T_c to compare the
# performance of DIEGO algorithm with CDIEGO using Synthetic Data.
# The file runs CDIEGO algorithm for different values of T_c and shows that for optimal T_c,
# the performance of CDIEGO(T_opt) is same as DIEGO.
# The file just runs the algorithm and save the data in sim_data folder.
#%% Import Libraries and functions
import numpy as np
from pca_data_functions import *
from algorithms import *
#%% Define Parameters:
param = np.zeros((10,1)) # Store the values of parameters
d = param[0,0] = 20 # Set dimensionality of data
tot_iter = param[1,0] = 50*1000 # Set no. of iterations
N = param[2,0] = 40 # Set No of Nodes
monte_carlo = param[3,0] = 10 # Set no. of monte carlo runs
p = param[4,0] = 0.1 # Set Parameter for Erdos-Reyni Convergence
step_size = param[5,0] = 0.05 # Set step size
#%% Set Eigengap
eig_gap_fac = 2 #Controls the factor of eigengap.
siga = ((1 - 0.1) / (np.arange(1, d + 1)) ** eig_gap_fac) + 0.1
eigen_gap = param[6,0] = np.round(siga[0] - siga[1], 3) # Set Eigengap
#%% Initialize empty array to store error for monte carlo simulations
# Save for DIEGO algorithm
diego_cdiego_m = np.zeros((4,monte_carlo,tot_iter))
#%% Generate Covariance Matrix and ground truth eigenvector
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
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
param[7,0] = T_a; param[8,0] = T_b; param[9,0] = T_opt;

#%% Begin Monte-Carlo Simulation Rounds
for mon in range(0,monte_carlo):
    print('Current Monte Carlo Run is: ',mon,'\n')
    #%% Generate N*tot_iter data samples using Cov Mat for N nodes per iteration.
    x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
    #%% Run DIEGO Algorithm
    diego_cdiego_m[0,mon,:] = DIEGO(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
    #%% Run CDIEGO Algorithm with different communication rounds
    diego_cdiego_m[1,mon,:] = CDIEGO(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,T_a)
    diego_cdiego_m[2,mon,:] = CDIEGO(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,T_b)
    diego_cdiego_m[3,mon, :] = CDIEGO(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, T_opt)

#%% Save the data of arrays
np.save('sim_data/1.eff_Tc_DvsCD_data.npy', diego_cdiego_m)
np.save('sim_data/1.eff_Tc_DvsCD_params.npy', param)