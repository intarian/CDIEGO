#%% This code uses MINST Dataset and see the best T_c for
#%% Use Multi_Processing libraries
#%% Import Libraries etc
import numpy as np
from matplotlib import pyplot as plt
from functions import *
import time
import os
#%% Begin define DIEGO Algorithm
def DIEGO_MINST(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
    # Data samples distribute among N nodes
    x_samples_n = np.reshape(x_samples, (N, tot_iter, d))
    # Now we initialize all N nodes with initial eigenvector vti
    v_n = np.matrix(np.repeat(vti, N, axis=1))  # Initialize eigenvector estimate with vti at all nodes
    err_n = np.zeros([N, tot_iter])  # store error across all N nodes in each iteration
    # Begin loop to draw sample from each node and then compute their estimates and updates using W
    for sample in range(0, tot_iter):
        gamma_ext = step_size / (1 + sample)
        # gamma_ext = np.ceil(((sample + 1)) / 10) # piecewise linear decay
        upd_n = np.matrix(np.zeros([d, N]))  # Store update values across all nodes for each sample
        # Begin loop to take sample from each node and update eigenvector estimate
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        upd_n = upd_n @ W
        # Update eigenvector estimate
        v_n = v_n + gamma_ext * N*upd_n # for fully connected graph. scale by 1/[W^*e_1]i i.e scale by N
        # Normalize the estimate to make unit norm
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] / np.linalg.norm(v_n[:, i_n], 2)
        # Compute Error for each iteration
        for i_n in range(0, N):
            err_n[i_n, sample] = np.sqrt(1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2))
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    return mean_err
#%% Begin define CDIEGO Algorithm
def CDIEGO_MINST(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,Tc):
    # Data samples distribute among N nodes
    x_samples_n = np.reshape(x_samples, (N, tot_iter, d))
    # Now we initialize all N nodes with initial eigenvector vti
    v_n = np.matrix(np.repeat(vti, N, axis=1))  # Initialize eigenvector estimate with vti at all nodes
    err_n = np.zeros([N, tot_iter])  # store error across all N nodes in each iteration
    # Begin loop to draw sample from each node and then compute their estimates and updates using W
    for sample in range(0, tot_iter):
        gamma_ext = step_size / (1 + sample)
        upd_n = np.matrix(np.zeros([d, N]))  # Store update values across all nodes for each sample
        # Begin loop to take sample from each node and update eigenvector estimate
        ## Calculate Communication Rounds
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        for round in range(0,Tc): # Run for multiple communication rounds.
            upd_n = upd_n @ W
        # After updating upd_n data. The next thing is to scale it by W^R instead of N.
        W_e1 = (W ** Tc)
        if (W_e1[:, 0] == 0).any():
            print(W)
            print('Failed to run. Weight matrix has a zero. Try increasing Tmix')
            break
        else:
            W_e1 = W_e1[:, 0] # + 1e-100 # as we are taking first column only. We need to add small eps to avoid scaling by 1/0
        # Update eigenvector estimate
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] + gamma_ext * (upd_n[:, i_n]) / (W_e1[i_n]).item()
        # Normalize the estimate to make unit norm
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] / np.linalg.norm(v_n[:, i_n], 2)
        # Compute Error for each iteration
        for i_n in range(0, N):
            err_n[i_n, sample] = np.sqrt(1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2))
    max_err = np.squeeze(np.array(np.max(err_n, axis=0)))
    return max_err # Give out max error among all nodes from q_1 instead of mean error.

#%% Begin Main Implementation
## Load MINST DATASET
import pickle
import gzip
# Load the dataset
(train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(
    gzip.open('MINST_dataset/mnist_py3k.pkl.gz', 'rb'))
data_concatenated = np.concatenate((train_inputs, valid_inputs))
#%% Compute True eigenvectors for MINST (Using Batch method)
A = (1/60000)*np.matrix(data_concatenated.T@data_concatenated) # Find covariance matrix Sigma using \Sigma = \tilde{A}^T \tilde{A}
eigv = np.linalg.eigh(A) # Find eigenvalue decomposition of eigv
ev = np.matrix(eigv[-1])  ## Fetch eigen vectors
pca_vect = ev[:, -1]

#%% Load Additional Parameters
N = 10
monte_carlo = 50
p = 0.1 # Parameter for Erdos-Reyni Convergence
step_size = 0.01
tot_data_samples = 60000
d = 784
tot_iter = int(tot_data_samples/N)
#%% Create empty array
cases = 3 # 3 cases to compare: diego vs cdiego (log(Nt) vs cdiego (Tmix 3/2 log(Nt)
diego_cdiego_m = np.zeros((cases,monte_carlo,tot_iter))
## Each a b c arrays created below will save different values of Tc
# cdiego_f_cnnc_m_a = np.zeros((monte_carlo,tot_iter))
# %% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
#%% Generate Erdos-Renyi Graph and weight matrix using Metropolis-Hastling Algorithm
A = gen_graph_adjacency(N, p)
W_nf = W_gen_M(N, A)
#%% Compute Tmix and Maximum No. Of Rounds
Tmix = Tmix_calc(W_nf, N)
R_a = int(np.ceil((np.log(N * (tot_iter)))))
R_max = int(np.ceil(Tmix*(3/2)*(np.log(N * tot_iter))))
#%% Store samples in all monte-carlo sim
# x_samples = np.zeros((monte_carlo,N*tot_iter,d))
# for sam in range(0,monte_carlo):
#     np.random.seed(1)
#     np.random.shuffle(data_concatenated)
#     x_samples[sam,:,:] = data_concatenated
#%% Run Monte carlo
start_time = time.time()
for im in range(0,monte_carlo):
    np.random.shuffle(data_concatenated)
    print('Current Monte Carlo index is: ', im, '\n')
    diego_cdiego_m[0, im, :] = DIEGO_MINST(W_f, N, d, vti, tot_iter, data_concatenated, pca_vect, step_size)
    diego_cdiego_m[1, im, :] = CDIEGO_MINST(W_nf, N, d, vti, tot_iter, data_concatenated, pca_vect, step_size, R_a)
    diego_cdiego_m[2, im, :] = CDIEGO_MINST(W_nf, N, d, vti, tot_iter, data_concatenated, pca_vect, step_size, R_max)

filename_data = 'sim_data/MP_DIEGO_CDIEGO_MNIST_amarel_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
    N) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) +  '_Ra_' + str(R_a)  + '_Rmax_' + str(R_max) + '.npy'
np.save(filename_data, diego_cdiego_m)
print("--- %s seconds ---" % (time.time() - start_time))