#%% This file contains implementation of CDIEGO algorithm. (for K=1 eigenvector)
#%% Inputs for function
# W = Weight Matrix for algorithm (contains weight of the edges)
# N = No. of Nodes of the network
# d = Dimensions of the data of each sample
# vti = Initial vector to initialize at all nodes
# tot_iter = Total No. of iterations to run
# x_samples = Data samples arrive at all nodes at each iterations. Total size of samples is N*tot_iter
# pca_vect: ground truth eigenvector. used to compute the error
# step_size: initial value of decaying stepsize.
# Tc: No. of consensus rounds. Only used for CDIEGO.
#%% Import basic libraries
import numpy as np
from pca_data_functions import *

#%% Begin define CDIEGO Algorithm
def CDIEGO(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,Tc):
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
            W_e1 = W_e1[:, 0]
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
    return max_err # Give out max error among all nodes from q_1.