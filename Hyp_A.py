#%% This files proves Hypothesis A which is as follows:
#%% Hypothesis A: This hypothesis shows that the eigenvector estimate \hat{v} from strongly connected graph follows
# the order of the curve 1/sqrt(Nt); i.e. verifies the rate of first term in the analysis. This also shows that as the
# N increases we are more close (i.e. faster convergence) to 1/sqrt(Nt) as opposed to low values of N.
#%% Begin Implementation. Start with Importing Libraries
import numpy as np
import networkx as nx
# import pandas as pd
# import itertools
# from scipy import stats as sps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
# import time
from functions import *
#%% Begin define DIEGO function (modified for Hypothesis A) Algorithm
# Introduction: This is a function of DIEGO algorithm. Implements 1 scale algorithm to verify Hyp A
# The algorithm takes parameters like:
# step_size and decays using \alpha_t = \alpha/t,
# W: Weight matrix of the graph
# x_samples: data samples generated first
# N: no of nodes
# d: Dimensionality of data
# vti: Initial vector to distribute among N nodes for estimation
# tot_iter: Total no. of iterations to run the algorithm
# pca_vect: Original PCA vect to compare error.
def DIEGO_HypA(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
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
            err_n[i_n, sample] = 1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2)
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    return mean_err
#%% Define Parameters
N = 10 # Set No. of Nodes, Each node gets one sample in single iteration
d = 5 # Set dimensionality of data
tot_iter = 1000 # Run for t iterations.
step_size = 0.2
eig_vala = 1
eig_valb = 0.1
eig_gap =  (eig_vala-eig_valb)/d #1 is largest eigval and 0.1 is smallest and there are d eigvalues so lambda_1 - lambda_2 = 0.9/d
monte_carlo = 50
# %% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
#%% Begin Monte Carlo simulation
diego_f_cnnc_m = np.zeros((monte_carlo,tot_iter))
for mon in range(0,monte_carlo):
    #%% Data Generation obtain covariance matrix and pca vector
    [pca_vect, Sigma] = data_gen_cov_mat(d,eig_vala,eig_valb) # Generate Cov Matrix and true eig vec
    vti = np.random.normal(0,1,(d)) # Generate random eigenvector
    vti = Column(vti/np.linalg.norm(vti)) # unit normalize for selection over unit sphere
    #%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration.
    x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
    #%% Run DIEGO Algorithm for Hyp A
    diego_f_cnnc_m[mon,:] = DIEGO_HypA(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
diego_f_cnnc= np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
#%% Compute true approximation (ignoring the scaling part)
diego_scaling_fac_a = 1/(((N*np.arange(1,tot_iter+1)))) # is scaling only by O(1/sqrt(Nt))
# diego_scaling_fac_b = ((1/(np.arange(1,tot_iter+1)) + 1/(N*np.arange(1,tot_iter+1)))) #is by O(sqrt(1/t +1/(Nt)))
diego_scaling_fac_c = 1/(np.sqrt((N*np.arange(1,tot_iter+1)))) # is scaling only by O(1/sqrt(Nt))
# diego_scaling_fac_d = np.sqrt((1/(np.arange(1,tot_iter+1)) + 1/(N*np.arange(1,tot_iter+1)))) #is by O(sqrt(1/t +1/(Nt)))
#%% Plot Results
plt.figure()
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size),linestyle='dashed',linewidth=2)
plt.semilogy(diego_scaling_fac_a, label='Scaling by O(1/(Nt))',linestyle='solid',linewidth=2)
# plt.semilogy(diego_scaling_fac_b, label='Scaling by O((1/t +1/(Nt)))',linestyle='solid',linewidth=2)
plt.semilogy(diego_scaling_fac_c, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# plt.semilogy(diego_scaling_fac_d, label='Scaling by O(sqrt(1/t +1/(Nt)))',linestyle='solid',linewidth=2)
plt.title('DIEGO 1-time scale with N= '+str(N)+' nodes and d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()
