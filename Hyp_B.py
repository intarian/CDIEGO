#%% This files proves Hypothesis A which is as follows:
#%% Hypothesis B: The hypothesis states that the mathematical analysis that is done for CDIEGO is correct. That is,
# T_c = \O(Tmix 3/2 log(Nt)) is necessary condition. If T_c \neq \O(T_mix 3/2 log(Nt)) then the rate mismatches.
# How to show: This can be shown by having fixed consensus rounds vs lograthimically increasing consensus rounds to
# show that the performance reaches 1/sqrt(Nt) curve.
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
#%% Begin define DIEGO function (modified for Hypothesis A) Algorithm (used for fully connected scenario)
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
            err_n[i_n, sample] = np.sqrt(1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2))
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    return mean_err
#%% Begin define CDIEGO function (modified for Hypothesis B) Algorithm (used for connected scenario)
# Introduction: This is a function of CDIEGO algorithm. Implements 2 scale algorithm to verify Hyp B
# The algorithm takes parameters like:
# step_size and decays using \alpha_t = \alpha/t,
# W: Weight matrix of the graph
# x_samples: data samples generated first
# N: no of nodes
# d: Dimensionality of data
# vti: Initial vector to distribute among N nodes for estimation
# tot_iter: Total no. of iterations to run the algorithm
# pca_vect: Original PCA vect to compare error.
# R: No of consensus rounds. If R is a fix positive number, then algorithm assumes fix no of consensus. Otherwise, if
# R is negative then algorithm assumes log increasing consensus rounds.
def CDIEGO_HypB(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R):
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
        ## Calculate Communication Rounds
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        for round in range(0,R): # Run for multiple communication rounds.
            upd_n = upd_n @ W
        # After updating upd_n data. The next thing is to scale it by W^R instead of N.
        W_e1 = (W ** R)
        W_e1 = W_e1[:, 0] + 0.0001 # as we are taking first column only. We need to add small eps to avoid scaling by 1/0
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
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    # mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    max_err = np.squeeze(np.array(np.max(err_n, axis=0)))
    return max_err # Give out max error among all nodes from q_1 instead of mean error.
#%% Define Parameters
d = 100 # Set dimensionality of data
tot_iter = 1000 # Run for t iterations.
step_size = 0.2
eig_gap_fac =  0.23 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 30 # no of nodes.
monte_carlo = 20
p = 0.2  # probability parameter to use for Erdos-Renyi graph generation.
#%% Compute true approximation (ignoring the scaling part)
# diego_scaling_true = 1/(np.sqrt((N*np.arange(1,tot_iter+1)))) # is scaling only by O(1/sqrt(Nt))
diego_scaling_true = 1/(np.sqrt((N*np.arange(1,tot_iter+1)))) + np.arange(1,tot_iter+1)
# %% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
#%% Generate Erdos-Renyi Graph weight matrix
A = gen_graph_adjacency(N, p)
W_nf = W_gen_M(N, A)
#%% Begin Monte Carlo simulation
diego_f_cnnc_m = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_a = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_b = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_c = np.zeros((monte_carlo,tot_iter))
Tmix = Tmix_calc(W_nf, N)
R_max = int(np.ceil((Tmix * 3/2 *np.log(N * (tot_iter)))))  # sample+1 as sample starts from 0.
R_max
for mon in range(0,monte_carlo):
    print('Currently Processing Nodes: ', N, ' of Monte Carlo: ',mon,' \n')
    #%% Data Generation obtain covariance matrix and pca vector
    [pca_vect, Sigma, eig_gap] = data_gen_cov_mat(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
    vti = np.random.normal(0,1,(d)) # Generate random eigenvector
    vti = Column(vti/np.linalg.norm(vti)) # unit normalize for selection over unit sphere
    #%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration.
    x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
    #%% Run DIEGO Algorithm
    diego_f_cnnc_m[mon,:] = DIEGO_HypA(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
    #%% Run CDIEGO Algorithm
    cdiego_round_m_a[mon,:] = CDIEGO_HypB(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R_max-50)
    cdiego_round_m_b[mon, :] = CDIEGO_HypB(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_max)
    # cdiego_round_m_c[mon, :] = CDIEGO_HypB(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_max+50)
#%% Take the mean of Monte-Carlo simulations
diego_f_cnnc= np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
cdiego_round_a= np.squeeze(np.array(np.mean(cdiego_round_m_a, axis=0)))
cdiego_round_b= np.squeeze(np.array(np.mean(cdiego_round_m_b, axis=0)))
# cdiego_round_c= np.squeeze(np.array(np.mean(cdiego_round_m_c, axis=0)))
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_round_a, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max-50),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_round_b, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_c, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max+50),linestyle='dashed',linewidth=2)
plt.title('CDIEGO 2-time scale with d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
x = int(np.round(np.random.random()*100,2)) # Assign random id to file to prevent overwriting
# filename_fig = 'figures/hypothesis_B_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(N)+'_fid_'+str(x)+'.jpg'
# plt.savefig(filename_fig)
plt.show()
# #%% Plot log R curve
# plt.figure()
# Tmix = Tmix_calc(W_nf, N)
# R = int(np.ceil(Tmix * (3/2) *np.log(N * (np.arange(1,tot_iter+1)))))  # sample+1 as sample starts from 0.
# plt.title('Communication rounds increasing in log sense')
# plt.ylabel('Comm Rounds')
# plt.xlabel('No. of Iterations')
# plt.show()