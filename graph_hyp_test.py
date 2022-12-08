#%% This hypothesis tests is if there is any difference between the convergence for fully connected and
# weakly connected graph. For instance, if I drop one connection from fully connected network then what would happen
# to the convergence rates.
#%% This file uses saved data to play with results from Hyp A
import numpy as np
# import pandas as pd
# import itertools
# from scipy import stats as sps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
# import time
from functions import *
#%% Begin definition of CDIEGO Algorithm
# The algorithm should perform consensus based on no of rounds R. If R>1 and graph is strongly connected
# then I am basically consuming computer resources
def CDIEGO_HypB(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,Tc):
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
        for round in range(0,Tc): # Run for multiple communication rounds.
            upd_n = upd_n @ W
        # After updating upd_n data. The next thing is to scale it by W^R instead of N.
        W_e1 = (W ** Tc)
        W_e1 = W_e1[:, 0] + 0.0001 # as we are taking first column only. We need to add small eps to avoid scaling by 1/0
        # Update eigenvector estimate
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] + gamma_ext * (upd_n[:, i_n]) / (W_e1[i_n]).item()
            # v_n[:, i_n] = v_n[:, i_n] + gamma_ext * (upd_n[:, i_n])
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
d = 10 # Set dimensionality of data
tot_iter = 10*1000 # Run for t iterations.
step_size = 0.2
eig_gap_fac = 0.3 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 40
monte_carlo = 20
p = 0.1
#%% Initialize empty tensor to store values of nodes and monte carlo simulations against total iteration
diego_f_cnnc_mn = np.zeros((1,monte_carlo,tot_iter))
#%% Use fixed covariance matrix for all number of nodes and monte carlo simulations
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
# %% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
#%% Generate Erdos-Renyi Graph weight matrix
A = gen_graph_adjacency(N, p)
W_nf = W_gen_M(N, A)
#%% Generate Covariance matrix
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
#%% Fix initial vector for all nodes
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Begin Monte Carlo simulation
diego_f_cnnc_m = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_a = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_b = np.zeros((monte_carlo,tot_iter))
cdiego_round_m_c = np.zeros((monte_carlo,tot_iter))
Tmix = Tmix_calc(W_nf, N)
R_a = int(np.ceil((np.log(N * (tot_iter)))))  # sample+1 as sample starts from 0
R_b = int(np.ceil(Tmix))  # sample+1 as sample starts from 0
R_max = int(np.ceil(Tmix*3/2*np.ceil((np.log(N * (tot_iter))))))  # sample+1 as sample starts from 0.
for mon in range(0,monte_carlo):
    print('Currently Processing Nodes: ', N, ' of Monte Carlo: ',mon,' \n')
    #%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration.
    x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
    #%% Run DIEGO Algorithm
    diego_f_cnnc_m[mon,:] = CDIEGO_HypB(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size,1)
    #%% Run CDIEGO Algorithm
    cdiego_round_m_a[mon,:] = CDIEGO_HypB(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R_a)
    cdiego_round_m_b[mon, :] = CDIEGO_HypB(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_b)
    cdiego_round_m_c[mon, :] = CDIEGO_HypB(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_max)
#%% Take the mean of Monte-Carlo simulations
diego_f_cnnc= np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
cdiego_round_a= np.squeeze(np.array(np.mean(cdiego_round_m_a, axis=0)))
cdiego_round_b= np.squeeze(np.array(np.mean(cdiego_round_m_b, axis=0)))
cdiego_round_c= np.squeeze(np.array(np.mean(cdiego_round_m_c, axis=0)))
#%% Create temp data array to store Tc values computed randomly using Erdos-Renyi Method
diego_f_cnnc_Tc = np.zeros((tot_iter,2))
cdiego_round_a_Tc = np.zeros((tot_iter,2))
cdiego_round_b_Tc = np.zeros((tot_iter,2))
cdiego_round_c_Tc = np.zeros((tot_iter,2))
## Save the original data to temp array
diego_f_cnnc_Tc[:,0] = diego_f_cnnc
diego_f_cnnc_Tc[:,1] = 1
cdiego_round_a_Tc[:,0] = cdiego_round_a
cdiego_round_a_Tc[:,1] = R_a
cdiego_round_b_Tc[:,0] = cdiego_round_b
cdiego_round_b_Tc[:,1] = R_b
cdiego_round_c_Tc[:,0] = cdiego_round_c
cdiego_round_c_Tc[:,1] = R_max
#%% Save results
np.save('sim_data/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy', diego_f_cnnc_Tc)
np.save('sim_data/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCa_'+'.npy', cdiego_round_a_Tc)
np.save('sim_data/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCb_'+'.npy', cdiego_round_b_Tc)
np.save('sim_data/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCc_'+'.npy', cdiego_round_c_Tc)
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_round_a, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_a),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_round_b, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_b),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_round_c, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max),linestyle='dashed',linewidth=2)
plt.title('CDIEGO 2-time scale with d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
x = int(np.round(np.random.random()*100,2)) # Assign random id to file to prevent overwriting
# filename_fig = 'figures/hypothesis_B_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(N)+'_fid_'+str(x)+'.jpg'
# plt.savefig(filename_fig)
plt.show()
