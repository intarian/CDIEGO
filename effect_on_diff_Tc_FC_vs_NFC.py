#%% Multi-processing is not used in this code
#%% This file uses both Algorithms DIEGO and CDIEGO under some certain parameters and finds the best step size.
# The step size which gives the performance close to diego case is choosen
#%% Import Libraries etc
import numpy as np
from matplotlib import pyplot as plt
from functions import *
#%% Begin define DIEGO Algorithm
def DIEGO_Hyp_Tc(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
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
def CDIEGO_Hyp_Tc(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,Tc):
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
## Define Parameters:
d = 20 # Set dimensionality of data
tot_iter = 50*1000 # Run for t iterations.
eig_gap_fac = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 40
monte_carlo = 100
p = 0.1 # Parameter for Erdos-Reyni Convergence
step_size = 0.05
#%% Initialize empty tensor to store values of nodes and monte carlo simulations against total iteration
diego_f_cnnc_m = np.zeros((monte_carlo,tot_iter))
# Each a b c arrays created below will save different values of Tc
cdiego_f_cnnc_m_a = np.zeros((monte_carlo,tot_iter))
cdiego_f_cnnc_m_b = np.zeros((monte_carlo,tot_iter))
# cdiego_f_cnnc_m_c = np.zeros((monte_carlo,tot_iter))
# cdiego_f_cnnc_m_d = np.zeros((monte_carlo,tot_iter))
cdiego_f_cnnc_m_rmax = np.zeros((monte_carlo,tot_iter))
#%% Use fixed covariance matrix for all number of nodes and monte carlo simulations
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
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
R_a = int(np.ceil(Tmix))
R_b = int(np.ceil((np.log(N * (tot_iter)))))
R_c = int(np.ceil((Tmix*np.log(N * (tot_iter)))))
R_max = int(np.ceil(Tmix*(3/2)*(np.log(N * tot_iter))))
R_d = R_max + 10
#%% Begin Monte-Carlo Simulation Rounds
for mon in range(0,monte_carlo):
    print('Currently Processing Nodes: ', N, ' of Monte Carlo: ',mon,'\n')
    #%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration. Using covariance Mat
    x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
    #%% Run DIEGO Algorithm
    diego_f_cnnc_m[mon,:] = DIEGO_Hyp_Tc(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
    #%% Run CDIEGO Algorithm with different communication rounds
    cdiego_f_cnnc_m_a[mon,:] = CDIEGO_Hyp_Tc(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R_a)
    cdiego_f_cnnc_m_b[mon,:] = CDIEGO_Hyp_Tc(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R_b)
    # cdiego_f_cnnc_m_c[mon,:] = CDIEGO_Hyp_Tc(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size,R_c)
    cdiego_f_cnnc_m_rmax[mon, :] = CDIEGO_Hyp_Tc(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_max)
    # cdiego_f_cnnc_m_d[mon, :] = CDIEGO_Hyp_Tc(W_nf, N, d, vti, tot_iter, x_samples, pca_vect, step_size, R_d)
#%% Save All Results
np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy', diego_f_cnnc_m)
np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_a)+'_NFCa_'+'.npy', cdiego_f_cnnc_m_a)
np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_b)+'_NFCb_'+'.npy', cdiego_f_cnnc_m_b)
# np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
#         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_c)+'_NFCc_'+'.npy', cdiego_f_cnnc_m_c)
np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max)+'_NFCmax_'+'.npy', cdiego_f_cnnc_m_rmax)
# np.save('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
#         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_d)+'_NFCd_'+'.npy', cdiego_f_cnnc_m_d)
#%% Compute Mean accross all Monte Carlo Simulations
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
cdiego_f_cnnc_a = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_a, axis=0)))
cdiego_f_cnnc_b = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_b, axis=0)))
# cdiego_f_cnnc_c = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_c, axis=0)))
cdiego_f_cnnc_rmax = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_rmax, axis=0)))
# cdiego_f_cnnc_d = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_d, axis=0)))
#%% Plot Results
# plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# ## Plot the curve
# plt.semilogy(diego_f_cnnc, label='DIEGO, StepSize='+str(ss_diego)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_a, label='CDIEGO, Tc = '+str(R_a)+' StepSize='+str(ss_cdiego_a)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_b, label='CDIEGO, Tc = '+str(R_b)+' StepSize='+str(ss_cdiego_b)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_c, label='CDIEGO, Tc = '+str(R_c)+' StepSize='+str(ss_cdiego_c)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='solid',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_rmax, label='CDIEGO, Tc = '+str(R_max)+' StepSize='+str(ss_cdiego_rmax)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_d, label='CDIEGO, Tc = '+str(R_d)+' StepSize='+str(ss_cdiego_d)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='solid',linewidth=2)
# plt.title('DIEGO vs CDIEOG with diff stepsize with d= '+str(d))
# plt.ylabel('Error')
# plt.xlabel('No. of Iterations')
# plt.legend()
# plt.show()
