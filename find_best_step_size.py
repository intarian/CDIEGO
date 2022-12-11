#%% Multi-processing is not used in this code
#%% This file uses both Algorithms DIEGO and CDIEGO under some certain parameters and finds the best step size.
# Changelog: We are only optimizing the best step size for DIEGO algorithm. The CIDEGO will be bound to that stepsize
# The step size which gives the performance close to diego case is choosen
#%% Import Libraries etc
import numpy as np
from matplotlib import pyplot as plt
from functions import *
#%% Begin define DIEGO Algorithm
def DIEGO_Hyp_comp(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
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
#%% Begin Main Implementation
## Define Parameters:
d = 20 # Set dimensionality of data
tot_iter = 5*1000 # Run for t iterations.
eig_gap_fac = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 40
monte_carlo = 10
p = 0.1 # Parameter for Erdos-Reyni Convergence
step_size = [0.01,0.05,0.1]
#%% Initialize empty tensor to store values of nodes and monte carlo simulations against total iteration
diego_f_cnnc_m = np.zeros((len(step_size),monte_carlo,tot_iter))
#%% Use fixed covariance matrix for all number of nodes and monte carlo simulations
[pca_vect, Sigma, eig_gap] = data_cov_mat_gen(d,eig_gap_fac) # Generate Cov Matrix and true eig vec
# %% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
for st in range(0,len(step_size)):
    #%% Begin Monte-Carlo Simulation Rounds
    for mon in range(0,monte_carlo):
        print('Currently Processing Nodes: ', N, ' of Monte Carlo: ',mon,', and StepSize: ',step_size[st],' \n')
        #%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration. Using covariance Mat
        x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
        #%% Run DIEGO Algorithm
        diego_f_cnnc_m[st,mon,:] = DIEGO_Hyp_comp(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size[st])
        #%% Run CDIEGO Algorithm
        # cdiego_f_cnnc_m[st,mon,:] = CDIEGO_Hyp_comp(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size[st],R_max)
#%% Compute Mean accross all Monte Carlo Simulations
#%% Plot Results
plt.figure()
for st in range(0,len(step_size)):
    # Compute Mean
    diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m[st,:,:], axis=0)))
    # Plot the curve
    plt.semilogy(diego_f_cnnc, label='DIEGO, StepSize='+str(step_size[st])+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.title('DIEGO with diff stepsize with d= '+str(d))
plt.ylabel('Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()
#%% Verify the performance through norm difference
# print(np.linalg.norm(diego_f_cnnc-cdiego_f_cnnc))
