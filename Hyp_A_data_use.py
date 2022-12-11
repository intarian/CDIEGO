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
#%% Load saved results
d = 4 # Set dimensionality of data
tot_iter = 5000 # Run for t iterations.
step_size = 0.1
eig_gap_fac = 0.3 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
eig_gap = np.round(siga[0]-siga[1],3)
N = 10
monte_carlo = 10
filename_data = 'sim_data_old/MP_hypothesis_A_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(N)+'_eg_'+str(eig_gap_fac)+'_ss_'+str(step_size)+'_mc_'+str(monte_carlo)+'.npy'
diego_f_cnnc_mn = np.load(filename_data)
#%% Plot Results
diego_scaling_true = 1/(np.sqrt((np.arange(1,tot_iter+1)))) # is scaling only by O(1/sqrt(Nt))
plt.figure()
# Plot true scaling
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# Plot nodes mean data against monte carlo simulations
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_mn, axis=0)))
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.title('DIEGO 1-time scale with d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()
#%% check scaling factors
t = 10000-8
print(diego_scaling_true[t-1]/diego_scaling_true[t])
print((1/np.sqrt(N*(t)))/(1/np.sqrt(N*(t+1))))
print(((t+1)/(t))**(11.21))
print(diego_f_cnnc[t-1]/diego_f_cnnc[t])



# #%% Check scaling by sqrt(N2/N1)
# a = 0
# b = 1
# t = 399*1000
# diego_f_cnnc_ma = np.squeeze(np.array(np.mean(diego_f_cnnc_mn[a,:,t], axis=0)))
# diego_f_cnnc_mb = np.squeeze(np.array(np.mean(diego_f_cnnc_mn[a,:,t-1000], axis=0)))
# x = diego_f_cnnc_ma/diego_f_cnnc_mb
# print(x)
# xa = np.sqrt(1/t + 1/(Node_count[a]*t))
# xb = np.sqrt(1/t + 1/(Node_count[b]*t))
# print ((xa/xb))
# xaa = np.sqrt(1/(Node_count[a]*t))
# xbb = np.sqrt(1/(Node_count[b]*t))
# print ((xaa/xbb))
# # plt.figure()
# # plt.semilogy(diego_f_cnnc_ma)
# # plt.semilogy(diego_f_cnnc_mb)
# # plt.show()
# # plt.figure()
# # plt.plot(x)
# # plt.show()
# #%% Plot Results
# # Compute eigengap
# siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
# eig_gap = np.round(siga[0]-siga[1],3)
# # True scaling for max No of nodes
# diego_scaling_true = 1/(np.sqrt((np.max(Node_count)*np.arange(1,tot_iter+1)))) # is scaling only by O(1/sqrt(Nt))
# plt.figure()
# # Plot true scaling
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# # Plot nodes mean data against monte carlo simulations
# for nodes in range(0,len(Node_count)):
#     diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_mn[nodes,:,:], axis=0)))
#     plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size)+', N= '+str(Node_count[nodes])+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.title('DIEGO 1-time scale with d= '+str(d))
# plt.ylabel('Mean Error')
# plt.xlabel('No. of Iterations')
# plt.legend()
# # filename_fig = 'figures/hypothesis_A_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(Node_count)+'.jpg'
# # plt.savefig(filename_fig)
# plt.show()
#
