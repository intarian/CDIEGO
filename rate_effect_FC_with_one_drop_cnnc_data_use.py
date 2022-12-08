#%% THis experiment check the performance of the algorithm DIEGO vs CDIEGO when the graph is almost fully connected.
# Use Saved Data and plot the results
#%% Import Libraries etc
import numpy as np
from matplotlib import pyplot as plt
from functions import *
#%% Begin Main Implementation
## Define Parameters:
d = 5 # Set dimensionality of data
tot_iter = 10*100 # Run for t iterations.
eig_gap_fac = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 20
monte_carlo = 20
## Compute eigengap
siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
eig_gap = np.round(siga[0]-siga[1],3)
#%% Different Step Size for algorithms
ss_diego = 0.2
ss_cdiego_af = 0.2
ss_cdiego_erdos = 0.2
## The following parameters can be set using filenames in the folder. Do this manually
R_max_f = 15
R_max_af = 1
R_max_erdos = 90
#%% Load All Results
diego_f_cnnc_m = np.load('sim_data/AFvFvErdos_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(ss_diego) + '_mc_' + str(monte_carlo)+ '_R_'+str(R_max_f) + '_FC_'+'.npy')
cdiego_af_cnnc_m = np.load('sim_data/AFvFvErdos_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(ss_cdiego_af) + '_mc_' + str(monte_carlo)+ '_R_'+str(R_max_af) + '_AF_'+'.npy')
cdiego_erdos_cnnc_m = np.load('sim_data/AFvFvErdos_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(ss_cdiego_af) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max_erdos)+ '_Erdos_'+'.npy')
#%% Compute Mean accross all Monte Carlo Simulations
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
# diego_one_n = np.squeeze(np.array(np.mean(diego_one_n_m, axis=0)))
cdiego_af_cnnc = np.squeeze(np.array(np.mean(cdiego_af_cnnc_m, axis=0)))
cdiego_erdos_cnnc = np.squeeze(np.array(np.mean(cdiego_erdos_cnnc_m, axis=0)))
#%% Plot Results
plt.figure()
# Plot the curve
plt.semilogy(diego_f_cnnc, label='DIEGO, StepSize='+str(ss_diego)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_af_cnnc, label='CDIEGO AF, Tc = '+str(R_max_af)+' StepSize='+str(ss_cdiego_af)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_erdos_cnnc, label='CDIEGO Erdos, Tc = '+str(R_max_erdos)+' StepSize='+str(ss_cdiego_erdos)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.title('DIEGO vs CDIEGO with diff graph connectivity with d= '+str(d))
plt.ylabel('Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()
