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
tot_iter = 5*1000 # Run for t iterations.
step_size = 0.1
eig_gap_fac = 0.3 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 10
monte_carlo = 10
# # Compute eigengap
siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
eig_gap = np.round(siga[0]-siga[1],3)
#%% Load Data
diego_f_cnnc_Tc = np.load('sim_data_old/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy')
diego_f_cnnc_mp = np.load('sim_data_old/MP_hypothesis_A_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '.npy')
cdiego_round_a_Tc = np.load('sim_data_old/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCa_'+'.npy')
cdiego_round_b_Tc = np.load('sim_data_old/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCb_'+'.npy')
cdiego_round_c_Tc = np.load('sim_data_old/graph_hyp_test_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_NFCc_'+'.npy')
#%% Fetch Data and Consensus Rounds value
diego_f_cnnc = diego_f_cnnc_Tc[:,0]
cdiego_round_a = cdiego_round_a_Tc[:,0]
R_a =  cdiego_round_a_Tc[0,1]
cdiego_round_b = cdiego_round_b_Tc[:,0]
R_b =  cdiego_round_b_Tc[0,1]
cdiego_round_c = cdiego_round_c_Tc[:,0]
R_max =  cdiego_round_c_Tc[0,1]
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
diego_f_cnnc_mp = np.squeeze(np.array(np.mean(diego_f_cnnc_mp, axis=0)))
plt.semilogy(diego_f_cnnc_mp, label='FCD, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_a, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_a),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_b, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_b),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_c, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max),linestyle='dashed',linewidth=2)
plt.title('CDIEGO 2-time scale with d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
x = int(np.round(np.random.random()*100,2)) # Assign random id to file to prevent overwriting
# filename_fig = 'figures/hypothesis_B_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(N)+'_fid_'+str(x)+'.jpg'
# plt.savefig(filename_fig)
plt.show()
