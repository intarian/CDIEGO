#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
#%% Load All Results
## Define Parameters:
d = 20 # Set dimensionality of data
tot_iter = 50*1000 # Run for t iterations.
eig_gap_fac = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 40
monte_carlo = 100
step_size = 0.05
## The following parameters can be set using filenames in the folder. Do this manually
R_a = 7
R_b = 15
# R_c = 60
R_max = 153
# R_d = R_max + 10
## Compute eigengap
siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
eig_gap = np.round(siga[0]-siga[1],3)
#%% Load All Results
diego_f_cnnc_m = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy')
cdiego_f_cnnc_m_a = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_a)+'_NFCa_'+'.npy')
cdiego_f_cnnc_m_b = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_b)+'_NFCb_'+'.npy')
# cdiego_f_cnnc_m_c = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
#         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_c)+'_NFCc_'+'.npy')
cdiego_f_cnnc_m_rmax = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max)+'_NFCmax_'+'.npy')
# cdiego_f_cnnc_m_d = np.load('sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
#         N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_d)+'_NFCd_'+'.npy')
#%% Compute Mean accross all Monte Carlo Simulations
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
cdiego_f_cnnc_a = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_a, axis=0)))
cdiego_f_cnnc_b = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_b, axis=0)))
# cdiego_f_cnnc_c = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_c, axis=0)))
cdiego_f_cnnc_rmax = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_rmax, axis=0)))
# cdiego_f_cnnc_d = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_d, axis=0)))
#%% Plot Results
plt.figure()
from scipy.interpolate import make_interp_spline
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# Plot the curve
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,5))).astype(int)
plt.semilogy(diego_f_cnnc[start_t:end_t], label='FC', linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_f_cnnc_a[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix}$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_f_cnnc_b[start_t:end_t], label='NFC, Tc = '+ ' $\log(Nt)$',linestyle='dashed',linewidth=1,marker='>',markersize=7, markevery=markers_on.tolist())
markers_cdiego = (np.ceil(np.linspace(start_t+5000,end_t-5000,5))).astype(int)
plt.semilogy(cdiego_f_cnnc_rmax[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego.tolist())
# plt.title(')
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.savefig('figures/FC_NFC_diff_TC.eps')
plt.show()
#%% check the convergence error rate
# tt = 50*1000 - 1
# print(cdiego_f_cnnc_d[tt])
# print(cdiego_f_cnnc_rmax[tt])
# print(diego_f_cnnc[tt])
# print(1/np.sqrt(tt*N))

