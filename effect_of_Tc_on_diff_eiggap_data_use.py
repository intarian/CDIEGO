#%% Add two different eigengap results to plot on one figure
#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
#%% Load All Results
## Define Parameters:
d = 20 # Set dimensionality of data
tot_iter = 50*1000 # Run for t iterations.
eig_gap_fac_eg1 = 1 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
eig_gap_fac_eg2 = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 40
monte_carlo = 10
step_size = 0.05
## The following parameters can be set using filenames in the folder. Do this manually
R_max = 109
## Compute eigengap
siga_eg1 = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac_eg1)+0.1
eig_gap_eg1 = np.round(siga_eg1[0]-siga_eg1[1],3)
siga_eg2 = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac_eg2)+0.1
eig_gap_eg2 = np.round(siga_eg2[0]-siga_eg2[1],3)
#%% Load All Results
diego_f_cnnc_m_eg1 = np.load('sim_data/eff_ega_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac_eg1) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy')
diego_f_cnnc_m_eg2 = np.load('sim_data/eff_egb_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac_eg2) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy')
cdiego_f_cnnc_m_eg1 = np.load('sim_data/eff_ega_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac_eg1) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max)+'_NFC_'+'.npy')
cdiego_f_cnnc_m_eg2 = np.load('sim_data/eff_egb_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac_eg2) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max)+'_NFC_'+'.npy')
#%% Compute Mean accross all Monte Carlo Simulations eg1
diego_f_cnnc_eg1 = np.squeeze(np.array(np.mean(diego_f_cnnc_m_eg1, axis=0)))
diego_f_cnnc_eg2 = np.squeeze(np.array(np.mean(diego_f_cnnc_m_eg2, axis=0)))
cdiego_f_cnnc_rmax_eg1 = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_eg1, axis=0)))
cdiego_f_cnnc_rmax_eg2 = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_eg2, axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,5))).astype(int)
markers_cdiego = (np.ceil(np.linspace(start_t+5000,end_t-5000,5))).astype(int)
# Plot
plt.semilogy(diego_f_cnnc_eg1[start_t:end_t], label='FC gap = '+str(eig_gap_eg1), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(diego_f_cnnc_eg2[start_t:end_t], label='FC gap = '+str(eig_gap_eg2), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_f_cnnc_rmax_eg1[start_t:end_t], label='NFC gap = '+str(eig_gap_eg1), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_cdiego.tolist())
plt.semilogy(cdiego_f_cnnc_rmax_eg2[start_t:end_t], label='NFC gap = '+str(eig_gap_eg2), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_cdiego.tolist())
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations')
plt.legend()
# plt.savefig('figures/FC_NFC_diff_TC_eg_1_eg_2.eps')
plt.show()