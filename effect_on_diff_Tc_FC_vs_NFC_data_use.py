#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
#%% Load All Results
## Define Parameters:
d = 4 # Set dimensionality of data
tot_iter = 50*1000 # Run for t iterations.
eig_gap_fac = 2 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
N = 10
monte_carlo = 100
step_size = 0.2
## The following parameters can be set using filenames in the folder. Do this manually
R_a = 4
R_b = 14
R_c = 53
R_max = 79
R_d = R_max + 10
## Compute eigengap
siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1
eig_gap = np.round(siga[0]-siga[1],3)
#%% Load All Results
diego_f_cnnc_m = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy')
cdiego_f_cnnc_m_a = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_a)+'_NFCa_'+'.npy')
cdiego_f_cnnc_m_b = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_b)+'_NFCb_'+'.npy')
cdiego_f_cnnc_m_c = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_c)+'_NFCc_'+'.npy')
cdiego_f_cnnc_m_rmax = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_max)+'_NFCmax_'+'.npy')
cdiego_f_cnnc_m_d = np.load('C:/Users/zulqa/PycharmProjects/SigProcMP/sim_data/dff_Tc_DvsCD_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_eg_' + str(eig_gap_fac) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_R_'+str(R_d)+'_NFCd_'+'.npy')
#%% Compute Mean accross all Monte Carlo Simulations
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
cdiego_f_cnnc_a = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_a, axis=0)))
cdiego_f_cnnc_b = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_b, axis=0)))
cdiego_f_cnnc_c = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_c, axis=0)))
cdiego_f_cnnc_rmax = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_rmax, axis=0)))
cdiego_f_cnnc_d = np.squeeze(np.array(np.mean(cdiego_f_cnnc_m_d, axis=0)))
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# Plot the curve
start_t = 0
end_t = 1000
plt.semilogy(diego_f_cnnc[start_t:end_t], label='DIEGO, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=1)
# plt.semilogy(cdiego_f_cnnc_a[start_t:end_t], label='CDIEGO, Tc = '+str(R_a) + ' ($T_{mix}$)' + ' StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_f_cnnc_b[start_t:end_t], label='CDIEGO, Tc = '+str(R_b)+ ' ($\log(Nt)$)'+' StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
plt.semilogy(cdiego_f_cnnc_c[start_t:end_t], label='CDIEGO, Tc = '+str(R_c)+ ' ($T_{mix} \log(Nt)$)'+' StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='solid',linewidth=2)
plt.semilogy(cdiego_f_cnnc_rmax[start_t:end_t], label='CDIEGO, Tc = '+str(R_max)+ ' ($T_{mix} 3/2 \log(Nt)$)'+' StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=1)
plt.semilogy(cdiego_f_cnnc_d[start_t:end_t], label='CDIEGO, Tc = '+str(R_d)+ ' ($T_{mix} 3/2 \log(Nt) + 10$)'+' StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='solid',linewidth=1)
plt.title('DIEGO vs CDIEOG with diff stepsize with d= '+str(d))
plt.ylabel('Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()
#%% check the convergence error rate
tt = 50*1000 - 1
print(cdiego_f_cnnc_d[tt])
print(cdiego_f_cnnc_rmax[tt])
print(diego_f_cnnc[tt])
print(1/np.sqrt(tt*N))

