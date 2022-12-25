#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
## Load Parameters & Data:
param = np.load('sim_data/2.eff_eiggap_synthetic_params_mp.npy')
cdiego_m_ega = np.load('sim_data/2.eff_eiggap_ega_synthetic_data_mp.npy')
cdiego_m_egb = np.load('sim_data/2.eff_eiggap_egb_synthetic_data_mp.npy')
cdiego_m_egc = np.load('sim_data/2.eff_eiggap_egc_synthetic_data_mp.npy')
## Define Parameters using Loaded file
d = int(param[0,0]) # Load dimensionality of data
tot_iter = int(param[1,0]) # Load no. of iterations
N = param[2,0] # Load No of Nodes
monte_carlo = param[3,0] # Load no. of monte carlo runs
p = param[4,0] # Load Parameter for Erdos-Reyni Convergence
step_size = param[5,0] # Load step size
## Load values of eigengap
eigen_gap_a = param[6,0]
eigen_gap_b = param[7,0]
eigen_gap_c = param[8,0]
## Load data of Consensus Rounds
T_opt = param[9,0]
#%% Compute Mean accross all Monte Carlo Simulations
cdiego_m_FC_ega = np.squeeze(np.array(np.mean(cdiego_m_ega[0, :, :], axis=0)))
cdiego_m_T_opt_ega = np.squeeze(np.array(np.mean(cdiego_m_ega[1, :, :], axis=0)))
cdiego_m_FC_egb = np.squeeze(np.array(np.mean(cdiego_m_egb[0, :, :], axis=0)))
cdiego_m_T_opt_egb = np.squeeze(np.array(np.mean(cdiego_m_egb[1, :, :], axis=0)))
cdiego_m_FC_egc = np.squeeze(np.array(np.mean(cdiego_m_egc[0, :, :], axis=0)))
cdiego_m_T_opt_egc = np.squeeze(np.array(np.mean(cdiego_m_egc[1, :, :], axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,10))).astype(int)
markers_cdiego_opt = (np.ceil(np.linspace(start_t+10,end_t-10,20))).astype(int)
plt.semilogy(cdiego_m_FC_ega[start_t:end_t], label='FC, $\Lambda: $'+str(eigen_gap_a), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_T_opt_ega[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$: '+str(T_opt)+', $\Lambda: $'+str(eigen_gap_a),linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego_opt.tolist())
plt.semilogy(cdiego_m_FC_egb[start_t:end_t], label='FC, $\Lambda: $'+str(eigen_gap_b), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_T_opt_egb[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$: '+str(T_opt)+', $\Lambda: $'+str(eigen_gap_b),linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego_opt.tolist())
plt.semilogy(cdiego_m_FC_egc[start_t:end_t], label='FC, $\Lambda: $'+str(eigen_gap_c), linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_T_opt_egc[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$: '+str(T_opt)+', $\Lambda: $'+str(eigen_gap_c),linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego_opt.tolist())
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations (t)')
plt.legend()
# plt.savefig('figures/FC_NFC_diff_TC.eps')
plt.show()