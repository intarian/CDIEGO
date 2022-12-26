#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
## Load Parameters & Data:
param = np.load('sim_data/1a.eff_Tc_mnist_params_mp.npy')
cdiego_m = np.load('sim_data/1a.eff_Tc_mnist_data_mp.npy')
## Define Parameters using Loaded file
d = int(param[0,0]) # Load dimensionality of data
N = param[1,0] # Load No of Nodes
tot_iter = int(param[2,0]) # Load no. of iterations
monte_carlo = param[3,0] # Load no. of monte carlo runs
p = param[4,0] # Load Parameter for Erdos-Reyni Convergence
step_size = param[5,0] # Load step size
## Load data of Consensus Rounds
T_a = param[6,0]
T_b = param[7,0]
T_opt = param[8,0]
#%% Compute Mean accross all Monte Carlo Simulations
cdiego_m_FC = np.squeeze(np.array(np.mean(cdiego_m[0,:,:], axis=0)))
cdiego_m_Ta = np.squeeze(np.array(np.mean(cdiego_m[1,:,:], axis=0)))
cdiego_m_Tb = np.squeeze(np.array(np.mean(cdiego_m[2,:,:], axis=0)))
cdiego_m_T_opt = np.squeeze(np.array(np.mean(cdiego_m[3,:,:], axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,10))).astype(int)
markers_cdiego_opt = (np.ceil(np.linspace(start_t+10,end_t-10,20))).astype(int)
plt.semilogy(cdiego_m_FC[start_t:end_t], label='FC', linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_Ta[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix}$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_Tb[start_t:end_t], label='NFC, Tc = '+ ' $\log(Nt)$',linestyle='dashed',linewidth=1,marker='>',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_T_opt[start_t:end_t], label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego_opt.tolist())
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations (t)')
plt.legend()
# plt.savefig('figures/FC_NFC_diff_TC.eps')
plt.show()