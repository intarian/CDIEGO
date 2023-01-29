#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
## Load Parameters & Data:
param = np.load('sim_data/4.eff_topologies_synthetic_param_mp.npy')
cdiego_m = np.load('sim_data/4.eff_topologies_synthetic_data_mp.npy')
## Define Parameters using Loaded file
d = int(param[0,0]) # Load dimensionality of data
tot_iter = int(param[1,0]) # Load no. of iterations
monte_carlo = param[2,0] # Load no. of monte carlo runs
N = param[3,0] # Load No of Nodes
step_size = param[4,0] # Load step size
eigen_gap = param[5,0] # Load eigengap
## Load data of Consensus Rounds
T_opt = param[6,0]
#%% Compute Mean accross all Monte Carlo Simulations
cdiego_m_fgd_T_1 = np.squeeze(np.array(np.mean(cdiego_m[0,:,:], axis=0)))
cdiego_m_fgd_T_opt = np.squeeze(np.array(np.mean(cdiego_m[1,:,:], axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,10))).astype(int)
markers_cdiego_opt = (np.ceil(np.linspace(start_t+10,end_t-10,20))).astype(int)
plt.semilogy(cdiego_m_fgd_T_1[start_t:end_t], label='FGD, Tc = 1',linestyle='dashed',linewidth=1,marker='>',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_fgd_T_opt[start_t:end_t], label='FGD, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$ = '+str(T_opt),linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego_opt.tolist())
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations (t)')
plt.legend()
# plt.savefig('figures/FC_NFC_diff_TC.eps')
plt.show()