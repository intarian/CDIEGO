#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
## Load Parameters & Data:
param = np.load('sim_data/3.eff_N_DvsCD_param_mp.npy')
cdiego_m = np.load('sim_data/3.eff_N_DvsCD_data_mp.npy')
## Define Parameters using Loaded file
d = int(param[0,0]) # Load dimensionality of data
tot_iter = int(param[1,0]) # Load no. of iterations
monte_carlo = param[2,0] # Load no. of monte carlo runs
step_size = param[3,0] # Load step size
eigen_gap = param[4,0] # Load eigengap a
## Load data of diff N
Na = param[5,0]
Nb = param[6,0]
Nc = param[7,0]
#%% Compute Mean accross all Monte Carlo Simulations
cdiego_m_Na = np.squeeze(np.array(np.mean(cdiego_m[0,:,:], axis=0)))
cdiego_m_Nb = np.squeeze(np.array(np.mean(cdiego_m[1,:,:], axis=0)))
cdiego_m_Nc = np.squeeze(np.array(np.mean(cdiego_m[2,:,:], axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,10))).astype(int)
plt.semilogy(cdiego_m_Na[start_t:end_t], label='FC, $N = $'+str(Na), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_Nb[start_t:end_t], label='FC, $N = $'+str(Nb), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.semilogy(cdiego_m_Nc[start_t:end_t], label='FC, $N = $'+str(Nc), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations')
plt.legend()
# plt.savefig('figures/FC_NFC_diff_TC.eps')
plt.show()
#%% Test the scaling
start_t = 40000
end_t = tot_iter
np.polyfit(np.arange(start_t,end_t),cdiego_m_Na[start_t:end_t]/cdiego_m_Nb[start_t:end_t],1)