#%% Import Libraries
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = False
## Load Parameters & Data:
param = np.load('sim_data/3.eff_N_synthetic_param_mp.npy')
## Load data of each N
cdiego_m_Na = np.load('sim_data/3.eff_Na_synthetic_data_mp.npy')
cdiego_m_Nb = np.load('sim_data/3.eff_Nb_synthetic_data_mp.npy')
cdiego_m_Nc = np.load('sim_data/3.eff_Nc_synthetic_data_mp.npy')
## Define Parameters using Loaded file
d = int(param[0,0]) # Load dimensionality of data
tot_iter = int(param[1,0]) # Load no. of iterations
monte_carlo = param[2,0] # Load no. of monte carlo runs
eigen_gap = param[3,0] # Load eigengap
## Load stepsize for each N
step_size_Na = param[4,0]
step_size_Nb = param[5,0]
step_size_Nc = param[6,0]
## Load data of diff N
Na = param[7,0]
Nb = param[8,0]
Nc = param[9,0]
#%% Compute Mean accross all Monte Carlo Simulations
cdiego_m_Na = np.squeeze(np.array(np.mean(cdiego_m_Na, axis=0)))
cdiego_m_Nb = np.squeeze(np.array(np.mean(cdiego_m_Nb, axis=0)))
cdiego_m_Nc = np.squeeze(np.array(np.mean(cdiego_m_Nc, axis=0)))
#%% Plot Results
plt.figure()
start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,10))).astype(int)
plt.plot(cdiego_m_Na[start_t:end_t], label='FC, $N = $'+str(Na)+', alpha = '+str(step_size_Na), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.plot(cdiego_m_Nb[start_t:end_t], label='FC, $N = $'+str(Nb)+', alpha = '+str(step_size_Nb), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.plot(cdiego_m_Nc[start_t:end_t], label='FC, $N = $'+str(Nc)+', alpha = '+str(step_size_Nc), linestyle='solid',linewidth=1,marker='^',markersize=4, markevery=markers_on.tolist())
plt.yscale("log")
plt.xscale("log")
plt.ylabel('Average Error')
plt.xlabel('No. of Iterations (t)')
plt.legend()
plt.savefig('images/effc_diff_N.eps')
plt.show()
#%% Test the scaling
start_t = tot_iter-10
end_t = tot_iter
print(cdiego_m_Nb[start_t:end_t]/cdiego_m_Nc[start_t:end_t])
print(np.sqrt(Nc/Nb))