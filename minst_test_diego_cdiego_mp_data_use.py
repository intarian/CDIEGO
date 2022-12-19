#%% This file uses saved data to play with results from MINST DATA
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
d = 784 # Set dimensionality of data
tot_iter = 1*6000 # Run for t iterations.
step_size = 0.01
N = 10
monte_carlo = 30
filename = 'sim_data/MP_DIEGO_CDIEGO_MNIST_amarel_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo)+'.npy'
#%% Load Data
diego_cdiego_mnist = np.load(filename)
#%% Compute Mean of Monte-Carlo Results
diego_result = np.squeeze(np.array(np.mean(diego_cdiego_mnist[0,:,:], axis=0)))
cdiego_result_Ra = np.squeeze(np.array(np.mean(diego_cdiego_mnist[1,:,:], axis=0)))
cdiego_result_Rmax = np.squeeze(np.array(np.mean(diego_cdiego_mnist[2,:,:], axis=0)))
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
# plt.semilogy(diego_result, label='MNIST DIEGO, StepSize='+str(step_size)+', N= '+str(N),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_result_Ra, label='MNIST CDIEGO Ra, StepSize='+str(step_size)+', N= '+str(N),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_result_Rmax, label='MNIST CDIEGO Rmax, StepSize='+str(step_size)+', N= '+str(N),linestyle='dashed',linewidth=2)
# diego_f_cnnc_mp = np.squeeze(np.array(np.mean(diego_f_cnnc_mp, axis=0)))



start_t = 0
end_t = tot_iter
markers_on = (np.ceil(np.linspace(start_t+1,end_t-1,5))).astype(int)
plt.semilogy(diego_result, label='FC', linestyle='solid',linewidth=1,marker='^',markersize=7, markevery=markers_on.tolist())
plt.semilogy(cdiego_result_Ra, label='NFC, Tc = '+ ' $\log(Nt)$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_on.tolist())
# plt.semilogy(cdiego_result_Rmax, label='WC, Tc = '+ ' $T_{mix} 3/2 \log(Nt)$',linestyle='dashed',linewidth=1,marker='>',markersize=7, markevery=markers_on.tolist())
markers_cdiego = (np.ceil(np.linspace(start_t+1000,end_t-1000,5))).astype(int)
plt.semilogy(cdiego_result_Rmax, label='NFC, Tc = '+ ' $T_{mix} (3/2) \log(Nt)$',linestyle='dashed',linewidth=1,marker='o',markersize=7, markevery=markers_cdiego.tolist())
# plt.title(')
plt.ylabel('Max Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.savefig('figures/MNIST_FC_NFC_diff_TC.eps')
plt.show()
