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
monte_carlo = 50
filename = 'sim_data/MP_DIEGO_MNIST_amarel_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo)+'.npy'
#%% Load Data
diego_f_cnnc_minst = np.load(filename)
#%% Fetch Data and Consensus Rounds value
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_minst, axis=0)))
#%% Plot Results
plt.figure()
# plt.semilogy(diego_scaling_true, label='Scaling by O(1/sqrt(Nt))',linestyle='solid',linewidth=2)
plt.semilogy(diego_f_cnnc, label='MNIST, StepSize='+str(step_size)+', N= '+str(N),linestyle='dashed',linewidth=2)
# diego_f_cnnc_mp = np.squeeze(np.array(np.mean(diego_f_cnnc_mp, axis=0)))
# plt.semilogy(diego_f_cnnc_mp, label='FCD, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_a, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_a),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_b, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_b),linestyle='dashed',linewidth=2)
# plt.semilogy(cdiego_round_c, label='NFC, StepSize='+str(step_size)+', N= '+str(N)+', gap= '+str(eig_gap)+', R= '+str(R_max),linestyle='dashed',linewidth=2)
plt.title('DIEGO with MNIST with d= '+str(d)+' MC ='+str(monte_carlo))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
# x = int(np.round(np.random.random()*100,2)) # Assign random id to file to prevent overwriting
# filename_fig = 'figures/hypothesis_B_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(N)+'_fid_'+str(x)+'.jpg'
# plt.savefig(filename_fig)
plt.show()
