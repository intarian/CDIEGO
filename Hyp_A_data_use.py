#%% This file uses saved data to play with results from Hyp A
import numpy as np
import networkx as nx
# import pandas as pd
# import itertools
# from scipy import stats as sps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
# import time
from functions import *
#%% Load saved results
d = 20 # Set dimensionality of data
tot_iter = 10000 # Run for t iterations.
step_size = 0.2
eig_gap_fac = 0.23 #Controls the factor of eigengap. See function data_gen_cov_mat in functions.py
Node_count = [20,30,40]

monte_carlo = 50
filename_data = 'sim_data/hypothesis_A_iter_count_'+str(tot_iter)+'_dimdata_'+str(d)+'_nodes_'+str(Node_count)+'.npy'
diego_f_cnnc_mn = np.load(filename_data)
#%% Check scaling by sqrt(N2/N1)
a = 2
b = 2
diego_f_cnnc_ma = np.squeeze(np.array(np.mean(diego_f_cnnc_mn[a,:,-3000:], axis=0)))
diego_f_cnnc_mb = np.squeeze(np.array(np.mean(diego_f_cnnc_mn[b,:,-3000:], axis=0)))
x = diego_f_cnnc_ma/diego_f_cnnc_mb
print(np.mean(x))
print((Node_count[b]/Node_count[a]))
print(np.sqrt(Node_count[b]/Node_count[a]))
plt.figure()
plt.semilogy(diego_f_cnnc_ma)
plt.semilogy(diego_f_cnnc_mb)
plt.show()
plt.figure()
plt.plot(x)
plt.show()
#%%
