#%% This is an implementation of Consensus DIEGO Algorithm
# Introduction: DIEGO algorithm as purposed in Arpita's thesis is one-time scale algorithm.
# Here we extend this algorithm to two time scale and showcase that by using multiple communication
# rounds (i.e. nodes exchange their updates multiple times) so that the performance dependence is
# decreased on network topology.
#%% Hypothesis: The dependence of performance on network topology can be reduced by using
# multiple rounds of communications per iteration in Consensus DIEGO (CDIEGO).
# To test this Hypothesis, we first generate different topologies using Erdos-Renyi graphs and test it out
# using against Fully connected Network and run multiple communication rounds to verify that Fully connected
# network still outperforms.
#%% Begin Implementation. Start with Importing Libraries
import numpy as np
import networkx as nx
import pandas as pd
import itertools
from scipy import stats as sps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
import time
from functions import *
#%% Define Parameters
N = 80 # Set No. of Nodes, Each node gets one sample in single iteration
d = 10 # Set dimensionality of data
tot_iter = 1500 # Run for 2000 iterations.
# eta = 0.045 # Set up eta for use in step size
# delta_prime = 0.1
# r_bound = 1
# v_bound = 1e-4
eig_vala = 1
eig_valb = 0.2
eig_gap =  (eig_vala-eig_valb)/d #1 is largest eigval and 0.1 is smallest and there are d eigvalues so lambda_1 - lambda_2 = 0.9/d
step_size = 0.2
#%% Find Beta
# beta = 20*np.max([(r_bound*eta/eig_gap),((v_bound+eig_vala**2)*eta**2)/(eig_gap**2 * np.log(1+delta_prime/100))])
# a_step = eta/(eig_gap*(beta+1))
#%% Data Generation obtain covariance matrix and pca vector
[pca_vect, Sigma] = data_gen_cov_mat(d,eig_vala,eig_valb) # Check eigengap here.
vti = np.random.normal(0,1,(d)) # Generate random eigenvector
vti = Column(vti/np.linalg.norm(vti)) # unit normalize for selection over unit sphere
#%% Generate data and distribute among N nodes. Each nodes gets 1 sample per iteration.
x_samples = np.random.multivariate_normal(np.zeros(d), Sigma,N*tot_iter)
#%% Generate Erdos-Renyi Graph weight matrix
p = 0.5 # probability parameter to use for Erdos-Renyi graph generation.
A = gen_graph_adjacency(N, p)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
W_nf = W_gen_M(N, A)
#%% Hyp 2 Verify
## CDIEGO 2-time scale verify
# For fully connected system
diego_f_cnnc = DIEGO(W_f,1,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
# For connected system
R_nf = [2,3]
diego_nf_cnnc = np.zeros([len(R_nf),tot_iter])
for round in range(0,len(R_nf)):
    diego_nf_cnnc[round,:] = DIEGO(W_nf,R_nf[round],N,d,vti,tot_iter,x_samples,pca_vect,step_size)
#%% Plot Results
marker = np.array(['solid','dashed','dotted','dashdot'])
plt.figure()
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size),linestyle=marker[0],linewidth=2)
for round in range(0,len(R_nf)):
    plt.semilogy(np.ravel(diego_nf_cnnc[round,:]), label='NFC, StepSize='+str(step_size)+', R='+str(R_nf[round])+', p= '+str(p),linestyle=marker[round+1],linewidth=2)
plt.title('DIEGO 2-time scale with N= '+str(N)+' nodes and d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()