#%% This files proves Hypothesis A which is as follows:
#%% Hypothesis A: This hypothesis shows that the eigenvector estimate \hat{v} from strongly connected graph
# and v_i which is an estimate of eigenvector at node i bounds the consensus error in CDIEGO algorithm as proved. i.e.
# ||\hat{v} - v_i || <= \frac{\delta N r \eta}{(\lambda_1 - \lambda_2)} (\frac{t}{beta} + \hat{\gamma})
# Basically, the above result shows that the for $\delta = (\frac{\eps}{Nt})^(3/2)$ the consensus error has a rate
# of $\O(1/sqrt{Nt}$ and Tc = \O(T_mix 3/2 log(eps/Nt)).
#%% For simplicity, we choose step size of \alpha_t = 1/t. which makes our bound comes out to be
# ||\hat{v} - v_i || <= \frac{\delta N r }{(\lambda_1 - \lambda_2)} (\frac{t}{beta} + \hat{\gamma})
#%% Begin Implementation. Start with Importing Libraries
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
#%% Define Parameters
N = 10 # Set No. of Nodes, Each node gets one sample in single iteration
d = 8 # Set dimensionality of data
tot_iter = 1000 # Run for 2000 iterations.
# eta = 0.045 # Set up eta for use in step size
# delta_prime = 0.1
# r_bound = 1
# v_bound = 1e-4
eig_vala = 1
eig_valb = 0.1
eig_gap =  (eig_vala-eig_valb)/d #1 is largest eigval and 0.1 is smallest and there are d eigvalues so lambda_1 - lambda_2 = 0.9/d
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
p = 0.1 # probability parameter to use for Erdos-Renyi graph generation.
A = gen_graph_adjacency(N, p)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
W_nf = W_gen_M(N, A)
#%% Hyp 2 Verify
## CDIEGO 2-time scale verify
# For fully connected system
step_size = 0.1
step_size_nf = 0.1
diego_f_cnnc = DIEGO(W_f,N,d,vti,tot_iter,x_samples,pca_vect,step_size)
diego_nf_cnnc = DIEGO(W_nf,N,d,vti,tot_iter,x_samples,pca_vect,step_size_nf)
#%% Plot Results
marker = np.array(['solid','dashed','dotted','dashdot'])
plt.figure()
plt.semilogy(diego_f_cnnc, label='FC, StepSize='+str(step_size),linestyle=marker[0],linewidth=2)
plt.semilogy(diego_nf_cnnc, label='NFC, StepSize='+str(step_size)+', p= '+str(p),linestyle=marker[1],linewidth=2)
plt.title('DIEGO 2-time scale with N= '+str(N)+' nodes and d= '+str(d))
plt.ylabel('Mean Error')
plt.xlabel('No. of Iterations')
plt.legend()
plt.show()