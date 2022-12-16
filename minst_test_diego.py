#%% This code uses MINST Dataset and see the best T_c for
#%% Import Libraries etc
import numpy as np
from matplotlib import pyplot as plt
from functions import *
#%% Begin define DIEGO Algorithm
def DIEGO_MINST(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
    # Data samples distribute among N nodes
    x_samples_n = np.reshape(x_samples, (N, tot_iter, d))
    # Now we initialize all N nodes with initial eigenvector vti
    v_n = np.matrix(np.repeat(vti, N, axis=1))  # Initialize eigenvector estimate with vti at all nodes
    err_n = np.zeros([N, tot_iter])  # store error across all N nodes in each iteration
    # Begin loop to draw sample from each node and then compute their estimates and updates using W
    for sample in range(0, tot_iter):
        gamma_ext = step_size / (1 + sample)
        # gamma_ext = np.ceil(((sample + 1)) / 10) # piecewise linear decay
        upd_n = np.matrix(np.zeros([d, N]))  # Store update values across all nodes for each sample
        # Begin loop to take sample from each node and update eigenvector estimate
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        upd_n = upd_n @ W
        # Update eigenvector estimate
        v_n = v_n + gamma_ext * N*upd_n # for fully connected graph. scale by 1/[W^*e_1]i i.e scale by N
        # Normalize the estimate to make unit norm
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] / np.linalg.norm(v_n[:, i_n], 2)
        # Compute Error for each iteration
        for i_n in range(0, N):
            err_n[i_n, sample] = np.sqrt(1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2))
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    return mean_err
#%% Begin define CDIEGO Algorithm
def CDIEGO_MINST(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size,Tc):
    # Data samples distribute among N nodes
    x_samples_n = np.reshape(x_samples, (N, tot_iter, d))
    # Now we initialize all N nodes with initial eigenvector vti
    v_n = np.matrix(np.repeat(vti, N, axis=1))  # Initialize eigenvector estimate with vti at all nodes
    err_n = np.zeros([N, tot_iter])  # store error across all N nodes in each iteration
    # Begin loop to draw sample from each node and then compute their estimates and updates using W
    for sample in range(0, tot_iter):
        gamma_ext = step_size / (1 + sample)
        upd_n = np.matrix(np.zeros([d, N]))  # Store update values across all nodes for each sample
        # Begin loop to take sample from each node and update eigenvector estimate
        ## Calculate Communication Rounds
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        for round in range(0,Tc): # Run for multiple communication rounds.
            upd_n = upd_n @ W
        # After updating upd_n data. The next thing is to scale it by W^R instead of N.
        W_e1 = (W ** Tc)
        if (W_e1[:, 0] == 0).any():
            print(W)
            print('Failed to run. Weight matrix has a zero. Try increasing Tmix')
            break
        else:
            W_e1 = W_e1[:, 0] # + 1e-100 # as we are taking first column only. We need to add small eps to avoid scaling by 1/0
        # Update eigenvector estimate
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] + gamma_ext * (upd_n[:, i_n]) / (W_e1[i_n]).item()
        # Normalize the estimate to make unit norm
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] / np.linalg.norm(v_n[:, i_n], 2)
        # Compute Error for each iteration
        for i_n in range(0, N):
            err_n[i_n, sample] = np.sqrt(1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2))
    max_err = np.squeeze(np.array(np.max(err_n, axis=0)))
    return max_err # Give out max error among all nodes from q_1 instead of mean error.

#%% Begin Main Implementation
## Load MINST DATASET
import pickle
import gzip
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
# Load the dataset
(train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(
    gzip.open('MINST_dataset/mnist_py3k.pkl.gz', 'rb'))
train_inputs = np.concatenate((train_inputs, valid_inputs))
data = train_inputs.transpose()  # dimensionxnum_samples
#%% Concatenate data to create more samples and perform a random shuffle
np.random.seed(1)
data_concatenated = np.zeros((data.shape[0],data.shape[1]*20))
for i in range(0,20):
    data_concatenated[:,i*data.shape[1]:(i+1)*data.shape[1]] = data
## Perform Random shuffle
np.random.shuffle(data_concatenated)
#%% Load True eigenvectors for MINST
with open("MINST_dataset/EV_mnist.pickle", 'rb') as f:
    X1 = pickle.load(f)
pca_vect = X1[:, 0]
#%% Load Additional Parameters
N = 10
monte_carlo = 10
p = 0.1 # Parameter for Erdos-Reyni Convergence
step_size = 10
tot_data_samples = data_concatenated.shape[1]
d = data_concatenated.shape[0]
tot_iter = int(tot_data_samples/N)
#%% Create empty array
diego_f_cnnc_m = np.zeros((monte_carlo,tot_iter))
# Each a b c arrays created below will save different values of Tc
cdiego_f_cnnc_m_a = np.zeros((monte_carlo,tot_iter))
# %% Generate random initial vector to be same for all monte-carlo simulations
vti = np.random.normal(0, 1, (d))  # Generate random eigenvector
vti = Column(vti / np.linalg.norm(vti))  # unit normalize for selection over unit sphere
#%% Generate Fully Connected Graph for Hyp A (which assumes Federated Learning)
W_f = np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))
for mon in range(0,monte_carlo):
    print('Currently Processing Nodes: ', N, ' of Monte Carlo: ',mon,'\n')
    np.random.shuffle(data_concatenated)
    #%% Run DIEGO Algorithm
    diego_f_cnnc_m[mon,:] = DIEGO_MINST(W_f,N,d,vti,tot_iter,data_concatenated,pca_vect,step_size)
#%% Save The data
np.save('sim_data/MINST_DIEGO_iter_count_' + str(tot_iter) + '_dimdata_' + str(d) + '_nodes_' + str(
        N) + '_ss_' + str(step_size) + '_mc_' + str(monte_carlo) + '_FC_'+'.npy', diego_f_cnnc_m)
#%% Compute Mean and Max of errors
diego_f_cnnc = np.squeeze(np.array(np.mean(diego_f_cnnc_m, axis=0)))
#%% Plot the results
plt.figure()
plt.semilogy(diego_f_cnnc, label='DIEGO, StepSize='+str(step_size)+', N= '+str(N),linestyle='dashed',linewidth=2)
plt.title('DIEGO vs CDIEOG with diff stepsize with d= '+str(d))
plt.ylabel('Error')
plt.xlabel('No. of Iterations')
plt.show()
