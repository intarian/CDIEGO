import numpy as np

# Convert numpy arrays to 2D array column vector
def Column(x):
    return np.atleast_2d(x).T

## This function generates pop covariance matrix and data samples too
# def data_gen(n,d,eig_vala,eig_valb):
#     # Using d=3 and a custom covariance matrix
#     A = np.random.normal(0,1,(d,d))  # Generate random normal matrix of size d x d
#     [u,sig,v] = np.linalg.svd(A) # Perform SVD decomposition of dxd
#     siga = (np.linspace(eig_vala,eig_valb,d)) # Re-define eigenvalues of A
#     At = u@np.diag(np.sqrt(siga))@v # Find \tilde{A} using new eigenvalues
#     Sigma = np.matrix(At.T@At) # Find covariance matrix Sigma using \Sigma = \tilde{A}^T \tilde{A}
#     eigv = np.linalg.eigh(Sigma) # Find eigenvalue decomposition of eigv
#     ev = np.matrix(eigv[-1])  ## Fetch eigen vectors
#     pca_vect = ev
#     x = np.random.multivariate_normal(np.zeros(d), Sigma, n)
#     # pca_vect = Sigma[:,0]
#     return x,pca_vect,Sigma


## This function generates pop covariance matrix only
def data_gen_cov_mat(d,eig_vala,eig_valb):
    # Using d=3 and a custom covariance matrix
    A = np.random.normal(0,1,(d,d))  # Generate random normal matrix of size d x d
    [u,sig,v] = np.linalg.svd(A) # Perform SVD decomposition of dxd
    siga = (np.linspace(eig_vala,eig_valb,d)) # Re-define eigenvalues of A
    At = u@np.diag(np.sqrt(siga))@v # Find \tilde{A} using new eigenvalues
    Sigma = np.matrix(At.T@At) # Find covariance matrix Sigma using \Sigma = \tilde{A}^T \tilde{A}
    eigv = np.linalg.eigh(Sigma) # Find eigenvalue decomposition of eigv
    ev = np.matrix(eigv[-1])  ## Fetch eigen vectors
    pca_vect = ev[:, -1]
    # x = np.random.multivariate_normal(np.zeros(d), Sigma, n)
    # pca_vect = Sigma[:,0]
    return pca_vect,Sigma

# Generate Random connected Graph. Use laplacian matrix 2nd eigenvalue to check for connectedness.
# The Erdos-Renyi Method
def gen_graph_adjacency(N,p):
    while (1):
        if (N > 1):
            # Create Upper Triangular Matrix to create Adjacecny Matrix which is symmetric
            U = np.zeros((N, N)) # Create an empty NxN matrix
            for r in range(0, N):
                for q in range(0, N):
                    if (r != q):
                        if (np.random.random() < p):
                            U[r, q] = 1  # connect edge between node r and q if the above condition satisfies
            # Once connection found, only upper triangular portion will be used and lower triangular will be discarded.
            U = np.triu(U) # This computation can be relatively reduced by adjusting for loops
            # For Adjacency matrix: As its a symmetric with 0's on diagonal i.e. node has 0 connection to itself
            A = (U.T + U) * (np.ones(N) - np.eye(N))
            ## Create Degree Matrix for Laplacian
            D = np.zeros((N, N))
            for i in range(N):
                D[i, i] = np.sum(A[i, :])
            ## Create Laplacian Matrix from Degree Matrix and Adjacency Matrix
            L = D - A
            #     print(L)
            ## Use eigenvalues of Laplacian Matrix, If 2nd eigenvalue is 0 its not connected graph and regenerate Adjacency matrix
            e = np.round(np.linalg.eigvalsh(L),
                         3)  # use eigenvalsh from scipy to find eigenvalues of symmetric L matrix
            if (e[1] > 0):  # check if second eigenvalue (ascending order) is greater than zero, then it means graph is connected
                return np.matrix(A)
            else:
                print('Graph Generation Failed. p value of ',p,' is too low. Trying again... \n')
        else:
            return np.matrix([1])


def W_gen(N,A):
    if (N>1):
        D = np.zeros((N, N))
        for i in range(N):
            D[i, i] = np.sum(A[i, :])
        ## Create Laplacian Matrix from Degree Matrix and Adjacency Matrix
        L = D - A
        if (N > 1):
            e = np.linalg.eigvalsh(L)
            alpha = 2 / (e[-1] + e[1])
            W = np.eye(N) - (alpha * L)
        return W
    else:
        return np.matrix([1])

def W_gen_M(N,A):
    # Calculate Laplacian Matrix using Adjacency matrix
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = np.sum(A[i, :])
    ## Create Laplacian Matrix from Degree Matrix and Adjacency Matrix
    L = D - A
    ## Calculate Incidence Matrix
    edges = (np.sum(((np.sum(A)) / 2), dtype=np.int32));  # finds the total no. of edges using adjacency matrix
    Ai = np.zeros((N, edges));
    Au = np.triu(A);
    e = 0;
    for j in range(0, N):
        for k in range(0, N):
            if (Au[j, k] == 1):
                Ai[j, e] = 1;
                Ai[k, e] = -1;
                e += 1;
    ## verify incidence matrix using laplacian generated previously
    if (L == (Ai @ Ai.T)).all():
        # print('The laplacian matrix generated using incidence matrix verifies with the original laplacian matrix \n')
        # print('The incidence matrix is: \n \n', Ai)

        ## Find weights on the edges of nodes
        edg_w = np.zeros((edges))
        eps = 1
        for i in range(0, edges):
            node_e = (np.nonzero(Ai[:, i]))[0]
            di = np.sum((A[node_e[0], :]), dtype=np.int32)
            dj = np.sum((A[node_e[1], :]), dtype=np.int32)
            w_e = 1 / (np.maximum(di, dj) + eps);
            edg_w[i] = w_e

        ## Calculate Weight Matrix (assign edge weights by using upper triangular portion of Adjacency Matrix)
        W_M = np.zeros((N, N));
        e = 0;
        for i in range(0, N):
            for j in range(0, N):
                if (Au[i, j] == 1):
                    W_M[i, j] = edg_w[e];
                    W_M[j, i] = edg_w[e];
                    e += 1;
        ## assign weights on node itself (to make W doubly stochastic)
        for i in range(0, N):
            W_M[i, i] = 1 - np.sum(W_M[i, :])
        # print('The weight matrix using Metropolis-Hasting method W_M is calculated as: \n \n', W_M)
        return np.matrix(W_M)

#%% Begin DIEGO Algorithm
# Introduction: This is a function of DIEGO algorithm. Implements 1 and 2 time scale algorithm.
# if R=1, the algorithm is simply 1 time scale and if R>2 this becomes 2 time scale.
# The algorithm takes parameters like:
# step_size and decays using piecewise decay,
# R: no of comm rounds
# W: Weight matrix of the graph
# x_samples: data samples generated first
# N: no of nodes
# d: Dimensionality of data
# vti: Initial vector to distribute among N nodes for estimation
# tot_iter: Total no. of iterations to run the algorithm
# step_c: Piecewise linear decaying factor
# pca_vect: Original PCA vect to compare error.
def DIEGO(W,N,d,vti,tot_iter,x_samples,pca_vect,step_size):
    ## Comm rounds are computed in each iterations as R = log(N*t) as t increases R increases
    # Data samples distribute among N nodes
    x_samples_n = np.reshape(x_samples, (N, tot_iter, d))
    # Now we initialize all N nodes with initial eigenvector vti
    v_n = np.matrix(np.repeat(vti, N, axis=1))  # Initialize eigenvector estimate with vti at all nodes
    err_n = np.zeros([N, tot_iter])  # store error across all N nodes in each iteration
    # Begin loop to draw sample from each node and then compute their estimates and updates using W
    for sample in range(0, tot_iter):
        # gamma_ext = eta/(eig_gap*(beta+sample))
        gamma_ext = step_size / (1 + sample)
        # gamma_ext = np.ceil(((sample + 1)) / 10) # piecewise linear decay
        upd_n = np.matrix(np.zeros([d, N]))  # Store update values across all nodes for each sample
        # Begin loop to take sample from each node and update eigenvector estimate
        ## Calculate Communication Rounds
        R = int(np.ceil(np.log(N*(sample+1)))) # sample+1 as sample starts from 0
        for i_n in range(0, N):
            x_n = Column(x_samples_n[i_n, sample, :])  # Draw sample across node i_n
            vn = v_n[:, i_n]  # Draw current estimate of eigenvector across node i_n
            upd_n[:, i_n] = (x_n @ x_n.T @ vn)
        # Exchange information among all nodes. Use value of R to achieve multiple rounds.
        if (N > 1 and R>0): # use rounds only for multiple nodes. If R=0, we don't use any communication between nodes
            for round in range(0,R):
                upd_n = upd_n @ W # Regardless of W. If fully connected it won't change upd_n if R>0. Though increases comp cost
        # Update eigenvector estimate
        if ((W == np.matrix(1 / N * (np.ones((N, 1)) @ np.ones((N, 1)).T))).all()):
            v_n = v_n + gamma_ext * N*upd_n
        else:
            if (R>0):
                W_e1 = (W**R)
                W_e1 = W_e1[:,0]+0.0001
            else:
                W_e1 = W[:,0]+0.0001
            for i_n in range(0,N):
                v_n[:,i_n] = v_n[:,i_n] + gamma_ext * (upd_n[:,i_n])/(W_e1[i_n]).item()
        # Normalize the estimate to make unit norm
        for i_n in range(0, N):
            v_n[:, i_n] = v_n[:, i_n] / np.linalg.norm(v_n[:, i_n], 2)
        # Compute Error for each iteration
        for i_n in range(0, N):
            err_n[i_n, sample] = 1 - ((v_n[:, i_n].T @ pca_vect) ** 2) / (
                        np.linalg.norm(v_n[:, i_n], 2) ** 2)
    # Compute mean error across all nodes at each iteration. For fully connected this is error across any node.
    mean_err = np.squeeze(np.array(np.mean(err_n, axis=0)))
    return mean_err