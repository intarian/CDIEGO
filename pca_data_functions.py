import numpy as np

# Convert numpy arrays to 2D array column vector
def Column(x):
    return np.atleast_2d(x).T


## This function generates pop covariance matrix only
def data_cov_mat_gen(d,eig_gap_fac):
    A = np.random.normal(0,1,(d,d))  # Generate random normal matrix of size d x d
    [u,sig,v] = np.linalg.svd(A) # Perform SVD decomposition of dxd
    siga = ((1-0.1)/(np.arange(1, d+1))**eig_gap_fac)+0.1  # Modify eigengap. large gap between \lamda_1 and \lambda_2 and less gap onwards
    At = u@np.diag(np.sqrt(siga))@v # Find \tilde{A} using new eigenvalues
    Sigma = np.matrix(At.T@At) # Find covariance matrix Sigma using \Sigma = \tilde{A}^T \tilde{A}
    eigv = np.linalg.eigh(Sigma) # Find eigenvalue decomposition of eigv
    ev = np.matrix(eigv[-1])  ## Fetch eigen vectors
    pca_vect = ev[:, -1]
    eig_gap = siga[0]-siga[1]
    return pca_vect,Sigma,np.round(eig_gap,3)

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
            # else:
            #     print('Graph Generation Failed. p value of ',p,' is too low. Trying again... \n')
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
#%% The function calculates Tmix (Mixing time of markov chain) according to
# T_mix = max_i=1..N inf(t \in N) norm[W^t ei - 1/N 1^T]<1/2
def Tmix_calc(W,N):
    T_mix_i = np.zeros([N])
    max_tmix = 10
    for i in range(0,N):
        for t in range(0,max_tmix):
            W_e = W**t
            if (np.linalg.norm(W_e[:, i] - (1 / N) * np.ones([N, 1]), 2) < 0.5):
                T_mix_i[i] = t
                break
    return np.max(T_mix_i)