# Implementation of Consensus-DIstributEd Generalized Oja Algorithm (C-DIEGO)
This project contains the implementation of CDIEGO and DIEGO algorithm in a file [algorithms.py](https://github.com/intarian/CDIEGO/blob/main/algorithms.py).
The algorithm depends on multiple inputs, including designing of weight matrix using $Erd\H{o}s-R\'{e}yni approach, generation of Synthetic data etc. 

We explain the process of each function below and implementatiopca_data_functions.py).

## Population Covariance Matrix Generation $\textbf{\Sigma}}$





## List of Experiments
### 1. Performance Effect of different values of consensus rounds.
This experiment is performed by [1.effect_of_diff_Tc_synthetic_data_mp.py](1.effect_of_diff_Tc_synthetic_data_mp.py) by generating synthetic data 
for three different values of consensus rounds $T_c$.
    1. $T_c = T_{mix}$
    2. $T_c = \log(Nt)$
    3. $T_c = T_{mix} *\frac{3}{2} \log(Nt)$. which we define to be optimal value of consensus rounds.
*Expected Outcome: *The experiment show that for optimal value of $T_c$, there is no gap between the output by DIEGO (which assumes a fully connected network) and 
that of output by C-DIEGO.
The same experiment is also performed using MNIST dataset is provided in [2.effect_of_diff_Tc_mnist_data_mp.py](2.effect_of_diff_Tc_mnist_data_mp.py)
Both experiments using Synthetic and Real dataset are averaged over 50 monte-carlo trials running on multiple cores.

### 2. Convergence rate effect on different values of $\Lambda$ (eigengap)

### 3. Convergence rate effect on different values of N (the number of nodes)

### 4. Convergence rate effect under different topologies