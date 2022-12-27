# Implementation of Consensus-DIstributEd Generalized Oja Algorithm (C-DIEGO)
This project contains the implementation of C-DIEGO algorithm in a file [algorithms.py](https://github.com/intarian/CDIEGO/blob/main/algorithms.py).  
The algorithm depends on multiple inputs, including designing of weight matrix using $Erd\H{o}s-R\'{e}yni$ approach, generation of Synthetic data etc.  

We explain the process of each function below and implementation in [pca_data_functions.py](https://github.com/intarian/CDIEGO/blob/main/pca_data_functions.py).  

## Population Covariance Matrix Generation 
Details of generation of population covariance matrix $\mathbf{\Sigma}$ with custom eigengap $\Lambda$ are as follows:  




## List of Experiments
### 1. Performance Effect of different values of consensus rounds.
This experiment is performed by [1.effect_of_diff_Tc_synthetic_data_mp.py](1.effect_of_diff_Tc_synthetic_data_mp.py) by generating synthetic data 
for three different values of consensus rounds $T_c$.  
1. $T_c = T_{mix}$
2. $T_c = \log(Nt)$
3. $T_c = T_{mix} \frac{3}{2} \log(Nt)$. which we define to be optimal value of consensus rounds.  

**Expected Outcome**: The experiment show that for optimal value of $T_c$, there is no gap between the output by DIEGO (which assumes a fully connected network) and 
that of output by C-DIEGO.  
The same experiment is also performed using MNIST dataset is provided in [1a.effect_of_diff_Tc_mnist_data_mp.py](1a.effect_of_diff_Tc_mnist_data_mp.py)  
Both experiments using Synthetic and Real dataset are averaged over 50 monte-carlo trials running on multiple cores.  

### 2. Convergence rate effect on different values of $\Lambda$ (eigengap)
This experiment uses three different values of eigengap $\Lambda$ to see the effect on convergence.
We generate three covariance matrices with three different eigengaps and run the
experiment under FC graph and NFC graph. The topology of NFC graph remains fixed for all three cases of eigengap.  
**Expected Outcome**: The effect of optimal Tc remains independent of eigengap. The larger the eigengap the better convergence.  
The experiment is performed by [2.effect_of_diff_eiggap_synthetic_data_mp.py](2.effect_of_diff_eiggap_synthetic_data_mp.py) and uses synthetic data.

### 3. Convergence rate effect on different values of N (the number of nodes)
This experiment uses three different values of $N$ and see the effect on convergence rate under FC network.  
**Expected Outcome**: The higher the value of $N$, the better the convergence to corroborate theoretical results.  
The experiment is performed by [3.effect_of_diff_N_synthetic_data_mp.py](3.effect_of_diff_N_synthetic_data_mp.py) and uses synthetic data.

### 4. Convergence rate effect under different topologies
This experiment see the effect of C-DIEGO FC vs NFC in three different topologies:
1. Ring Graph
2. Erdos-Reyni (FC) with one dropped node
3. Line Graph  
We use (topologies_gen.py)[topologies_gen.py] file to generate Line and Ring graph and their weight matrices.  
   (Experiment in-progress)