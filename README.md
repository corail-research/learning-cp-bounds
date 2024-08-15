# Learning bounds in Constraint Programming


### 1. Importing the repository

```shell
git clone https://github.com/corail-research/learning-bounds.git
```

### 2. Building Gecode

Please refer to the setup instructions available on the [official website](https://www.gecode.org/).

### 3. Download Libtorch

Please download the libtorch version that suits your compilation environnment : (https://pytorch.org/get-started/locally/)

### 4. Compiling the solver

A makefile is available in the root repository. First, modify it by adding your python path and your libtrch path. You also need to modifiy the CMakeLists.txt to add your pybind11 path. Then, you can compile the project as follows:

```shell
make mknapsack
```
It will create the executable ```solver_mknapsack```.

### 5. Test your installation

```shell
./solver_mknapsack 
```


# Learning valid bounds in Constraint Programming
Lagrangian decomposition (LD) is a relaxation method that provides a dual bound for constrained optimization problems by decomposing them into more manageable sub-problems. This bound can be used in branch-and-bound algorithms to prune the search space effectively. In brief, a vector of Lagrangian multipliers is associated with each sub-problem, and an iterative procedure (e.g., a sub-gradient optimization) adjusts these multipliers to find the tightest bound. 

Initially applied to integer programming, Lagrangian decomposition has succeeded in constraint programming due to its versatility and the fact that global constraints provide natural sub-problems. However, the non-linear and combinatorial nature of sub-problems in constraint programming makes it computationally intensive to optimize the Lagrangian multipliers with sub-gradient methods at each node of the tree search. 

This currently limits the practicality of LD as a general bounding mechanism for constraint programming. To address this challenge, we propose a self-supervised learning approach that leverages neural networks to generate multipliers directly, yielding tight bounds. This approach significantly reduces the number of sub-gradient optimization steps required, enhancing the pruning efficiency and reducing the execution time of constraint programming solvers.

This contribution is one of the few that leverage learning to enhance bonding mechanisms on the dual side, a critical element in the design of combinatorial solvers. To our knowledge, this work presents  the first generic method for learning valid dual bounds in constraint programming. 

We validate our approach on two challenging combinatorial problems:  The multi-dimensional knapsack problem and the shift scheduling problem. The  results show that our approach can solve more instances than the standard application of LD to constraint programming, reduce execution time by more than half, and has promising generalization ability through fine-tuning.

Please be aware that this project is still at research level.

## Content of the repository

For each problem that we have considered, you can find:

*  A GNN model serving as a predictor for lagrangian multipliers.
*  A DL training algorithm based.
*  The models, and the hyperparameters used, that we trained.
*  Three CP solving algorithms leveraging the learned models
*  A random instance generators for training the model and evaluating the solver.

```bash
.
├── run_training_x_y.sh  # script for running the training. It is where you have to enter the parameters 
├── trained_models/  # directory where the models that you train will be saved
└── src/ 
	├── problem/  # problems that we have implemented
		└── mnkapsack/
		      ├── training/
              ├── lbounds.py
              ├── bounds.py
              ├── lbounds-weish.py 
		      ├── solving/
              ├── solver.cpp 
		├── ssp/    
```
## Installation instructions

### 1. Importing the repository

```shell
git clone https://github.com/corail-research/learning-bounds.git
```
### 2. Setting up the conda virtual environment

```shell
conda env create -f conda_env.yml 
```
Note: install a [DGL version](https://www.dgl.ai/pages/start.html) compatible with your CUDA installation.
### 3. Building Gecode

Please refer to the setup instructions available on the [official website](https://www.gecode.org/).

### 4. Compiling the solver

A makefile is available in the root repository. First, modify it by adding your python path. Then, you can compile the project as follows:

```shell
make [problem] # e.g. make tsptw
```
It will create the executable ```solver_tsptw```.

## Basic use

### 1. Training a model
(Does not require Gecode)
```shell
./run_training_mknapsack.sh # for PPO
./run_training_ssp.sh # for DQN
```
### 2. Solving the problem
(Require Gecode)
```shell
# For mknapsack
./solver_mknapsack --model=rl-ilds-dqn --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --d_l=5000 --cache=1 --seed=1  # Solve with ILDS-DQN
./solver_mknapsack --model=rl-bab-dqn --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --cache=1 --seed=1 # Solve with BaB-DQN
./solver_mknapsack --model=rl-rbs-ppo --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --cache=1 --luby=1 --temperature=1 --seed=1 # Solve with RBS-PPO
./solver_mknapsack --model=nearest --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --d_l=5000 --seed=1 # Solve with a nearest neigbour heuristic (no learning)

# For SSP
./solver_ssp --model=rl-ilds-dqn --time=60000 --size=50 --capacity_ratio=0.5 --lambda_1=1 --lambda_2=5 --lambda_3=5 --lambda_4=5  --discrete_coeffs=0 --cache=1 --seed=1 

```
For learning based methods, the model selected by default is the one located in the corresponding ```selected_model/``` repository. For instance:

```shell
selected-models/ppo/tsptw/n-city-20/grid-100-tw-10-100/ 

```

## Example of results

The table recaps the solution obtained for an instance generated with a seed of 0, and a timeout of 60 seconds. 
Bold results indicate that the solver has been able to proof the optimality of the solution and a dash that no solution has been
found within the time limit.

### Tour cost for the TSPTW

| Model name  | 20 cities | 50 cities | 100 cities |
| ------------------ 	|---------------- 	| -------------- 	| --------------|
| DQN  			|    959        	|     -     		|      -       	| 
| PPO (beam-width=16)   |    959        	|     -    		|      -       	| 
| CP-nearest  		|    **959**        	|     -     		|      -       	| 
| BaB-DQN   		|     **959**       	|      **2432**        	|     4735     	| 
| ILDS-DQN   		|    **959**           	|      **2432**      	|     -      	| 
| RBS-PPO   		|    **959**          	|      **2432**     	|      4797     | 

```shell
./benchmarking/tsptw_bmk.sh 0 20 60000 # Arguments: [seed] [n_city] [timeout - ms]
./benchmarking/tsptw_bmk.sh 0 50 60000
./benchmarking/tsptw_bmk.sh 0 100 60000
```

### Profit for Portfolio Optimization

| Model name  		| 20 items 	    | 50 items       	| 100 items      |
| ------------------ 	|----------------   | -------------- 	| -------------- |
| DQN  	  		|     247.40        |      1176.94     |      2223.09      | 
| PPO (beam-width=16)  	|     264.49        |      1257.42      |      2242.67      | 
| BaB-DQN   		|     **273.04**    |      1228.03      |      2224.44      | 
| ILDS-DQN   		|     273.04        |      1201.53      |      2235.89       | 
| RBS-PPO   		|     267.05       |      1265.50      |      2258.65       | 

```shell
./benchmarking/portfolio_bmk.sh 0 20 60000 # Arguments: [seed] [n_item] [timeout - ms]
./benchmarking/portfolio_bmk.sh 0 50 60000
./benchmarking/portfolio_bmk.sh 0 100 60000
```

## Technologies and tools used

* The code, at the exception of the CP model, is implemented in Python 3.7.
* The CP model is implemented in C++ and is solved using [Gecode](https://www.gecode.org/). The reason of this design choice is that there is no CP solver in Python with the requirements we needed. 
* The graph neural network architecture has been implemented in Pytorch together with DGL. 
* The set embedding is based on [SetTransformer](https://github.com/juho-lee/set_transformer).
* The interface between the C++ and Python code is done with [Pybind11](https://github.com/pybind).

## Current implemented problems

At the moment, only the travelling salesman problem with time windows and the 4-moments portfolio optimization are present in this repository. However, we also have the TSP, and the 0-1 Knapsack problem available. If there is demand for these problems, I will add them in this repository. Feel free to open an issue for that or if you want to add another problem.

## Cite

Please use this reference:

```latex
@misc{dabert2024learningbounds,
    title={},
    author={},
    year={},
}
```

## Licence

This work is under MIT licence (https://choosealicense.com/licenses/mit/). It is a short and simple very permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code. 

