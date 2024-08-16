
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
ANONYMIZED
```


### 2  Building Gecode

Please refer to the setup instructions available on the [official website](https://www.gecode.org/).

### 3 Download Libtorch

Please download the libtorch version that suits your compilation environnment : (https://pytorch.org/get-started/locally/)

### 4 Compiling the solver

A makefile is available in the root repository. First, modify it by adding your python path and your libtrch path. You also need to modifiy the CMakeLists.txt to add your pybind11 path. Then, you can compile the project as follows:

```shell
make mknapsack
```
It will create the executable ```solver_mknapsack```.


## Basic use

### 1. Training a model
(Does not require Gecode)
In src/problem/mknapsack/training/
```shell
python3 lbounds --size_instances=100 #train a model for instances with up to 100 items
```
### 2. Solving the problem
(Require Gecode)
```shell
# For mknapsack
./solver_mknapsack --number_of_size=3 --number_of_models=1  --start_size=2 --start_models=0  --n_file=0 --write_samples=run --use_gpu=use_gpu # Solve with CP+SG for mk100 and first instance
./solver_mknapsack --number_of_size=3 --number_of_models=2  --start_size=2 --start_models=1  --n_file=0 --write_samples=run --use_gpu=use_gpu # Solve with CP+Learning(root) + SG for mk100 and first instance
./solver_mknapsack --number_of_size=3 --number_of_models=3  --start_size=2 --start_models=2  --n_file=0 --write_samples=run --use_gpu=use_gpu # Solve with CP+Learning(all) for mk100 and first instance
./solver_mknapsack --number_of_size=3 --number_of_models=4  --start_size=2 --start_models=3  --n_file=0 --write_samples=run --use_gpu=use_gpu # Solve with CP+Learning(all)+SG for mk100 and first instance

# For SSP
./solver_ssp --number_of_size=3 --number_of_models=4  --start_size=2 --start_models=3  --n_file=0 --write_samples=run --use_gpu=use_gpu # Solve with CP+SG for ssp10-80 and first instance



## Technologies and tools used

* The code, at the exception of the CP model, is implemented in Python 3.11.
* The CP model is implemented in C++ and is solved using [Gecode](https://www.gecode.org/). The reason of this design choice is that there is no CP solver in Python with the requirements we needed. 
* The graph neural network architecture has been implemented in torch_geometric. 
* The interface between the C++ and Python code is done with [Pybind11](https://github.com/pybind) and libtorch.

```

## Licence

This work is under MIT licence (https://choosealicense.com/licenses/mit/). It is a short and simple very permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code. 

