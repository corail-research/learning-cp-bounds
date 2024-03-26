# Learning bounds in Constraint Programming


### 1. Importing the repository

```shell
git clone https://github.com/corail-research/learning-bounds.git
```

### 2. Building Gecode

Please refer to the setup instructions available on the [official website](https://www.gecode.org/).

### 3. Compiling the solver

A makefile is available in the root repository. First, modify it by adding your python path. Then, you can compile the project as follows:

```shell
make mknapsack
```
It will create the executable ```solver_mknapsack```.

### 4. Test ypur installation

```shell
./solver_mknapsack
```

