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

