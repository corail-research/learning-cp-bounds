import knapsack
import bounding
import utils
import numpy as np
import jax
import jax.numpy as jnp
import optax

# Data: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2.txt

if __name__ == '__main__':
    print('Hello World!')

    # Create a knapsack problem
    # Methode 1 : Create a knapsack problem from scratch
    #n_item = 3
    #profit = np.array([2, 3, 4])
    #weights = np.array([[12, 19, 30], [49, 40, 31]])
    #budgets = np.array([46, 76])
    # knapsack = knapsack.MultiKnapsack(n_item, profit, weights, budgets)

    # Methode 2 : Load a knapsack problem from a data file
    knapsack = utils.load_instance("data/weish10")
    print(knapsack)

    # Create a bounding object
    bounding = bounding.Bounding(knapsack)

    init_multiplier = 10.0
    #print(knapsack.solve())
    multipliers = np.ndarray(shape=(len(knapsack.budgets)-1, knapsack.n_item), dtype=float, buffer=np.ones((len(knapsack.budgets)-1, knapsack.n_item)) * init_multiplier)
    
    bound = bounding.subgradient_optimization(multipliers, 0.1, 5)
    print("bound subgradient_optimization : ", bound)
    bound = int(bounding.unsupervised_optimization(multipliers, 0.1, 5))
    print("bound unsupervised_optimization : ", bound)
   
    # print("solving without bound", knapsack.solve())
    print("solving with bound", knapsack.solve())

