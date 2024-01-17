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
    #n_item = 3
    #profit = np.array([2, 3, 4])
    #weights = np.array([[12, 19, 30], [49, 40, 31]])
    #budgets = np.array([46, 76])
    # knapsack = knapsack.MultiKnapsack(n_item, profit, weights, budgets)
    knapsack = utils.load_instance("data/weish10")
    print(knapsack)

    

    bounding = bounding.Bounding(knapsack)

    init_multiplier = 10.0
    #print(knapsack.solve())
    multipliers = np.ndarray(shape=(len(knapsack.budgets)-1, knapsack.n_item), dtype=float, buffer=np.ones((len(knapsack.budgets)-1, knapsack.n_item)) * init_multiplier)
    #print(bounding.subgradient_optimization(multipliers, 20, 5000))
    bound = int(bounding.unsupervised_optimization(multipliers, 0.1, 1000))
    
   
    print("solving without bound", knapsack.solve())
    print("solving with bound", knapsack.solve())

