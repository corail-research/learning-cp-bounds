import knapsack
import numpy as np
def load_instance(filename: str) -> knapsack.MultiKnapsack:
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_constraint, n_item = [int(x) for x in lines[0].split()] 
        profit = np.array([int(x) for x in lines[1].split()])
        budgets = np.array([int(x) for x in lines[2].split()])
        print([[len(line.split())] for line in lines[3:-1]])
        weights = np.array([[int(x) for x in line.split()] for line in lines[3:-1]])
        
        return knapsack.MultiKnapsack(n_item, profit, weights, budgets)