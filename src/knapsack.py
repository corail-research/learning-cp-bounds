import cpmpy as cp
import pandas as pd
from cpmpy.solvers import CPM_minizinc

class MultiKnapsack():

    def __init__(self, n_item: int, profit: list[int], weights: list[list[int]], budgets: list[int]) -> None:
        self.n_item: int = n_item
        self.profit: list[int] = profit
        self.weights: list[list[int]] = weights
        self.budgets: list[int] = budgets

    def __str__(self) -> str:
        return f"MultiKnapsack(n_item={self.n_item}, profit={self.profit}, weights={self.weights}, budgets={self.budgets})"
        
    def solve(self, bound=-1):
        variables = cp.intvar(0,1, shape=self.n_item)
        
        model = cp.Model()

        for c in range(len(self.budgets)):
            model += cp.sum(self.weights[c][i] * variables[i] for i in range(self.n_item)) <= self.budgets[c]
        
        profit = cp.sum(self.profit[i] * variables[i] for i in range(self.n_item))
       
        if bound > 0: model += (profit <= bound)
        model.maximize(profit)
        #print(model)

        solver = cp.SolverLookup.get("ortools", model)
        #
        if solver.solve(log_search_progress = False, linearization_level=0,  cp_model_presolve=False):
            print("solution:", variables.value())
            print("profit:", model.objective_value())
            return variables.value(), model.objective_value(), model.status()
        else:
            raise Exception("No solution found")