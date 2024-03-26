from knapsack import MultiKnapsack
import jax
import jax.numpy as jnp
import numpy as np
import optax

class Bounding():

    def __init__(self, instance: MultiKnapsack) -> None:
        self.instance: MultiKnapsack = instance

    def __str__(self) -> str:
        return f"Bounding(knapsack={self.instance})"
    
    def compute_bound(self, multipliers) -> dict:
        # TODO : now only working on the root node, but need to be adapt to work on every node => take bound as an argument with default value = 0 for root node
        """
        Compute the bound of the problem with the given multipliers 
        by solving the subproblems.
        Return : subproblem_sol::np.ndarray
                 bound::float
        """
        bound = 0.0
        cur_profit = np.zeros((self.instance.n_item))

        cur_profit = self.instance.profit + np.sum(multipliers, axis=0)

        # create and solve the first subproblem
        cur_constraint = [self.instance.weights[0]]
        cur_budget = [self.instance.budgets[0]]
        cur_subproblem = MultiKnapsack(n_item=self.instance.n_item, profit=cur_profit, weights=cur_constraint, budgets=cur_budget)
        subproblem_sol = np.ndarray(shape=(len(self.instance.budgets), self.instance.n_item))
        sol, obj, _ = cur_subproblem.solve()
        bound += obj
        subproblem_sol[0,:] = sol

        # create and solve the m-1 following subproblems
        for i in range(1,len(self.instance.budgets)):
            cur_profit = - multipliers[i-1,:]
            cur_constraint = [self.instance.weights[i]]
            cur_budget = [self.instance.budgets[i]]
            cur_subproblem = MultiKnapsack(n_item=self.instance.n_item, profit=cur_profit, weights=cur_constraint, budgets=cur_budget)
            sol, obj, _ = cur_subproblem.solve()
            subproblem_sol[i,:] = sol
            bound += obj
        return subproblem_sol, bound
    
    
    def subgradient_optimization(self, multipliers, lr: float, n_iter: int = 10) -> list[float]:
        """
        Compute the current solution and the current bound of the problem
        and it updates the multipliers with the subgradient method.
        """
        for i in range(n_iter):
            cur_sol, cur_bound = self.compute_bound(multipliers)
            # update multipliers as in the article
            for j in range(len(multipliers)):
                multipliers[j] = multipliers[j] - lr * (cur_sol[0][:] - cur_sol[j+1][:])
        return cur_bound


    def ub_expr(multipliers, cur_sol, profits):
        """
        Explicit the upper bound expression to be comprehensive by jax.grad
        """
        ub = 0.0

        profit_expr = jax.numpy.add(profits,  jnp.sum(-multipliers, axis = 0)) 
        ub = jnp.dot(profit_expr,  cur_sol[0,:])

        for c in range(1,cur_sol.shape[0]):
            ub += jnp.dot(multipliers[c-1,:], cur_sol[c,:])
        return ub

    def unsupervised_optimization(self, multipliers, lr: float, n_iter: int = 10) -> list[float]:
        start_learning_rate = 5e-1
        optimizer = optax.adabelief(-start_learning_rate)
        opt_state = optimizer.init(jnp.array(multipliers))

        for i in range(n_iter):
            cur_sol, cur_bound = self.compute_bound(multipliers)

            grads = jax.grad(Bounding.ub_expr)(multipliers, cur_sol, self.instance.profit)
            updates, opt_state = optimizer.update(grads, opt_state)
            multipliers = optax.apply_updates(multipliers, updates)
            multipliers = np.array(multipliers)

        return cur_bound
