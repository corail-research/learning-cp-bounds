from knapsack import MultiKnapsack
import jax
import jax.numpy as jnp
import numpy as np
import optax

# class Bounding():

#     def __init__(self, instance: MultiKnapsack) -> None:
#         self.instance: MultiKnapsack = instance

#     def __str__(self) -> str:
#         return f"Bounding(knapsack={self.instance})"
    
#     def compute_bound(self, multipliers) -> dict:
#         # TODO : now only working on the root node, but need to be adapt to work on every node => take bound as an argument with default value = 0 for root node
#         """
#         Compute the bound of the problem with the given multipliers 
#         by solving the subproblems.
#         Return : subproblem_sol::np.ndarray
#                  bound::float
#         """
#         print("[START compute_bound]")
#         bound = 0.0
#         print("multipliers : ", multipliers, multipliers.shape)
#         cur_profit = np.zeros((self.instance.n_item))
#         print("cur_profit : ", cur_profit)

#         cur_profit = self.instance.profit + np.sum(multipliers, axis=0)
#         print("cur_profit : ", cur_profit)
#         # TODO : check code below
#         #cur_profit = cur_profit.astype(int)
#         #for j in range(len(self.instance.n_item)):
#         #    cur_profit += [self.instance.profit[i] + multipliers[j,i]  for i in  range(len(self.instance.budgets)-1)]

#         # create and solve the first subproblem
#         cur_constraint = [self.instance.weights[0]]
#         cur_budget = [self.instance.budgets[0]]
#         cur_subproblem = MultiKnapsack(n_item=self.instance.n_item, profit=cur_profit, weights=cur_constraint, budgets=cur_budget)
#         print("cur_constraint : ", cur_constraint)
#         print("cur_budget : ", cur_budget)
#         print("cur_subproblem : ", cur_subproblem)
#         subproblem_sol = np.ndarray(shape=(len(self.instance.budgets), self.instance.n_item))
#         sol, obj, _ = cur_subproblem.solve()
#         print("sol : ", sol)
#         print("obj : ", obj)
#         bound += obj
#         subproblem_sol[0,:] = sol
#         print("bound : ", bound)
#         print("subproblem_sol : ", subproblem_sol)

#         # create and solve the m-1 following subproblems
#         for i in range(1,len(self.instance.budgets)):
#             print("subrpoblem : ", i)
#             # TODO : check code below
#             #cur_profit = [- multipliers[j]  for j in range(self.instance.n_item)]
#             cur_profit = - multipliers[i-1,:]
#             print("cur_profit : ", cur_profit)
#             cur_constraint = [self.instance.weights[i]]
#             cur_budget = [self.instance.budgets[i]]
#             cur_subproblem = MultiKnapsack(n_item=self.instance.n_item, profit=cur_profit, weights=cur_constraint, budgets=cur_budget)
#             print("cur_constraint : ", cur_constraint)
#             print("cur_budget : ", cur_budget)
#             print("cur_subproblem : ", cur_subproblem)
#             sol, obj, _ = cur_subproblem.solve()
#             print("sol : ", sol)
#             print("obj : ", obj)
#             subproblem_sol[i,:] = sol
#             bound += obj
#             print("bound : ", bound)
#             print("subproblem_sol : ", subproblem_sol)
#         print("[END compute_bound]")
#         return subproblem_sol, bound
    
    
#     def subgradient_optimization(self, multipliers, lr: float, n_iter: int = 10) -> list[float]:
#         """
#         Compute the current solution and the current bound of the problem
#         and it updates the multipliers with the subgradient method.
#         """
#         print("[START subgradient_optimization]")
#         for i in range(n_iter):
#             print("ITERATION", i)
#             cur_sol, cur_bound = self.compute_bound(multipliers)
#             print("solution:", cur_sol)
#             print("bound:", cur_bound)
#             print("multipliers:", multipliers)
#             print("--------------------")
#             # update multipliers as in the article
#             for j in range(len(multipliers)):
#                 multipliers[j] = multipliers[j] - lr * (cur_sol[0][:] - cur_sol[j+1][:])
#         print("[END subgradient_optimization]")
#         return cur_bound


#     def ub_expr(multipliers, cur_sol, profits):
#         """
#         Explicit the upper bound expression to becomprehensive by jax.grad
#         """
#         print("[START ub_expr]")
#         ub = 0.0

#         profit_expr = jax.numpy.add(profits,  jnp.sum(-multipliers, axis = 0)) 
#         print("profit_expr : ", profit_expr)
#         ub = jnp.dot(profit_expr,  cur_sol[0,:])
#         print("ub 1 : ", ub)
#         # TODO : check code below
#         #for c in range(len(profits)):
#         #    ub += jnp.dot(profits[:] - jnp.sum(multipliers[c,:]), cur_sol[0,:]) 

#         for c in range(1,cur_sol.shape[0]):
#             ub += jnp.dot(multipliers[c-1,:], cur_sol[c,:])
#         print("ub 2 : ", ub)
#         # TODO : check code below
#         #ub = jnp.dot((profits[:] - multipliers[0,:]), cur_sol[0,:]) 
#         #jnp.dot((profits[:] - multipliers[0,:]), cur_sol[0,:]) + jnp.dot(multipliers[0,:], cur_sol[1,:])
#         print("[END ub_expr]")
#         return ub

#     def unsupervised_optimization(self, multipliers, lr: float, n_iter: int = 10) -> list[float]:
#         print("[START unsupervised_optimization]")
#         start_learning_rate = 5e-1
#         optimizer = optax.adabelief(-start_learning_rate)
#         opt_state = optimizer.init(jnp.array(multipliers))
#         print("optimizer : ", optimizer)
#         print("opt_state : ", opt_state)
#         for i in range(n_iter):
#             print("ITERATION", i)
#             cur_sol, cur_bound = self.compute_bound(multipliers)
#             print("cur_sol:", cur_sol)
#             print("bound:", cur_bound)
#             print("multipliers:", multipliers)
#             print("--------------------")

#             grads = jax.grad(Bounding.ub_expr)(multipliers, cur_sol, self.instance.profit)
#             updates, opt_state = optimizer.update(grads, opt_state)
#             multipliers = optax.apply_updates(multipliers, updates)
#             multipliers = np.array(multipliers)
#             print("grads:", grads)
#             print("updates:", updates)
#             print("opt_state:", opt_state)
#             print("multipliers:", multipliers)

#         print("[END unsupervised_optimization]")
#         return cur_bound



        

    


  

























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
