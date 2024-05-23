import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as ss


def generate_knapsack(nb_items,nb_constraints, P, tightness):
    profits = ss.expon.rvs(*P, size=nb_items)
    for i in range(len(profits)):
        profits[i]=int(profits[i])
    profits = np.random.randint(low=0,high=1000,size=nb_items)
    profits = sorted(profits,reverse=True)
    W=[]
    capacities=[]
    for j in range (0,nb_constraints):
        weights= np.random.randint(low=0,high=1000,size=nb_items)
        weights=np.fmax(np.zeros(nb_items),weights)
        capacity=int(tightness[j]*np.sum(weights))
        W.append(weights)
        capacities.append(capacity)
    P=[]
    P=P+[nb_constraints,nb_items]+list(profits)+capacities
    for j in range (0,nb_constraints):
        P=P+list(W[j])
    return(P)

with open("train/pissinger/knapsacks-data-trainset.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 30
    for i in range (0, 30):
        tightness=np.array([0.3 + 0.03 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("test/pissinger/knapsacks-data-testset.txt",'w') as f:

    Ps = (5, 300)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 30
    for i in range (0, 3):
        tightness=np.array([0.3 + 0.03 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)
                