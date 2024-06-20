import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as ss


def generate_knapsack(nb_items,nb_constraints, P, tightness):
    profits = ss.expon.rvs(*P, size=nb_items)
    for i in range(len(profits)):
        profits[i]=int(profits[i])
    profits = np.random.randint(low=0,high=500,size=nb_items)
    profits = sorted(profits,reverse=True)
    W=[]
    capacities=[]
    for j in range (0,nb_constraints):
        weights= np.random.randint(low=50,high=100,size=nb_items)
        weights=np.fmax(np.zeros(nb_items),weights)
        capacity=int(tightness[j]*np.sum(weights))
        W.append(weights)
        capacities.append(capacity)
    P=[]
    P=P+[nb_constraints,nb_items]+list(profits)+capacities
    for j in range (0,nb_constraints):
        P=P+list(W[j])
    return(P)

with open("train/pissinger/knapsacks-data-trainset30.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 30
    for i in range (0, 500):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("train/pissinger/knapsacks-data-trainset50.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 50
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("train/pissinger/knapsacks-data-trainset100.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 100
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("train/pissinger/knapsacks-data-trainset200.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 200
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("test/pissinger/knapsacks-data-testset30.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 30
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("test/pissinger/knapsacks-data-testset50.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 50
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("test/pissinger/knapsacks-data-testset100.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 100
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)

with open("test/pissinger/knapsacks-data-testset200.txt",'w') as f:

    Ps = (10, 150)
    # plot the fitted distribution
    nb_constraints=5
    nb_items = 200
    for i in range (0, 50):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        knapsack=generate_knapsack(nb_items,5,Ps, tightness)
        line=""
        for i in range(len(knapsack)-1):
            line=line+str(int(knapsack[i]))+","
        line += str(int(knapsack[-1]))
        line += "\n"
        f.write(line)
