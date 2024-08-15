import argparse

import json
from pathlib import Path
import pandas as pd
import os
import sklearn as sk
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict

import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Dataset, download_url
import torch.optim as optim
import wandb

from numba import cuda

# Créer un analyseur d'arguments
parser = argparse.ArgumentParser(description="Get the size of the problem")

# Ajouter des arguments
parser.add_argument('size_instances', type=str, help='number of items of the instances')

# Analyser les arguments
args = parser.parse_args()

size_instances = args.size_instances

class Node:
    def __init__(self, index, state):
        self.index = index
        self.state = state

    def __hash__(self):
            return hash((self.index, self.state))

    def __eq__(self, other):
        return self.index == other.index and self.state == other.state

class Edge:
    def __init__(self, prevNode, newNode, label, cost):
        self.prevNode = prevNode
        self.newNode = newNode
        self.label = label
        self.cost = cost

def intersect_paths(solutions, n, v):
    # Initialize common edges with the edges of the first sub-graph
    common_edges = [ [0 for i in range(v * v)] for j in range(n)]
    for i in range(n):
        for j in range(v * v):
            common_edges[i][j] = solutions[0][i][j]
    
    # Intersect the edges of the first sub-graph with the other sub-graphs
    for i in range(1, len(solutions)):
        for j in range(n):
            for k in range(v * v):
                common_edges[j][k] = common_edges[j][k] * solutions[i][j][k]

    return common_edges

def find_path(edges, n, v):
    # edges is a list of n lists of v * v elements :
    # edges[i][j *v + k] = 1 if there is an edge between the j-th and (j+1)-th variables with values j and k
    # edges[i][j *v + k] = 0 otherwise

    # the function return true if there is a path from the first variable to the last variable
    # that respects the edges constraints

    visited = [ [False for i in range(v)] for j in range(n)]

    def dfs(i, j):
        if i == n - 1:
            return True
        visited[i][j] = True
        for k in range(v):
            if edges[i][j * v + k] == 1 and not visited[i + 1][k]:
                if dfs(i + 1, k):
                    return True
        return False
    
    for i in range(v):
        if dfs(0, i):
            return True

    return False

def load_dataset(file_path):
    """
    Function to load the dataset from the text files
    :param data_split: Path to the folder containing the text
    :return: List of graphs
    """

    graphs = []


    with open(file_path, 'r') as file:
        index = 0
        for line in file.readlines():
            index += 1
            print(index)
            if index > 0:
                break
            try:
                graph = torch_geometric.data.Data()
                probleme = line.split(sep=',')
                problem = [int(probleme[i]) for i in range(len(probleme) - 1)]
                problem.append(int(float(probleme[len(probleme) - 1].strip('\n'))))
                
                n = int(problem[1])  # Number of variables
                m = int(problem[0])  # Number of constraints
                v = int(problem[2])  # Number of values for each variable
                s = int(problem[3])  # Number of states
                F = int(problem[4])  # Number of final states
                
                X = []  # Nodes of the problem graph
                edge_index = []  # Edges of the problem graph
                edge_weights = []  # Weights of the edges of the problem graph
                edge_attributes = []

                one_hot_encoding = torch.nn.functional.one_hot(torch.tensor([i for i in range(s + 1)]), num_classes=s + 1)

                for j in range(m):
                    for i in range(n):
                        for k in range(v):
                            X.append([problem[6 + F + i * v + k], j, i, k])

                all_adjLists = []
                solutions = []

                for l in range(m):
                    global_edges = defaultdict(list)
                    layers = [{} for _ in range(n + 1)]
                    adjList = {}

                    layers[0][0] = Node(0, 0)

                    for j in range(1, n):
                        for key, prevNode in layers[j - 1].items():
                            for a in range(v):
                                if problem[6 + F + n * v + (j - 1) * v + a] != 0:
                                    if problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a] != -1:
                                        newNode = Node(j, problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a])
                                        layers[j][newNode.state] = newNode
                                        cost = problem[6 + F + (j - 1) * v + a]
                                        if j - 1 not in adjList:
                                            adjList[j - 1] = []
                                        adjList[j-1].append(Edge(prevNode, newNode, a, cost))
                                        global_edges[prevNode].append(newNode)

                    for key, prevNode in layers[n - 1].items():
                        for a in range(v):
                            if problem[6 + F + n * v + (n - 1) * v + a] != 0:
                                if (problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a] != -1 and
                                        problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a] in problem[6: 6 + F]):
                                    newNode = Node(n, problem[6 + F + 2 * n * v + l * v * s + prevNode.state * v + a])
                                    layers[n][newNode.state] = newNode
                                    cost = problem[6 + F + (n - 1) * v + a]
                                    if n - 1 not in adjList:
                                        adjList[n - 1] = []
                                    adjList[n-1].append(Edge(prevNode, newNode, a, cost))
                                    global_edges[prevNode].append(newNode)

                    all_adjLists.append(adjList)

                    filtered_adjList = {}

                    current_layer = layers[n]

                    for j in range(n-1, -1, -1):
                        temp = {}
                        for key, node in current_layer.items():
                            for edge in adjList[j]:
                                if node.state == edge.newNode.state:
                                    if j not in filtered_adjList:
                                        filtered_adjList[j] = []
                                    filtered_adjList[j].append(edge)
                                    temp[edge.prevNode.state] = edge.prevNode
                        current_layer = temp
                    
                    all_adjLists[l] = filtered_adjList

                    solution = [ [0 for i in range(v * v)] for j in range(n)]

                    for j in range(n - 1):
                        for edge in filtered_adjList[j]:
                            for next_edge in filtered_adjList[j + 1]:
                                if edge.newNode.state == next_edge.prevNode.state:
                                    edge_index.append([l * v * n + j * v + edge.label,l * v * n + (j + 1) * v + next_edge.label])
                                    edge_weights.append(1)
                                    edge_attributes.append(one_hot_encoding[edge.newNode.state])
                                    solution[j][edge.label * v + next_edge.label] = 1
                    solutions.append(solution)

                common_edges = intersect_paths(solutions, n, v)

                if not find_path(common_edges, n, v):
                    raise Exception("No feasible solution found")
                
                for i in range(n):
                    for k in range(v):
                        for j in range(m - 1):
                            for l in range(j + 1, m):
                                edge_index.append([j * n * v + i * v + k, l * n * v + i * v + k])
                                edge_weights.append(1)
                                edge_attributes.append(one_hot_encoding[s])

                graph_id = problem[0] * problem[1]

                graph = torch_geometric.data.Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(edge_index).T,
                         edge_weight=torch.FloatTensor(edge_weights), edge_attr=torch.stack(edge_attributes), opt = problem[-1],  fix_bound = problem[-2],  graph_id=graph_id, graph_problem=problem)
                
                graphs.append(graph)
            except Exception as e:
                print(f'error: {e}')
                pass

    return graphs


def dp_ssp(profits, domain_values, transitions, states, final_states, nb_states, nb_values, nb_items, initialState, verbose=False):
    feasible = False

    # Build a graph
    layers = [defaultdict(Node) for _ in range(nb_items + 1)]
    adjList = defaultdict(list)

    layers[0][0] = Node(0, initialState)

    for j in range(1, nb_items):
        for key, prevNode in layers[j - 1].items():
            for a in range(nb_values):
                if domain_values[(j-1)*nb_values + a] != 0:
                    if transitions[prevNode.state * nb_values + a] != -1:
                        newNode = Node(j, transitions[prevNode.state * nb_values + a])
                        layers[j][newNode.state] = newNode
                        cost = profits[(j - 1) * nb_values + a]
                        if j - 1 not in adjList:
                            adjList[j - 1] = []
                        adjList[j - 1].append(Edge(prevNode, newNode, a, cost))

    for key, prevNode in layers[nb_items - 1].items():
        for a in range(nb_values):
            if domain_values[(nb_items-1)*nb_values + a] != 0:
                if transitions[prevNode.state * nb_values + a] != -1 and transitions[prevNode.state * nb_values + a] in final_states:
                    newNode = Node(nb_items, transitions[prevNode.state * nb_values + a])
                    layers[nb_items][newNode.state] = newNode
                    cost = profits[(nb_items - 1) * nb_values + a]
                    adjList[nb_items - 1].append(Edge(prevNode, newNode, a, cost))
                    if nb_items- 1 not in adjList:
                        adjList[nb_items - 1] = []
                    feasible = True

    R = defaultdict(lambda: -np.inf)
    predecessor = {}

    R[Node(0, initialState)] = 0

    for j in range(1, nb_items + 1):
        for key, node in layers[j].items():
            maxCost = -np.inf
            bestPredecessor = None
            bestLabel = -1
            for edge in adjList[j - 1]:
                if edge.newNode.index == j and edge.newNode.state == node.state:
                    newCost = R[Node(j - 1, edge.prevNode.state)] + edge.cost
                    if newCost > maxCost:
                        maxCost = newCost
                        bestPredecessor = edge.prevNode
                        bestLabel = edge.label
            R[node] = maxCost
            if maxCost != -np.inf:
                predecessor[node] = (bestPredecessor, bestLabel)

    # Find the optimal bound for the constraint
    optimalBound = -np.inf
    endNode = None
    for key, finalNode in layers[nb_items].items():
        if R[finalNode] > optimalBound:
            optimalBound = R[finalNode]
            endNode = finalNode

    if not feasible:
        print("No feasible solution found")
        return []

    if optimalBound == -np.inf:
        print("No valid path found to final states")

    # Backtrack to find the solution path
    solutionPath = []
    if optimalBound != -np.inf:
        currentNode = endNode
        while currentNode.index > 0:
            solutionPath.append(predecessor[currentNode][1])
            currentNode = predecessor[currentNode][0]
        solutionPath.reverse()

    return solutionPath


class GNN(torch.nn.Module):
    def __init__(self, n_features_nodes, n_classes, n_hidden, dropout, device):
        super(GNN, self).__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)

        # Define the linear layers for the graph embedding
        self.linear_embedding = nn.Linear(n_features_nodes, n_hidden[0])
        self.linear_embedding2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.linear_embedding3 = nn.Linear(n_hidden[1], n_hidden[2])

        # Define the graph convolutional layers
        self.conv1 = gnn.ResGatedGraphConv( n_hidden[2], n_hidden[3], edge_dim = 81)
        self.conv2 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 81)
        self.conv3 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 81)

        # Define the linear layers for the graph embedding
        self.linear_graph1 = nn.Linear(n_hidden[3], n_hidden[4])
        self.linear_graph2 = nn.Linear(n_hidden[4], n_hidden[5])

        # Define the linear layers for the final prediction
        self.linear = nn.Linear( 2 * n_hidden[5], n_hidden[6])
        self.linear2 = nn.Linear(n_hidden[6], n_hidden[7])
        self.linear3 = nn.Linear(n_hidden[7], n_classes)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, edge_index, edge_weight, edge_attribute, problem):
        # Perform graph convolution and activation function
        G = self.linear_embedding(G)
        G = F.relu(G)
        G = self.linear_embedding2(G)
        G = F.relu(G)
        G = self.linear_embedding3(G)
        G = F.relu(G)

        G = self.conv1(G, edge_index, edge_attribute)
        G = F.relu(G)
        G = self.conv2(G, edge_index, edge_attribute)
        G = F.relu(G)

        # Perform linear transformation and activation function
        G = self.linear_graph1(G)
        G = F.relu(G)
        G = self.linear_graph2(G)

        graph_embedding = torch.mean(G, dim = 0)
        u = []
        # Concatenate the graph embeddings with the nodes of the problem
        for j in range(problem[0] * problem[1] * problem[2]):
            u.append(torch.cat((G[j], graph_embedding)))
        u = torch.stack(u)

        # Perform linear transformation for final prediction
        u = u.squeeze(-1)
        #u = self.dropout(u)
        u = self.linear(u)
        u = F.relu(u)
        u = self.linear2(u)
        u = F.relu(u)
        u = self.linear3(u)

        u = u.to(self.device)
        u = u.squeeze(-1)
        # Solve the knapsack problem for each graph of the batch
        value_var_solutions = []
        bound = torch.zeros(1).to(self.device)

        for idx_constraint in range(problem[0]):
            profits = [0 for i in range(problem[1] * problem[2])]
            for idx in range(0, problem[1] * problem[2]):
              if idx_constraint == 0:
                    profits[idx] = problem[6 + problem[4] + idx] 
                    for j in range(1, problem[0]):
                        profits[idx] += u[j * problem[1] * problem[2] + idx]
              else:
                profits[idx] =  - u[idx_constraint * problem[1] * problem[2] + idx]
            values = problem[6 + problem[4] + problem[1] * problem[2] : 6 + problem[4] + 2 * problem[1] * problem[2]]
            transitions = problem[6 + problem[4] + 2 * problem[1] * problem[2] + idx_constraint * problem[2] * problem[3]: 6 + problem[4] + 2 * problem[1] * problem[2] + (idx_constraint + 1) * problem[2] * problem[3]]
            states = [i for i in range(problem[3])]
            final_states = problem[6: 6 + problem[4]]
            nb_states = problem[3]
            nb_values = problem[2]
            nb_items = problem[1]
            initialState = problem[5]
            value_var_solutions = dp_ssp(profits, values, transitions, states, final_states, nb_states, nb_values, nb_items, initialState, verbose=False)

            for idx_var in range(problem[1]):
              if idx_constraint == 0:
                    bound +=  problem[6 + problem[4] + idx_var * problem[2] + value_var_solutions[idx_var]]
                    for j in range(1, problem[0]):
                        bound += u[j * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]
              else:
                    bound -= u[idx_constraint * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]

        return bound, u


print('start')
## Train the models
graphs_training = load_dataset('../../../../data/ssp/train/ssp-data-testset'+ size_instances + '.txt')

print("len(graphs_training): ", len(graphs_training))
train_loader = torch_geometric.loader.DataLoader(graphs_training, batch_size=1, shuffle=False)

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GNN(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)
model.load_state_dict(torch.load('../../../../trained_models/SSP/GNN-SSP-'+ size_instances + '.pt'))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bounds_learning = []

for data in train_loader:
    model.train()
    optimizer.zero_grad()
    bound = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem[0])[0]
    y = data.opt.to(device)
    gobal_bound = bound.to(device)
    bounds_learning.append(torch.mean(gobal_bound).detach().item())

bounds_learning_grad = []

for data in train_loader:
    problems = data.graph_problem
    bounds_learning_grad_sublist = []
    model.train()
    optimizer.zero_grad()
    bound, u = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem[0])
    y = data.opt.to(device)
    for k in range(1):
        compteur = 0
        for idx_batch, problem in enumerate(data.graph_problem):
            values_var_solutions = []
            bound = 0
            for idx_constraint in range(problem[0]):
                profits = [0 for i in range(problem[1] * problem[2])]
                for idx in range(0, problem[1] * problem[2]):
                    if idx_constraint == 0:
                        profits[idx] = problem[6 + problem[4] + idx] 
                        for j in range(1, problem[0]):
                            profits[idx] += u[j * problem[1] * problem[2] + idx]
                    else:
                        profits[idx] =  - u[idx_constraint * problem[1] * problem[2] + idx]
                values = problem[6 + problem[4] + problem[1] * problem[2] : 6 + problem[4] + 2 * problem[1] * problem[2]]
                transitions = problem[6 + problem[4] + 2 * problem[1] * problem[2] + idx_constraint * problem[2] * problem[3]: 6 + problem[4] + 2 * problem[1] * problem[2] + (idx_constraint + 1) * problem[2] * problem[3]]
                states = [i for i in range(problem[3])]
                final_states = problem[6: 6 + problem[4]]
                nb_states = problem[3]
                nb_values = problem[2]
                nb_items = problem[1]
                initialState = problem[5]
                value_var_solutions = dp_ssp(profits, values, transitions, states, final_states, nb_states, nb_values, nb_items, initialState, verbose=False)
                values_var_solutions.append(value_var_solutions)

                for idx_var in range(problem[1]):
                    if idx_constraint == 0:
                        bound +=  problem[6 + problem[4] + idx_var * problem[2] + value_var_solutions[idx_var]]
                        for j in range(1, problem[0]):
                            bound += u[j * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]
                    else:
                        bound -= u[idx_constraint * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]
        
            bounds_learning_grad_sublist.append(bound.to('cpu').detach().item())

            for i in range(1, problem[0]):
                for j in range(problem[1]):
                    u[i * problem[1] * problem[2] + j * problem[2] + values_var_solutions[i][j]] += 0.1
                    u[i * problem[1] * problem[2] + j * problem[2] + values_var_solutions[0][j]] -= 0.1


    
    bounds_learning_grad.append(bounds_learning_grad_sublist)
        

bounds_grad = []

for data in train_loader:
    problems = data.graph_problem
    u = torch.rand(problems[0][0] * problems[0][1] * problems[0][1]).to(device)
    bounds_grad_sublist = []
    for k in range(1):
        compteur = 0
        for idx_batch, problem in enumerate(data.graph_problem):
            values_var_solutions = []
            bound = 0
            for idx_constraint in range(problem[0]):
                profits = [0 for i in range(problem[1] * problem[2])]
                for idx in range(0, problem[1] * problem[2]):
                    if idx_constraint == 0:
                        profits[idx] = problem[6 + problem[4] + idx]
                        for j in range(1, problem[0]):
                            profits[idx] += u[j * problem[1] * problem[2] + idx]
                    else:
                        profits[idx] =  - u[idx_constraint * problem[1] * problem[2] + idx]
                values = problem[6 + problem[4] + problem[1] * problem[2] : 6 + problem[4] + 2 * problem[1] * problem[2]]
                transitions = problem[6 + problem[4] + 2 * problem[1] * problem[2] + idx_constraint * problem[2] * problem[3]: 6 + problem[4] + 2 * problem[1] * problem[2] + (idx_constraint + 1) * problem[2] * problem[3]]
                states = [i for i in range(problem[3])]
                final_states = problem[6: 6 + problem[4]]
                nb_states = problem[3]
                nb_values = problem[2]
                nb_items = problem[1]
                initialState = problem[5]
                value_var_solutions = dp_ssp(profits, values, transitions, states, final_states, nb_states, nb_values, nb_items, initialState, verbose=False)
                values_var_solutions.append(value_var_solutions)

                for idx_var in range(problem[1]):
                    if idx_constraint == 0:
                        bound +=  problem[6 + problem[4] + idx_var * problem[2] + value_var_solutions[idx_var]]
                        for j in range(1, problem[0]):
                            bound += u[j * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]
                    else:
                        bound -= u[idx_constraint * problem[1] * problem[2] + value_var_solutions[idx_var] + idx_var * problem[2]]
            print(bound)
            print(u)
        
            bounds_grad_sublist.append(bound.to('cpu').detach().item())

            for i in range(1, problem[0]):
                for j in range(problem[1]):
                    u[i * problem[1] * problem[2] + j * problem[2] + values_var_solutions[i][j]] += 1
                    u[i * problem[1] * problem[2] + j * problem[2] + values_var_solutions[0][j]] -= 1

    
    bounds_grad.append(bounds_grad_sublist)

bound_learning = 0
for bound in bounds_learning:
    bound_learning += bound
bound_learning /= len(bounds_learning)

bound_learning_grad = []
for bound in bounds_learning_grad:
    bound_learning_grad.append(np.mean(bound))

bound_grad = []
for bound in bounds_grad:
    bound_grad.append(np.mean(bound))

# save these data to a file
with open("bounds_learning-ssp-'+ size_instances + '.txt", "w") as file:
    file.write(json.dumps(bounds_learning))

with open("bounds_learning_grad-ssp-'+ size_instances + '.txt", "w") as file:
    file.write(json.dumps(bounds_learning_grad))

with open("bounds_grad-ssp-'+ size_instances + '.txt", "w") as file:
    file.write(json.dumps(bounds_grad))
