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

import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Dataset, download_url
import torch.optim as optim
import wandb

from numba import cuda


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


def load_dataset(file_path):
    """
    Function to load the dataset from the text files
    :param data_split: Path to the folder containing the text
    :return: List of graphs
    """
    graphs = []
    graph_id = 0

    with open(file_path, 'r') as file:
            # A line is a node of the exploration tree
                for line in file.readlines():
                    try:
                    # Create a new graph object
                        graph = torch_geometric.data.Data()
                    # Converte the problem to a list of integers
                        probleme = line.split(sep = ',')
                        problem = []
                        for i in range(len(probleme)-1):
                            problem.append(int(probleme[i]))
                        problem.append(int(float(probleme[len(probleme)-1].strip('\n'))))
                        n = int(problem[1]) # Number of variable
                        m = int(problem[0]) # Number of constraints
                        v = int(problem[2]) # Number of values for each variable
                        s = int(problem[3]) # Number of states
                        F = int(problem[4]) # Number of final states
                        X = [] # Nodes of the problem graph
                        edge_index = [] # Edges of the problem graph
                        edge_weights = [] # Weights of the edges of the problem graph
                        edge_attributes = []

                        one_hot_encoding = torch.nn.functional.one_hot(torch.tensor([i for i in range (s+1)]), num_classes=s+1)

                    # Create the nodes of the graph (one node per triplet variable, constraint, value)
                        for j in range(m):
                            for i in range(n):
                                for k in range(v):
                                # The node contains the following features:
                                # - The profit of the variable
                                    X.append([problem[6 + F + i * v + k], j, i, k])

                        for l in range(m):

                            layers = [{} for _ in range(n + 1)]
                            adjList = {}

                            layers[0][0] = Node(0, 0)

                            for j in range(1, n):
                                for key, prevNode in layers[j - 1].items():
                                    for a in range((6 + F + n * v + (j - 1) * v), 6 + F + n * v + j * v):
                                        if problem[a] != 0:
                                            if problem[n * v  - (j-1) * v + l * s * v + prevNode.state * v + a] != -1:
                                                newNode = Node(j, problem[n * v  - (j-1) * v + l * s * v + prevNode.state * v + a])
                                                layers[j][newNode.state] = newNode
                                                cost = problem[ v + a - n * v]
                                                if j - 1 not in adjList:
                                                    adjList[j - 1] = []
                                                adjList[j - 1].append(Edge(prevNode, newNode, a - (6 + F + n * v + (j - 1) * v) , cost))

                            for key, prevNode in layers[n - 1].items():
                                for a in range(6 + F + n * v + (n - 1) * v, 6 + F + n * v + n * v):
                                    if problem[a] != 0:
                                        if (problem[n * v  - (n-1) * v + l * s * v + prevNode.state * v + a] != -1 and
                                                problem[n * v  - (n-1) * v + l * s * v + prevNode.state * v + a] in problem[6: 6 + F]):
                                            newNode = Node(n, problem[n * v  - (n-1) * v + l * s * v + prevNode.state * v + a])
                                            layers[n][newNode.state] = newNode
                                            cost = problem[ v + a - n * v]
                                            if n - 1 not in adjList:
                                                adjList[n - 1] = []
                                            adjList[n - 1].append(Edge(prevNode, newNode, a - (6 + F + n * v + (n - 1) * v) , cost))

                            for j in range(n - 1):
                                for edge in adjList[j]:
                                    for next_edge in adjList[j + 1]:
                                        if edge.newNode.state == next_edge.prevNode.state:
                                            edge_index.append([l * v * n + j * v + edge.label,l * v * n + (j + 1) * v + next_edge.label])
                                            edge_weights.append(1)
                                            edge_attributes.append(one_hot_encoding[edge.newNode.state])


                    # Create the edges of the graph between nodes that share the same variable
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
                    except:
                        pass
    return graphs


def dp_ssp(profits, domain_values, transitions, states, final_states, nb_states, nb_values, nb_items, initialState, verbose=False):

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
                    adjList[j - 1].append(Edge(prevNode, newNode, a, cost))

    for key, prevNode in layers[nb_items - 1].items():
        for a in range(nb_values):
          if domain_values[(nb_items-1)*nb_values + a] != 0:
            if transitions[prevNode.state * nb_values + a] != -1 and transitions[prevNode.state * nb_values + a] in final_states:
                newNode = Node(nb_items, transitions[prevNode.state * nb_values + a])
                layers[nb_items][newNode.state] = newNode
                cost = profits[(nb_items - 1) * nb_values + a]
                adjList[nb_items - 1].append(Edge(prevNode, newNode, a, cost))

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
    feasible = False
    endNode = None
    for key, finalNode in layers[nb_items].items():
        if R[finalNode] > optimalBound:
            optimalBound = R[finalNode]
            endNode = finalNode
            feasible = True

    if not feasible:
        print("No feasible solution found")
        return []
    
    # Backtrack to find the solution path
    solutionPath = []
    currentNode = endNode
    while currentNode.index > 0:
        solutionPath.append(predecessor[currentNode][1])
        currentNode = predecessor[currentNode][0]

    solutionPath.reverse()
    value_var_solution = solutionPath

    return value_var_solution


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
        self.conv1 = gnn.ResGatedGraphConv( n_hidden[2], n_hidden[3], edge_dim = 21)
        self.conv2 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 21)
        self.conv3 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 21)

        # Define the linear layers for the graph embedding
        self.linear_graph1 = nn.Linear(n_hidden[3], n_hidden[4])
        self.linear_graph2 = nn.Linear(n_hidden[4], n_hidden[5])

        # Define the linear layers for the final prediction
        self.linear = nn.Linear( 2 * n_hidden[5], n_hidden[6])
        self.linear2 = nn.Linear(n_hidden[6], n_hidden[7])
        self.linear3 = nn.Linear(n_hidden[7], n_classes)

        # Define the dropout laye£r
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
        # G = self.conv3(G, edge_index, edge_attribute)
        # G = F.relu(G)


        # Perform linear transformation and activation function
        G = self.linear_graph1(G)
        G = F.relu(G)
        G = self.linear_graph2(G)

        graph_embeddings = []
        compteur = 0

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


       # u = u.to(torch.device('cpu'))
#        u2=list(torch.clone(u).squeeze(-1).detach().numpy())
        u = u.to(self.device)

        # Solve the knapsack problem for each graph of the batch
        sum_fixed = 0
        value_var_solutions = []
        bound = 0

        for idx_constraint in range(problem[0]):
            profits = [0 for i in range(problem[1] * problem[2])]
            for idx in range(0, problem[1] * problem[2]):
              if idx_constraint == 0:
                profits[idx] = problem[6 + problem[4] + idx] + u[1 * problem[1] * problem[2] + idx]
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

            for i in range(problem[1]):
              if idx_constraint == 0:
                    bound += u[1 * problem[1] * problem[2] + value_var_solutions[i] + i * problem[2] ] + problem[6 + problem[4] + i * problem[2] + value_var_solutions[i]]
              else:
                    bound -= u[idx_constraint * problem[1] * problem[2] + value_var_solutions[i] + i * problem[2]]

        return bound
    

def train(model, optimizer, criterion, scheduler, train_loader, val_loader, n_epochs, device="cpu"):
    """
  Function to train the model.

  Args:
  - model: The GNN model to be trained
  - optimizer: The optimizer used for training
  - criterion: The loss function
  - scheduler: The learning rate scheduler
  - train_loader: The data loader for the training set
  - val_loader: The data loader for the validation set
  - n_epochs: The number of epochs to train for
  - device: The device to run the training on (default: "cpu")

  Returns:
  - train_loss: List of training losses for each epoch
  - val_loss: List of validation losses for each epoch
  - train_diff_ecart_opt: List of training relative differences between bound and optimal solutions for each epoch
  - val_diff_ecart_opt: List of validation relative differences between bound and optimal solutions for each epoch
    """

  # Metrics for each epoch
    train_loss = []
    val_loss = []
    train_diff_ecart_opt = []
    val_diff_ecart_opt = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} : ")
        train_loss_sublist = []
        train_diff_ecart_opt_sublist = []
        sum = 0
        for data in train_loader:
            model.train()
            optimizer.zero_grad()
            #bound = model(input_parameter, data.graph_problem[0])
            bound = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem[0])
            #print(bound)
            sum += bound
            loss = criterion(bound)
            loss.backward()
            optimizer.step()
            y = data.opt.to(device)
            gobal_bound = data.fix_bound.to(device) + bound.to(device)
            train_loss_sublist.append(torch.mean(gobal_bound).detach().item())
            train_diff_ecart_opt_sublist.append((torch.mean(torch.div(gobal_bound - y, y))).detach().item())

        train_loss.append(np.mean(train_loss_sublist))
        train_diff_ecart_opt.append(np.mean(train_diff_ecart_opt_sublist))

        val_loss_sublist = []
        val_diff_ecart_opt_sublist = []
        sum = 0
        for data in val_loader:
            model.eval()
            with torch.no_grad():
                bound = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem[0])
                loss = criterion(bound)
                sum += bound
                y = data.opt.to(device)
                gobal_bound = data.fix_bound.to(device) + bound.to(device)
                val_loss_sublist.append(torch.mean(gobal_bound).detach().item())
                val_diff_ecart_opt_sublist.append((torch.mean(torch.div(gobal_bound - y, y))).detach().item())


        val_loss.append(np.mean(val_loss_sublist))
        val_diff_ecart_opt.append(np.mean(val_diff_ecart_opt_sublist))

    #check if scheduler is none
        if scheduler is not None:
            scheduler.step(val_loss[-1])

    #, val loss {val_loss[-1]}, , val diff {val_diff[-1]}
        if epoch%1 ==0:
            print(f"Epoch {epoch} : "
      f"train loss {train_loss[-1]},  val loss {val_loss[-1]},\n "
      f"train diff ecart opt {train_diff_ecart_opt[-1] * 100}%, "
      f"val diff ecart opt {val_diff_ecart_opt[-1] * 100}%,\n"

                 )
        wandb.log({"train_loss": train_loss[-1], "val_loss": val_loss[-1], "train_diff_ecart_opt": train_diff_ecart_opt[-1], "val_diff_ecart_opt": val_diff_ecart_opt[-1]})




    return train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="learning_bounds_ssp",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "GNN-GatedGraphConv",
    "layers" : " 32, 16, 32, 64, 128, 64, 32, 16, 16",
    "dataset": "ssp-20-20",
    "epochs": 10
    }
)

## Train the models
graphs_training = load_dataset('/home/darius/scratch/learning-bounds/data/ssp/train/ssp-data-trainset20-20-subnodes.txt')

train_loader = torch_geometric.loader.DataLoader(graphs_training[:-150], batch_size=1, shuffle=False)
val_loader = torch_geometric.loader.DataLoader(graphs_training[-100:], batch_size=1, shuffle=False)

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def criterion(bounds):
    return torch.mean(bounds)

model = GNN(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt = train(model, optimizer, criterion, scheduler, train_loader, val_loader, 10, device)



class GNNsup1(torch.nn.Module):
    def __init__(self, n_features_nodes, n_classes, n_hidden, dropout, device):
        super(GNNsup1, self).__init__()

        self.device = device

       # Define the linear layers for the graph embedding
        self.linear_embedding = nn.Linear(n_features_nodes, n_hidden[0])
        self.linear_embedding2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.linear_embedding3 = nn.Linear(n_hidden[1], n_hidden[2])

        # Define the graph convolutional layers
        self.conv1 = gnn.ResGatedGraphConv( n_hidden[2], n_hidden[3], edge_dim = 21)
        self.conv2 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 21)
        self.conv3 = gnn.ResGatedGraphConv( n_hidden[3], n_hidden[3], edge_dim = 21)

        # Define the linear layers for the graph embedding
        self.linear_graph1 = nn.Linear(n_hidden[3], n_hidden[4])
        self.linear_graph2 = nn.Linear(n_hidden[4], n_hidden[5])

        # Define the dropout laye£r
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, edge_index, edge_attribute):
        # Perform graph convolution and activation function
        #print(G)
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

        G = self.linear_graph1(G)
        G = F.relu(G)
        G = self.linear_graph2(G)

        return (G)

class GNNsup2(torch.nn.Module):
    def __init__(self, n_features_nodes, n_classes, n_hidden, dropout, device):
        super(GNNsup2, self).__init__()

        self.device = device

       # Define the linear layers for the final prediction
        self.linear = nn.Linear( 2 * n_hidden[5], n_hidden[6])
        self.linear2 = nn.Linear(n_hidden[6], n_hidden[7])
        self.linear3 = nn.Linear(n_hidden[7], n_classes)

        # Define the dropout laye£r
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):

        # Perform linear transformation for final prediction
        u = u.squeeze(-1)
        u = self.linear(u)
        u = F.relu(u)
        u = self.linear2(u)
        u = F.relu(u)
        u = self.linear3(u)
        return(u)
    
def copy_weights(model_old,model_new):
    for name, param in model_old.named_children():
        if hasattr(model_new, name):
            getattr(model_new, name).load_state_dict(param.state_dict())

torch.save(model.state_dict(), "GNN-SSP-20-20.pt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model1 = GNNsup1(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)
model2 = GNNsup2(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)

copy_weights(model,model1)
copy_weights(model,model2)

traced_script_module = torch.jit.trace(model2, (torch.ones(128)).to(device))
traced_script_module.save("../../../../trained_models/SSP/model_prediction-GPU20-20.pt")
traced_script_module = torch.jit.trace(model1, (graphs_training[0].x.to(device), graphs_training[0].edge_index.to(device), graphs_training[0].edge_attr.to(device)))
traced_script_module.save("../../../../trained_models/SSP/model_graph_representation-GPU20-20.pt")

#CPU

device = torch.device('cpu')
model1 = GNNsup1(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)
model2 = GNNsup2(n_features_nodes=4, n_classes=1, n_hidden=[32,16,32, 64, 128, 64, 32, 16, 16], dropout=0.15, device=device).to(device)

copy_weights(model,model1)
copy_weights(model,model2)

traced_script_module = torch.jit.trace(model2, (torch.ones(128)).to(device))
traced_script_module.save("../../../../trained_models/SSP/model_prediction-CPU20-20.pt")
traced_script_module = torch.jit.trace(model1, (graphs_training[0].x.to(device), graphs_training[0].edge_index.to(device), graphs_training[0].edge_attr.to(device)))
traced_script_module.save("../../../../trained_models/SSP/model_graph_representation-CPU20-20.pt")


wandb.finish()