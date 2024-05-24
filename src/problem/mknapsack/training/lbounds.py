import json
from pathlib import Path
import pandas as pd
import os
import sklearn as sk
import numpy as np
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

from numba import cuda

def load_dataset(data_split):
    """
    Function to load the dataset from the text files
    :param data_split: Path to the folder containing the text
    :return: List of graphs
    """
    graphs = []      
    graph_id = 0

  # Loop through all the files in the folder
    for file_path in os.listdir(data_split):
        print(file_path)
        if file_path.endswith('.txt'):
        # Open the file
            with open(data_split+ '/' + file_path, 'r') as file:
            # A line is a node of the exploration tree
                for line in file.readlines():
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
                    X = [] # Nodes of the problem graph
                    edge_index = [] # Edges of the problem graph
                    edge_weights = [] # Weights of the edges of the problem graph
                    edge_attributes = []

                # Create the nodes of the graph (one node per couple variable, constraint)
                    for j in range(m):
                        for i in range(n):
                        # The node contains the following features:
                        # - The weight of the variable in the constraint
                        # - The profit of the variable
                        # - The capacity of the constraint
                        # - The ratio profit/capacity
                            X.append([problem[2+i], problem[2 + n + m + j*n + i], max(problem[2 + n + m + j*n + i], 1)/ max(problem[2 + n + j],1) , problem[2 + i] / max(problem[2 + n + m + j*n + i], 1), i, j])
                # Create the edges of the graph between nodes that share the same constraint,
                    for k in range(m):
                        for i in range(n):
                            for j in range(i+1, n):
                                edge_index.append([k * n + i, k*n +  j])
                                edge_weights.append(1/n)
                                edge_attributes.append(([1,0]))
                # Create the edges of the graph between nodes that share the same variable
                    for k in range(n):
                        for i in range(1, m):
                            edge_index.append([k + i*n, k])
                            edge_weights.append(1)
                            edge_attributes.append([0,1])

                    graph_id = problem[0] * problem[1]

                    graph = torch_geometric.data.Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(edge_index).T,
                      edge_weight=torch.FloatTensor(edge_weights), edge_attr=torch.LongTensor(edge_attributes), solutions = problem[-1], subgrad_bound = problem[-2], fix_bound = problem[-3],  graph_id=graph_id, graph_problem=problem)

                    graphs.append(graph)
    return graphs

@cuda.jit
def dp_knapsack_gpu_batch(Lglobal_bound,capacities, Lweights, Lval, N, M,value_var_solution, Ldp):
    idx_batch = 0
    idx_thread = cuda.grid(1)
    while idx_thread >= M[idx_batch]:
        idx_thread -= M[idx_batch]
        idx_batch += 1
    idx_constraint = idx_thread
    offset_n_m = 0
    offset_m = 0
    for i in range(idx_batch):
        offset_n_m += N[i] * M[i]
        offset_m += M[i]

    capacity=capacities[offset_m + idx_constraint]
    weights=Lweights[offset_m + idx_constraint]
    val=Lval[offset_m + idx_constraint]
    for i in range(1, N[idx_batch] + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                Ldp[offset_m + idx_constraint][w][i] = max(Ldp[offset_m + idx_constraint][w][i - 1], Ldp[offset_m + idx_constraint][w - weights[i - 1]][i - 1] + val[i - 1])
            else:
                Ldp[offset_m + idx_constraint][w][i] = Ldp[offset_m + idx_constraint][w][i - 1]

    # Backtracking to find selected items
    w = capacity
    for i in range(N[idx_batch], 0, -1):
        if Ldp[offset_m + idx_constraint][w][i] != Ldp[offset_m + idx_constraint][w][i - 1]:
            value_var_solution[offset_n_m + i - 1 + N[idx_batch] * idx_constraint] = 1
            w -= weights[i - 1]

    Lglobal_bound[0]+=Ldp[offset_m + idx_constraint][capacity][N[idx_batch]]

def solve_knapsack_gpu_batch(problems, u):
    batch_size = len(problems)
    compteur_n_m = 0
    compteur_m = 0
    N = []
    M = []
    value_var_solution = []
    Lweights = []
    Lval = []
    capacities=[]
    for idx_batch in range(batch_size):
        N.append(problems[idx_batch][1]) #nb_items
        M.append(problems[idx_batch][0]) #nb_constraints

    for idx_batch in range(batch_size):
        n = N[idx_batch]
        m = M[idx_batch]
        value_var_solution += [0] * N[idx_batch] * M[idx_batch]
        Lweights+=[0]*m
        Lval+=[0]*m
        u_1 =[]
        for i in range(N[idx_batch]):
            temp = 0
            for j in range(1, M[idx_batch]):
                temp += u[compteur_n_m + j * N[idx_batch] + i]
            u_1.append(temp)

        for idx_constraint in range(M[idx_batch]):
            val = []
            capacity = problems[idx_batch][2 + N[idx_batch] + idx_constraint]
            for i in range(max(N)):
                if i < N[idx_batch]:
                    if idx_constraint == 0:
                        val.append( u_1[i] +  problems[idx_batch][2 + i])
                    else:
                        val.append(- u[compteur_n_m + idx_constraint * N[idx_batch] + i ])
                else:
                    val.append(0)

            weights = []
            for i in range(max(N)):
                if i < N[idx_batch]:
                    weights.append(problems[idx_batch][2 + n + m + idx_constraint*n + i])
                else:
                    weights.append(0)
            Lweights[compteur_m + idx_constraint]=weights
            Lval[compteur_m + idx_constraint]=val
            capacities.append(capacity)
        compteur_n_m += N[idx_batch] * M[idx_batch]
        compteur_m += M[idx_batch]
    device = torch.device("cuda")
    Lglobal_bound=cuda.to_device(np.array([0]))
    Ldp=cuda.to_device(np.zeros((len(capacities),np.max(capacities)+1,max(N) + 1)))
    value_var_solution = cuda.to_device(np.array(value_var_solution))

    dp_knapsack_gpu_batch[compteur_m , 1](Lglobal_bound,cuda.to_device(np.array(capacities)), cuda.to_device(np.array(Lweights)), cuda.to_device(np.array(Lval)), cuda.to_device(np.array(N)), cuda.to_device(np.array(M)),value_var_solution, Ldp)
    solutions = value_var_solution.copy_to_host()
    return solutions

class GNN(torch.nn.Module):
    def __init__(self, n_features_nodes, n_classes, n_hidden, dropout, device):
        super(GNN, self).__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)

        # Define the linear layers for the graph embedding
        self.linear_embedding = nn.Linear(n_features_nodes, n_hidden[0])
        self.linear_embedding2 = nn.Linear(n_hidden[0], n_hidden[1])

        # Define the graph convolutional layers
        self.conv1 = gnn.GatedGraphConv(  n_hidden[2], 2)
        self.conv2 = gnn.GatedGraphConv( n_hidden[3], 2)

        # Define the linear layers for the graph embedding
        self.linear_graph1 = nn.Linear(n_hidden[3], n_hidden[4])
        self.linear_graph2 = nn.Linear(n_hidden[4], n_hidden[5])

        # Define the linear layers for the final prediction
        self.linear = nn.Linear( 2 * n_hidden[5], n_hidden[6])
        self.linear2 = nn.Linear(n_hidden[6], n_hidden[7])
        self.linear3 = nn.Linear(n_hidden[7], n_classes)

        # Define the dropout laye£r
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, edge_index, edge_weight, edge_attribute, problems):
        # Perform graph convolution and activation function
        G = self.linear_embedding(G)
        G = F.relu(G)
        G = self.linear_embedding2(G)
        G = F.relu(G)


        G = self.conv1(G, edge_index, edge_weight)
        G = F.relu(G)
        G = self.conv2(G, edge_index, edge_weight)
        G = F.relu(G)

        # Perform linear transformation and activation function
        G = self.linear_graph1(G)
        G = F.relu(G)
        G = self.linear_graph2(G)

        graph_embeddings = []
        compteur = 0
        for problem in problems:
            graph_embeddings.append(torch.mean(G[compteur: compteur + problem[0] * problem[1]], dim = 0))
            compteur += problem[0] * problem[1]
        u = []

        compteur = 0

        # Concatenate the graph embeddings with the nodes of the problem
        for i in range(len(problems)):
            for j in range(compteur, compteur + problems[i][0] * problems[i][1]):
                u.append(torch.cat((G[j], graph_embeddings[i])))
            compteur += problems[i][0] * problems[i][1]
        u = torch.stack(u)

        # Perform linear transformation for final prediction
        u = u.squeeze(-1)
        #u = self.dropout(u)
        u = self.linear(u)
        u = F.relu(u)
        u = self.linear2(u)
        u = F.relu(u)
        u = self.linear3(u)


        u = u.to(torch.device('cpu'))
        u2=list(torch.clone(u).squeeze(-1).detach().numpy())
        u = u.to(self.device)

        # Solve the knapsack problem for each graph of the batch
        sum_fixed = 0
        compteur = 0
        bounds = torch.zeros(len(problems))
        value_var_solutions = solve_knapsack_gpu_batch(problems, u2)
        for idx_batch, problem in enumerate(problems):
            dx = [0] * (problem[1] * problem[0])
            for i in range(problem[1]):
                for j in range( problem[0]):
                    dx[i + j * problem[1] ] = (- value_var_solutions[compteur + i + j * problem[1]] + value_var_solutions[compteur + i])
            dx = torch.Tensor(dx).to(self.device)
            u_temp = torch.Tensor(u[compteur: compteur + problem[0] * problem[1]]).squeeze(-1)
            bound = torch.dot(u_temp , dx)
            temp = torch.zeros(1)
            for i in range(problem[1]):
                temp += problem[2 + i] * value_var_solutions[compteur + i]

            bound += temp.squeeze(-1)
            bounds[idx_batch] = bound
            compteur += problem[0] * problem[1]
        return bounds
    
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
        train_loss_sublist = []
        train_diff_ecart_opt_sublist = []
        for data in train_loader:
            model.train()
            optimizer.zero_grad()
            bound = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem)
            loss = criterion(bound)
            loss.backward()
            optimizer.step()
            y = data.opt.to('cpu')
            gobal_bound = data.fix_bound.to('cpu') + bound.to('cpu')
            train_loss_sublist.append(torch.mean(gobal_bound).detach().item())
            train_diff_ecart_opt_sublist.append((torch.mean(torch.div(gobal_bound - y, y))).detach().item())

        train_loss.append(np.mean(train_loss_sublist))
        train_diff_ecart_opt.append(np.mean(train_diff_ecart_opt_sublist))
        
        val_loss_sublist = []
        val_diff_ecart_opt_sublist = []
        for data in val_loader:
            model.eval()
            with torch.no_grad():
                bound = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem)
                loss = criterion(bound)
                y = data.opt.to('cpu')
                gobal_bound = data.fix_bound.to('cpu') + bound.to('cpu')
                val_loss_sublist.append(torch.mean(gobal_bound).detach().item())
                val_diff_ecart_opt_sublist.append((torch.mean(torch.div(gobal_bound - y, y))).detach().item())


        val_loss.append(np.mean(val_loss_sublist))
        val_diff_ecart_opt.append(np.mean(val_diff_ecart_opt_sublist))

    #check if scheduler is none
        if scheduler is not None:
            scheduler.step(val_loss[-1])

    #, val loss {val_loss[-1]}, , val diff {val_diff[-1]}
        if epoch%5 ==0:
            print(f"Epoch {epoch} : "
      f"train loss {train_loss[-1]},  val loss {val_loss[-1]},\n "
      f"train diff ecart opt {train_diff_ecart_opt[-1] * 100}%, "
      f"val diff ecart opt {val_diff_ecart_opt[-1] * 100}%,\n"
                  
                 )


    return train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt

def plotter(train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt):
    """
  Function to plot the training and validation metrics.

  Args:
  - train_loss: List of training losses for each epoch
  - val_loss: List of validation losses for each epoch
  - train_acc: List of training accuracies for each epoch
  - val_acc: List of validation accuracies for each epoch
  - train_f1: List of training F1 scores for each epoch
  - val_f1: List of validation F1 scores for each epoch
    """

    fig, axs = plt.subplots(1, 2, figsize=(20,5))
    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(val_loss, label="Val Loss")
    axs[0].axhline(y=3200, color='r', linestyle='-', label="Baseline")
    axs[0].legend()

    axs[1].plot(train_diff_ecart_opt, label="Train Diff Global Bound")
    axs[1].plot(val_diff_ecart_opt, label="Val Diff Global Bound")
    axs[1].legend()

    plt.show()


## Train the models
graphs_training = load_dataset('../../../../data/mknapsack/train/pissinger/')
graphs_test = load_dataset('../../../../data/mknapsack/train/pissinger/')

# Create train_set and val_set
train_data, val_data = train_test_split(graphs_training, test_size=0.2, random_state = 0)
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def criterion(bounds):
    return torch.mean(bounds)

model = GNN(n_features_nodes=6, n_classes=1, n_hidden=[128, 8, 64, 64, 128, 128, 32, 32], dropout=0.15, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt = train(model, optimizer, criterion, scheduler, val_loader, train_loader, 500, device)

## Print the results

# plot
plotter(train_loss, val_loss, train_diff_ecart_opt, val_diff_ecart_opt)

## Save the models

class GNNsup1(torch.nn.Module):
    def __init__(self, n_features_nodes, n_classes, n_hidden, dropout, device):
        super(GNNsup1, self).__init__()

        self.device = device

       # Define the linear layers for the graph embedding
        self.linear_embedding = nn.Linear(n_features_nodes, n_hidden[0])
        self.linear_embedding2 = nn.Linear(n_hidden[0], n_hidden[1])

        # Define the graph convolutional layers
        self.conv1 = gnn.GatedGraphConv(  n_hidden[2], 2)
        self.conv2 = gnn.GatedGraphConv( n_hidden[3], 2)

        # Define the linear layers for the graph embedding
        self.linear_graph1 = nn.Linear(n_hidden[3], n_hidden[4])
        self.linear_graph2 = nn.Linear(n_hidden[4], n_hidden[5])

        # Define the dropout laye£r
        self.dropout = nn.Dropout(dropout)

    def forward(self, G, edge_index, edge_weight):
        # Perform graph convolution and activation function
        #print(G)
        G = self.linear_embedding(G)
        G = F.relu(G)
        G = self.linear_embedding2(G)
        G = F.relu(G)

        G = self.conv1(G, edge_index, edge_weight)
        G = F.relu(G)
        G = self.conv2(G, edge_index, edge_weight)
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

torch.save(model.state_dict(), "GNN.pt")

device = torch.device('cpu')
model1 = GNNsup1(n_features_nodes=6, n_classes=1, n_hidden=[128, 8, 64, 64, 128, 128, 32, 32], dropout=0.15, device=device).to(device)
model2 = GNNsup2(n_features_nodes=6, n_classes=1, n_hidden=[128, 8, 64, 64, 128, 128, 32, 32], dropout=0.15, device=device).to(device)

copy_weights(model,model1)
copy_weights(model,model2)

traced_script_module = torch.jit.trace(model2, (torch.ones(64)))
traced_script_module.save("../../../../trained_models/mknapsack/model_graph_representation.pt")
traced_script_module = torch.jit.trace(model1, (graphs_test[0].x.to(device), graphs_test[0].edge_index.to(device), graphs_test[0].edge_attr.to(device)))
traced_script_module.save("../../../../trained_models/mknapsack/model_prediction.pt")
