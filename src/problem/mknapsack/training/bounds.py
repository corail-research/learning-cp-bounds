import argparse
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import DataLoader
import torch_geometric.nn as gnn

from numba import cuda

import wandb

# Créer un analyseur d'arguments
parser = argparse.ArgumentParser(description="Get the size of the problem")

# Ajouter des arguments
parser.add_argument('size_instances', type=str, help='number of items of the instances')

# Analyser les arguments
args = parser.parse_args()

size_instances = args.size_instances

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
        # Create a new graph object
            try:
                graph = gnn.data.Data()
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

                graph = gnn.data.Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(edge_index).T,
                edge_weight=torch.FloatTensor(edge_weights), edge_attr=torch.LongTensor(edge_attributes), opt = problem[-1],  fix_bound = problem[-2],  graph_id=graph_id, graph_problem=problem)

                graphs.append(graph)
            except:
                print('error')
    return graphs

@cuda.jit
def dp_knapsack_gpu_batch(Lglobal_bound, capacities, Lweights, Lval, N, M, value_var_solution, Ldp):
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

    capacity = capacities[offset_m + idx_constraint]
    weights = Lweights[offset_m + idx_constraint]
    val = Lval[offset_m + idx_constraint]
    for i in range(1, N[idx_batch] + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                Ldp[offset_m + idx_constraint][w][i] = max(Ldp[offset_m + idx_constraint][w][i - 1],
                                                           Ldp[offset_m + idx_constraint][w - weights[i - 1]][i - 1] + val[i - 1])
            else:
                Ldp[offset_m + idx_constraint][w][i] = Ldp[offset_m + idx_constraint][w][i - 1]

    # Backtracking to find selected items
    w = capacity
    for i in range(N[idx_batch], 0, -1):
        if Ldp[offset_m + idx_constraint][w][i] != Ldp[offset_m + idx_constraint][w][i - 1]:
            value_var_solution[offset_n_m + i - 1 + N[idx_batch] * idx_constraint] = 1
            w -= weights[i - 1]

    cuda.atomic.add(Lglobal_bound, 0, Ldp[offset_m + idx_constraint][capacity][N[idx_batch]])

def solve_knapsack_gpu_batch(problems, u):
    batch_size = len(problems)
    compteur_m = 0
    compteur_n_m = 0
    N = []
    M = []
    value_var_solution = []
    Lweights = []
    Lval = []
    capacities = []
    
    for idx_batch in range(batch_size):
        N.append(problems[idx_batch][1])  # nb_items
        M.append(problems[idx_batch][0])  # nb_constraints

    max_n = max(N)
    
    for idx_batch in range(batch_size):
        n = N[idx_batch]
        m = M[idx_batch]
        value_var_solution.extend([0] * n * m)
        Lweights.extend([0] * m)
        Lval.extend([0] * m)
        u_1 = []
        for i in range(n):
            temp = 0
            for j in range(1, m):
                temp += u[compteur_n_m + j * n + i].item()  # Convert to scalar
            u_1.append(temp)

        for idx_constraint in range(m):
            val = []
            capacity = problems[idx_batch][2 + n + idx_constraint]
            for i in range(max_n):
                if i < n:
                    if idx_constraint == 0:
                        val.append(u_1[i] + problems[idx_batch][2 + i])
                    else:
                        val.append(-u[compteur_n_m + idx_constraint * n + i].item())  # Convert to scalar
                else:
                    val.append(0)

            weights = []
            for i in range(max_n):
                if i < n:
                    weights.append(problems[idx_batch][2 + n + m + idx_constraint * n + i])
                else:
                    weights.append(0)
            Lweights[compteur_m + idx_constraint] = weights
            Lval[compteur_m + idx_constraint] = val
            capacities.append(capacity)
        compteur_n_m += N[idx_batch] * M[idx_batch]
        compteur_m += M[idx_batch]

    # Convert lists to PyTorch tensors and then to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Lglobal_bound = torch.tensor([0], dtype=torch.int32, device=device)
    value_var_solution = torch.tensor(value_var_solution, dtype=torch.int32, device=device)

    Lweights_tensor = torch.tensor(Lweights, dtype=torch.int32, device=device)
    Lval_tensor = torch.tensor(Lval, device=device)
    capacities_tensor = torch.tensor(capacities, dtype=torch.int32, device=device)
    Ldp = torch.zeros((len(capacities), torch.max(capacities_tensor) + 1, max(N) + 1), dtype=torch.int32, device=device)
    N_tensor = torch.tensor(N, dtype=torch.int32, device=device)
    M_tensor = torch.tensor(M, dtype=torch.int32, device=device)

    dp_knapsack_gpu_batch[compteur_m.item(), 1](Lglobal_bound, 
                                                  capacities_tensor, 
                                                  Lweights_tensor, 
                                                  Lval_tensor, 
                                                  N_tensor, 
                                                  M_tensor, 
                                                  value_var_solution, Ldp)
    return value_var_solution

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

        # Solve the knapsack problem for each graph of the batch
        compteur = 0
        bounds = torch.zeros(len(problems))
        value_var_solutions = solve_knapsack_gpu_batch(problems, u)
        for idx_batch, problem in enumerate(problems):
            dx = [0] * (problem[1] * problem[0])
            for i in range(problem[1]):
                for j in range( problem[0]):
                    dx[i + j * problem[1] ] = (- value_var_solutions[compteur + i + j * problem[1]] + value_var_solutions[compteur + i])
            dx = torch.Tensor(dx).to(self.device)
            u_temp = torch.Tensor(u[compteur: compteur + problem[0] * problem[1]]).squeeze(-1)
            bound = torch.dot(u_temp , dx)
            temp = torch.zeros(1).to(device)
            for i in range(problem[1]):
                temp += problem[2 + i] * value_var_solutions[compteur + i]

            bound += temp.squeeze(-1)
            bounds[idx_batch] = bound
            compteur += problem[0] * problem[1]
        return bounds, u

## Train the models
graphs_training = load_dataset("/home/darius/scratch/learning-bounds/data/mknapsack/test/pissinger/knapsacks-data-testset" + size_instances +".txt")
print("len(graphs_training)", len(graphs_training))

# Create train_set and val_set
train_loader = DataLoader(graphs_training, batch_size=1, shuffle=False)

torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def criterion(bounds):
    return torch.mean(bounds)

model = GNN(n_features_nodes=6, n_classes=1, n_hidden=[128, 16, 64, 64, 256, 128, 32, 32], dropout=0.15, device=device).to(device)

model.load_state_dict(torch.load("GNN-" + size_instances + ".pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=20)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bounds_learning = []

for data in train_loader:
    model.train()
    optimizer.zero_grad()
    tensor_problems = [torch.tensor(sublist) for sublist in data.graph_problem]
    data.graph_problem = [tensor.to(device) for tensor in tensor_problems]
    bound = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem)[0]
    y = data.opt.to(device)
    gobal_bound = bound.to(device)
    bounds_learning.append(torch.mean(gobal_bound).detach().item())

bounds_learning_grad = []

for data in train_loader:
    problems = data.graph_problem
    bounds_learning_grad_sublist = []
    model.train()
    optimizer.zero_grad()
    tensor_problems = [torch.tensor(sublist) for sublist in data.graph_problem]
    data.graph_problem = [tensor.to(device) for tensor in tensor_problems]
    bound, u = model(data.x.to(device), data.edge_index.to(device),data.edge_weight.to(device), data.edge_attr.to(device),data.graph_problem)
    y = data.opt.to(device)
    for k in range(1000):
        value_var_solutions = solve_knapsack_gpu_batch(data.graph_problem, u)
        compteur = 0
        for idx_batch, problem in enumerate(problems):
                dx = [0] * (problem[1] * problem[0])
                for i in range(problem[1]):
                    for j in range( problem[0]):
                        dx[i + j * problem[1] ] = (- value_var_solutions[compteur + i + j * problem[1]] + value_var_solutions[compteur + i])
                dx = torch.Tensor(dx).to(device)
                u_temp = torch.Tensor(u[compteur: compteur + problem[0] * problem[1]]).squeeze(-1)
                bound = torch.dot(u_temp , dx)
                temp = torch.zeros(1).to(device)
                for i in range(problem[1]):
                    temp += problem[2 + i] * value_var_solutions[compteur + i]

                bound += temp.squeeze(-1)
                global_bound = bound.to(device)
                bounds_learning_grad_sublist.append(bound.to('cpu').detach().item())
                compteur += problem[0] * problem[1]
        compteur = 0

        for i in range(problems[0][0]):
            for j in range(compteur, compteur + problems[0][1]):
                u[j + i * problems[0][1]] -= 1 * ( - value_var_solutions[compteur + j + i * problems[0][1]] + value_var_solutions[compteur + j])
    
    bounds_learning_grad.append(bounds_learning_grad_sublist)
        

bounds_grad = []

for data in train_loader:
    problems = data.graph_problem
    tensor_problems = [torch.tensor(sublist) for sublist in data.graph_problem]
    data.graph_problem = [tensor.to(device) for tensor in tensor_problems]
    u = torch.zeros(problems[0][0] * problems[0][1]).to(device)
    bounds_grad_sublist = []
    for k in range(1000):
        value_var_solutions = solve_knapsack_gpu_batch(data.graph_problem, u)
        compteur = 0
        for idx_batch, problem in enumerate(problems):
                dx = [0] * (problem[1] * problem[0])
                for i in range(problem[1]):
                    for j in range( problem[0]):
                        dx[i + j * problem[1] ] = (- value_var_solutions[compteur + i + j * problem[1]] + value_var_solutions[compteur + i])
                dx = torch.Tensor(dx).to(device)
                u_temp = torch.Tensor(u[compteur: compteur + problem[0] * problem[1]]).squeeze(-1)
                bound = torch.dot(u_temp , dx)
                temp = torch.zeros(1).to(device)
                for i in range(problem[1]):
                    temp += problem[2 + i] * value_var_solutions[compteur + i]

                bound += temp.squeeze(-1)
                global_bound = bound.to(device)
                bounds_grad_sublist.append(bound.to('cpu').detach().item())
                compteur += problem[0] * problem[1]
        compteur = 0

        for i in range(problems[0][0]):
            for j in range(compteur, compteur +  problems[0][1]):
                u[j + i * problems[0][1] ] -= 1 * (- value_var_solutions[compteur + j + i * problems[0][1]] + value_var_solutions[compteur + j])

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
with open("bounds_learning-" + size_instances +".txt", "w") as file:
    file.write(json.dumps(bounds_learning))

with open("bounds_learning_grad-" + size_instances +".txt", "w") as file:
    file.write(json.dumps(bounds_learning_grad))

with open("bounds_grad-" + size_instances +".txt", "w") as file:
    file.write(json.dumps(bounds_grad))