import os
import pandas as pd
import numpy as np


def extract_data(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    nodes = []
    nodes_per_models = []
    times = []
    times_per_models = []
    nodes_per_sizes = []
    times_per_sizes = []
    index = 0
    
    for line in lines:
        # if index == 2:
        #     break
        line = line.strip()
        if line == "separateur_de_modeles":
            index += 1
            if nodes_per_models:
                nodes_per_sizes.append(nodes_per_models)
                nodes_per_models = []
            if times_per_models:
                times_per_sizes.append(times_per_models)
                times_per_models = []
        elif line == "separateur de taille":
            if nodes_per_sizes:
                nodes.append(nodes_per_sizes)
                nodes_per_sizes = []
            if times_per_sizes:
                times.append(times_per_sizes)
                times_per_sizes = []
        elif line.startswith("runtime:"):
            time = float(line.split(":")[1].strip().split(' ')[0].strip())
            times_per_models.append(time)
        elif line.startswith("nodes"):
            node = int(line.split(":")[1].strip())
            nodes_per_models.append(node)
    
    # Final append if the last batch was not added
    if nodes_per_models:
        nodes_per_sizes.append(nodes_per_models)
    if nodes_per_sizes:
        nodes.append(nodes_per_sizes)
    
    if times_per_models:
        times_per_sizes.append(times_per_models)
    if times_per_sizes:
        times.append(times_per_sizes)
    
    return nodes, times

models = ["model1", "model2", "model3", "model4", "model5"]
sizes = ["30", "50", "100", "200"]

nodes, times = extract_data("solutions.txt")

data_nodes = {}
data_times = {}

# Convert lists to numpy arrays for easy manipulation
nodes_array = np.array(nodes, dtype=np.float64)
times_array = np.array(times, dtype=np.float64)
print(times_array)
print(times_array.shape)

# Average over the third dimension (the model dimension)
nodes_data = np.mean(nodes_array, axis=2)
times_data = np.mean(times_array, axis=2)

for i, nodes_per_sizes in enumerate(nodes_data):
    data_nodes[sizes[i]] = nodes_per_sizes

for i, times_per_sizes in enumerate(times_data):
    data_times[sizes[i]] = times_per_sizes

# Create DataFrames from the dictionaries
df_nodes = pd.DataFrame(data_nodes, index=models)
df_times = pd.DataFrame(data_times, index=models)

# Save the DataFrames to Excel files
df_nodes.to_excel('nodes.xlsx', index=True)
df_times.to_excel('times.xlsx', index=True)
