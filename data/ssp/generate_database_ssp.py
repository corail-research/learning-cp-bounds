import random as rd
import numpy as np
import matplotlib.pyplot as plt

def generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition):
    # generate the SSP
    # nb_items: number of items
    # nb_constraints: number of constraints
    # prop_final_state: proportion of final state
    # prop_transition: proportion of transition
    # return: the SSP

    states = [i for i in range(nb_states)]

    values = [i for i in range(nb_values)]

    # generate the final state
    final_states = rd.sample(states, k=int(nb_states*prop_final_state))
    # generate the transition

    index_transition = [i for i in range(nb_constraints * nb_states * nb_values * nb_states)]
    transition = np.array([1] * nb_constraints * nb_states * nb_values * nb_states)

    profits = rd.randint(1, 100, size=(nb_items * nb_values))

    # randomly set to 0 prop_transition of the transitions
    index_transition = rd.sample(index_transition, int(nb_constraints * nb_states * nb_values * nb_states * prop_transition))
    for i in index_transition:
        transition[i] = 0

    problem = []
    problem.append(nb_constraints)
    problem.append(nb_items)
    problem.append(nb_values)
    problem.append(nb_states)
    problem.append(nb_states * prop_final_state)
    problem.append(initial_states)
    problem += final_states
    problem += profits
    problem += values
    problem += transition.tolist()
    
    return problem 

with open("train/ssp-data-trainset10-20.txt",'w') as f:

    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("train/ssp-data-trainset10-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 80
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("train/ssp-data-trainset20-20.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 20
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("train/ssp-data-trainset20-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 20
    nb_states = 80
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)


with open("test/ssp-data-testset10-20.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("test/ssp-data-testset10-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 80
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("test/ssp-data-testset20-20.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 20
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

with open("test/ssp-data-testset20-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 5):
        tightness=np.array([0.2 + 0.05 *j for j in range(nb_constraints)])
        problem=generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for i in range(len(problem)-1):
            line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)