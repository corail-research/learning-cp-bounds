import random as rd
import numpy as np

def generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition):
    # generate the SSP
    # nb_items: number of items
    # nb_constraints: number of constraints
    # prop_final_state: proportion of final state
    # prop_transition: proportion of undefined transition
    # return: the SSP

    states = [i for i in range(nb_states)]

    values = [1 for i in range(nb_values)]

    # generate the final state
    final_states = rd.sample(states, k=int(nb_states*prop_final_state))

    # generate the transition
    index_transition = [[i for i in range(nb_states * nb_values)] for j in range(nb_constraints)]
    transition = [[rd.randint(0, nb_states - 1) for i in range (nb_states * nb_values)] for j in range(nb_constraints)]

    # randomly set to -1 prop_transition of the transitions (undefined transitions)
    index_transition_s = [rd.sample(index_transition[j], int(prop_transition * nb_states * nb_values)) for j in range(nb_constraints)] 
    for j in range(nb_constraints):
        for i in index_transition_s[j]:
            transition[j][i] = -1

    # generate the profits
    profits = [(list(np.random.randint(1, 100, size=(nb_values)))) for j in range(nb_items)]
    profits_sorted = [sorted(profits[j]) for j in range(nb_items)]

    problem = []
    problem.append(nb_constraints)
    problem.append(nb_items)
    problem.append(nb_values)
    problem.append(nb_states)
    problem.append(int(nb_states * prop_final_state))
    problem.append(initial_states)
    problem += final_states
    for j in range(nb_items):
        problem += profits_sorted[j]
    for i in range(nb_items):
        problem += values
    for j in range(nb_constraints):
        problem += transition[j]
    
    return problem

with open("train/ssp-data-trainset10-20.txt",'w') as f:

    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 200):
        problem, problem_sorted =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem_sorted)-1):
            line=line+str(int(problem_sorted[j]))+","
        line += str(int(problem_sorted[-1]))
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
    for i in range (0, 200):
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)

        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)
    line=""
    for i in range(len(problem)-1):
        line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
    f.write(line)

with open("train/ssp-data-trainset10-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 80
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 200):
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)
    line=""
    for i in range(len(problem_sorted)-1):
        line=line+str(int(problem_sorted[i]))+","
        line += str(int(problem_sorted[-1]))
    f.write(line)


with open("train/ssp-data-valset10-20.txt",'w') as f:

    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 30):
        problem, problem_sorted =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem_sorted)-1):
            line=line+str(int(problem_sorted[j]))+","
        line += str(int(problem_sorted[-1]))
        line += "\n"
        f.write(line)

with open("train/ssp-data-valset20-20.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 20
    nb_states = 20
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 30):
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)

        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)
    line=""
    for i in range(len(problem)-1):
        line=line+str(int(problem[i]))+","
        line += str(int(problem[-1]))
    f.write(line)

with open("train/ssp-data-valset10-80.txt",'w') as f:
    
    nb_constraints=2
    nb_items = 50
    nb_values = 10
    nb_states = 80
    initial_states = 0
    prop_final_state = 0.5
    prop_transition = 0.3
    for i in range (0, 30):
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)
    line=""
    for i in range(len(problem_sorted)-1):
        line=line+str(int(problem_sorted[i]))+","
        line += str(int(problem_sorted[-1]))
    f.write(line)



for i in range (0, 50):
    with open("test/ssp-data-testset10-20-" + str(i) +".txt",'w') as f:

        nb_constraints=2
        nb_items = 50
        nb_values = 10
        nb_states = 20
        initial_states = 0
        prop_final_state = 0.5
        prop_transition = 0.3
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

for i in range (0, 50):
    with open("test/ssp-data-testset20-20-" + str(i) + ".txt",'w') as f:
    
        nb_constraints=2
        nb_items = 50
        nb_values = 20
        nb_states = 20
        initial_states = 0
        prop_final_state = 0.5
        prop_transition = 0.3
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)

for i in range (0, 50):
    with open("test/ssp-data-testset10-80-" + str(i) + ".txt",'w') as f:
    
        nb_constraints=2
        nb_items = 50
        nb_values = 10
        nb_states = 80
        initial_states = 0
        prop_final_state = 0.5
        prop_transition = 0.3
        problem =generate_ssp(nb_items,nb_constraints, initial_states, nb_states, nb_values, prop_final_state, prop_transition)
        line=""
        for j in range(len(problem)-1):
            line=line+str(int(problem[j]))+","
        line += str(int(problem[-1]))
        line += "\n"
        f.write(line)