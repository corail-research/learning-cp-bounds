import numpy as np
bool = True
n = 200
m = 5
with open("/home/darius/scratch/learning-bounds/data/mknapsack/train/pissinger/trainset-artificial-200-subnodes.txt",'w') as f:
    with open("/home/darius/scratch/learning-bounds/data/mknapsack/train/pissinger/knapsacks-data-trainset200.txt",'r') as file:
                lines = file.readlines()
                for line in lines:
                    probleme = line.split(sep = ',')
                    for k in range(n-1, 5, -1):
                        for p in range(1):
                            problem = []
                            problem.append(m)
                            problem.append(k)
                            for idx in range(2, k + 2):
                                problem.append(int(probleme[n - k + idx]) + np.random.randint(-10, 10))
                            p = (-k +n) / n
                            binary_variables = np.random.choice([1, 0], n - k, p=[1 - p,  p])
                            for idx_capacity in range(n + 2, n + m + 2):
                                cap_temp = 0
                                for i in range(n - k):
                                    cap_temp += int(probleme[2 + n + m + (idx_capacity -n - 2) * n + i]) * binary_variables[i]
                                if cap_temp > int(probleme[idx_capacity]):
                                     bool = False
                                problem.append(int(probleme[idx_capacity]) - cap_temp)
                            for i in range(m): 
                                for j in range(0, k):
                                    problem.append(max(0,int(probleme[n + m + 2 + n - k + j + i*n]) + np.random.randint(-10, 10)))
                            fix_bound = 0
                            for i in range(n - k):
                                fix_bound += int(probleme[2 + i]) * binary_variables[i]
                            problem.append(fix_bound)
                            if bool:
                                line_w = str(problem[0])
                                for i in range(1, len(problem)):
                                    line_w += ',' + str(problem[i])
                                f.write(line_w + '\n') 
                            bool = True                          

