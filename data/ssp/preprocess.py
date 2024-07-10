import os

with open('/home/darius/scratch/learning-bounds/data/ssp/train/ssp-data10-20.txt', 'w') as file_output:
    for i in range(1, 201):
        with open("/home/darius/scratch/learning-bounds/data/ssp/train/ssp-data-trainset10-20-subnodes" + str(i) + ".txt", 'r') as file_input:
            lines = file_input.readlines()
            N = len(lines)
            for j in range(N-1):
                    file_output.write(lines[j])
            file_input.close()

file_output.close()
