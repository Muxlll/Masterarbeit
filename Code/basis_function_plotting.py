import numpy as np

def func_x0(x):
    return max(1-abs(x/(0.25) - 1), 0)

def func_x1(x):
    return max(1-abs(x/(0.5) - 1), 0)

text_file = open("data.dat", "w")


coordinates = ""

samples_per_dim = 50

x0 = 0.0
x1 = 0.0

for i in range(samples_per_dim+1):
    x0 = 0.0
    for j in range(samples_per_dim+1):
        coordinates += str(x0) + "\t" + str(x1) + "\t" + str(func_x1(x0) * func_x0(x1)) + "\n"
        x0 += 1/samples_per_dim

    coordinates += "\n"
    x1 += 1/samples_per_dim


#write string to file
text_file.write(coordinates)

#close file
text_file.close()