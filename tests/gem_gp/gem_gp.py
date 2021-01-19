import os
import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
import csv

N_runs = 625
frac_train = 0.5
input_dim = 4
output_dim = 2

input_samples = np.zeros((N_runs, input_dim))
output_samples = np.zeros((N_runs, output_dim))

with open('', 'r') as inputfile:
    datareader = csv.reader(inputfile, delimeter=',')
    i = 0
    for row in datareader:
        input_samples[i] = row[0:input_dim-1]
        output_samples[i] = row[input_dim:input_dim+output_dim]
        i = i + 1
