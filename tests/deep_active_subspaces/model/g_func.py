#!/usr/bin/env python3

# scalar analytic test function, defined on [0, 1]**d
import numpy as np
import json
import sys


def poly_model(theta):

    sol = 1.0
    for i in range(D):
        # sol *= 2.0 * (np.abs(4.0 * theta[i] - 2.0) + a[i]) / (1.0 + a[i])
        sol *= (3 * a[i] * theta[i]**2 + 1.0) / 2**D
    return sol

# the json input file containing the values of the parameters, and the
# output file
json_input = sys.argv[1]

with open(json_input, "r") as f:
    inputs = json.load(f)
output_filename = inputs['outfile']

# stocastic dimension of the problem
D = inputs['D']

# load the a vector, which determines the importance of each input parameter
a = np.zeros(D)
for i in range(D):
    a[i] = float(inputs['a%d' % (i + 1)])

# load the input parameter values (x)
theta = np.zeros(D)
for i in range(D):
    theta[i] = float(inputs['x%d' % (i + 1)])

# run the model
result = poly_model(theta)

# output csv file
header = 'f'
np.savetxt(output_filename, np.array([result]),
           delimiter=",", comments='',
           header=header)
