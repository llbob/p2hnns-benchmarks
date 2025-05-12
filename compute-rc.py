import math
import sys
import h5py
import numpy as np
import random
from p2hnns_benchmarks.distance import euc_p2hdist, ang_p2hdist

f = h5py.File(sys.argv[1])

# default: relate average distance to distance of 10-th NN
k = 10

if len(sys.argv) > 2:
    k = int(sys.argv[2])

samples = 1000
m = len(f['normals'])

distances = np.array(f['distances'])

normals = np.array(f['normals'])
biases = np.array(f['biases'])
dataset = np.array(f['points'])

estimates = np.zeros(m, dtype=float)

# sample random points from the dataset
random_matrix = np.array([random.choice(f['points']) for _ in range(samples)])

# calculate average distances from random points to each hyperplane
hyperplane_avg_distances = np.zeros(m, dtype=float)


for i in range(m):
    hyperplane = (normals[i], biases[i])
    distances_to_hyperplane = np.array([euc_p2hdist(x, hyperplane) for x in random_matrix])
    hyperplane_avg_distances[i] = np.mean(distances_to_hyperplane)

assert len(hyperplane_avg_distances) == len(normals)


for i in range(len(normals)):
    for j in range(k - 1, 100):
        if distances[i][j] > 1e-6:
            dist = distances[i][j]
            break

    estimates[i] = (hyperplane_avg_distances[i] / dist)

for i, e in enumerate(estimates):
    print(i, e)

print("rc average:", np.mean(estimates))



