# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_t1000_aco_cvrp_sample_t1000_rep3_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_cvrp, rep: 3
# train_objective: 10.39134

import numpy as np
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    node_prob = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                edge_prob = 0.4 / distance_matrix[i, j] * demands[j] / capacity
                node_prob[i, j] = edge_prob
    return node_prob

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    node_prob = heuristics(distance_matrix, coordinates, demands, capacity)
    heuristic_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j and i!= 0 and j!= 0:
                heuristic_matrix[i, j] = np.max([node_prob[0, i], node_prob[i, j]])
            elif i!= j and i == 0:
                heuristic_matrix[i, j] = node_prob[i, j]
            elif i!= j and j == 0:
                heuristic_matrix[i, j] = node_prob[i, j]
    return heuristic_matrix

# Test the function
distance_matrix = np.array([[0, 10, 15, 20],
                           [10, 0, 35, 25],
                           [15, 35, 0, 30],
                           [20, 25, 30, 0]])

coordinates = np.array([[0, 0], [10, 0], [5, 5], [0, 10]])

demands = np.array([0, 10, 20, 30])

capacity = 30

heuristic_matrix = heuristics_v2(distance_matrix, coordinates, demands, capacity)
print(heuristic_matrix)
