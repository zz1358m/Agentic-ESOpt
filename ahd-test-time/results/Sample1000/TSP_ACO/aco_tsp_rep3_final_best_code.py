# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_sample_t1000_aco_tsp_sample_t1000_rep3_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_tsp, rep: 3
# train_objective: 5.95606

import numpy as np
import random

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix, dtype=float)
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = (1 / distance_matrix[i, j]) ** 2
    heuristics_matrix += np.eye(n)
    heuristics_matrix /= heuristics_matrix.sum(axis=1, keepdims=True)
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    heuristics_matrix = heuristics(distance_matrix)
    return heuristics_matrix

# Example usage:
distance_matrix = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
print(heuristics_v2(distance_matrix))
