# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_sample_t2000_aco_cvrp_sample_t2000_from_rep3_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: aco_cvrp, rep: 3
# train_objective: 9.60416

import numpy as np
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    heuristics_matrix = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristic = 0
                if demands[i] + demands[j] <= capacity: 
                    x0, y0 = coordinates[i]
                    x1, y1 = coordinates[j]
                    distance = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                    heuristic = 1 / distance
                    if distance > 0:
                        heuristic *= demands[j] / (distance * capacity)
                heuristics_matrix[i, j] = heuristic
    return heuristics_matrix

def heuristics_v2(distance_matrix, coordinates, demands, capacity, num_samples=100):
    n = len(demands)
    heuristics_matrices = [heuristics(distance_matrix, coordinates, demands, capacity) for _ in range(num_samples)]
    heuristics_matrix = np.mean(heuristics_matrices, axis=0)
    return heuristics_matrix
