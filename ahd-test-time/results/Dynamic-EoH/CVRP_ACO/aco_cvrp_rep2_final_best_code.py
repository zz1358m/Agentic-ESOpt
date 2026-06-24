# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_m1m2_sigma1e-3_alpha5e-4_rep2_m1m2_fixed_rerun_cvrp_only_20260610_044756/results/pops_best/population_generation_25.json

import numpy as np
from math import sqrt

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        neighbors = np.delete(np.arange(n), i)
        avg_demands = np.mean(demands[neighbors])
        
        if avg_demands > 0 and capacity > 0:
            entropy = np.std(demands[neighbors]) + (avg_demands / (capacity + 1)) ** 2
            entropy_decay = (entropy + 1) / (entropy + 1 + (avg_demands / (capacity + 1)) ** 2)
            for j in range(n):
                if i == j:
                    heuristics_matrix[i, j] = 0
                elif i == 0 or j == 0 or distance_matrix[i, j] == 0:
                    heuristics_matrix[i, j] = 0
                else:
                    dist = sqrt((coordinates[i, 0] - coordinates[j, 0])**2 + (coordinates[i, 1] - coordinates[j, 1])**2)
                    growth_rate = 1 + capacity / (capacity + avg_demands)
                    score = dist**(-growth_rate * avg_demands) * (1 + growth_rate) * (1 + (entropy * entropy_decay) ** 2) * (dist + avg_demands)
                    heuristics_matrix[i, j] = score
        else:
            heuristics_matrix[i, i] = 1
    
    return heuristics_matrix
