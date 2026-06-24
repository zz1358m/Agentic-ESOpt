# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_m1m2_sigma1e-3_alpha5e-4_rep3_m1m2_fixed_20260608_0731/results/pops_best/population_generation_25.json
# objective: 5.7981
# algorithm: The "Modified Heuristic" algorithm modifies the Improved Heuristic by further penalizing solutions with long paths and rewarding consecutive paths, and its implementation is based on the provided algorithm, but with the score function having different parameter settings: the base probability is now 0.2, the distance penalty coefficient is doubled and the reward coefficient for consecutive edges is halved, with the absolute difference in indices of nodes being raised to the power of 2 instead of 1 and having half the power.

import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        total_distance = np.sum(distance_matrix[i])
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = (0.2 / (distance_matrix[i, j]**3 + 1/total_distance * distance_matrix[i,j])**2 
                                         + 0.05 / (abs(i - j) + 1)**2 
                                         + (0.1**0.8)/(1 + distance_matrix[i, j])**0.4
                                         + 0.25/(1 + (abs(i-j) + n - 1)**0.5))
    return heuristics_matrix
