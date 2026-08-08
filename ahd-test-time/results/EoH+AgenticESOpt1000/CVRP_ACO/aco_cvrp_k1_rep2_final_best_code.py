# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 8.76927
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0003705904774487396

import numpy as np
from scipy.spatial import distance

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == j or np.isnan(distance_matrix[i, j]):
                heuristics_matrix[i, j] = 0
                continue
            
            # Calculate edge length and edge disparity ratio
            d_ij = distance_matrix[i, j]
            d_i0 = distance.euclidean(coordinates[i], coordinates[0])
            d_j0 = distance.euclidean(coordinates[j], coordinates[0])
            edge_disparity_ratio = d_i0 + d_j0 - d_ij
            
            # Calculate node disparity ratio (proportional to node demand and distance)
            node_disparity_ratio = demands[i] * d_i0 + demands[j] * d_j0
            
            # Calculate the score
            if demands[i] <= capacity - demands[j]:
                score = (node_disparity_ratio / d_i0) * (1 / d_ij) * np.exp(edge_disparity_ratio / d_ij)
            else:
                score = (node_disparity_ratio / d_i0) * (1 / d_ij) * np.exp(edge_disparity_ratio / d_ij) * (demands[i] / capacity - 1)
            
            heuristics_matrix[i, j] = score
    
    return heuristics_matrix
