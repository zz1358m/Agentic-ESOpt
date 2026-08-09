# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 9.18996
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np
from scipy.spatial import distance

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i!= j:
                euclidean_distance = distance.euclidean(coordinates[i], coordinates[j])
                if i == 0:
                    cost_effectiveness = demands[j] * capacity
                else:
                    cost_effectiveness = demands[j] / distance_matrix[i, j]
                cost_weight = cost_effectiveness / (demands[j] + 1)
                distance_weight = 1 / euclidean_distance**2
                heuristics_matrix[i, j] = cost_weight * distance_weight
                
    # Normalization
    max_score = heuristics_matrix.max()
    if max_score == 0:
        heuristics_matrix.fill(1)
    else:
        heuristics_matrix = heuristics_matrix / max_score
    
    return heuristics_matrix
