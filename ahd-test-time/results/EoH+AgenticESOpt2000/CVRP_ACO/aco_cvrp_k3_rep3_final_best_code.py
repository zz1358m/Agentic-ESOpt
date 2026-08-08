# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 8.79621
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0

import numpy as np
from scipy.spatial import distance

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics = np.zeros_like(distance_matrix)

    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics[i, j] = 0
            elif i!= 0 and j!= 0:
                edge_length = distance.euclidean(coordinates[i], coordinates[j])**2 / 10
                fraction_remaining_capacity = (capacity - (demands[i] + demands[j])) / capacity
                capacity_utilization_ratio = 1 / ((capacity - (demands[i] + demands[j])) / capacity + 1)
                
                importance = np.exp(-edge_length) * (capacity_utilization_ratio ** 3)
                if fraction_remaining_capacity > 0.7:
                    importance *= 1.5  
                
                distance_to_depot = distance.euclidean(coordinates[0], coordinates[i])**2 + distance.euclidean(coordinates[j], coordinates[0])**2
                if capacity_utilization_ratio > 0:
                    return_power = (distance_to_depot / (distance.euclidean(coordinates[i], coordinates[j])**2))**3
                    importance *= (return_power / (capacity_utilization_ratio + 1))
                
                heuristics[i, j] = importance
    
    return heuristics
