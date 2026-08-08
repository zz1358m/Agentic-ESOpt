# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 8.919
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0005

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or i == 0 or j == 0:
                heuristics_matrix[i, j] = 0
            else:
                # Calculate the score of the edge
                distance_ratio = (distance_matrix[i, j] / np.max(distance_matrix))
                demand_ratio = (min(demands[i], demands[j]) / capacity)
                vehicle_utilization_ratio = 1 / (1 + demands[i] / (capacity * 0.9)) * 1 / (1 + demands[j] / (capacity * 0.9))
                score = (distance_ratio ** -3 + (0.8 ** demand_ratio) + (0.8 ** vehicle_utilization_ratio)) * (distance_matrix[i, j] ** -8)
                if demand_ratio > 0.9:
                    score *= 0.5
                heuristics_matrix[i, j] = score
    return heuristics_matrix
