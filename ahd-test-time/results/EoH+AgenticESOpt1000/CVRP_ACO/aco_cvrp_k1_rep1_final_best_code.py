# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 9.07758
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.00043473690388997434

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                if i == 0 or j == 0 or (demands[i] > capacity or demands[j] > capacity or demands[i] + demands[j] > 2 * capacity):
                    heuristics_matrix[i, j] = 0
                else:
                    exp_distance_score = np.exp(1 / distance_matrix[i, j])
                    capacity_probability = np.minimum(demands[i] / capacity, demands[j] / capacity)
                    proximity_inverse_score = 1 / (np.abs(coordinates[i, 0] - coordinates[j, 0]) + np.abs(coordinates[i, 1] - coordinates[j, 1]))
                    total_demand_ratio = (demands[i] + demands[j]) / (2 * capacity)
                    max_ratio = np.maximum(demands[i], demands[j]) / capacity
                    min_ratio = np.minimum(demands[i], demands[j]) / capacity
                    penalty = max_ratio / (min_ratio + 1e-9)
                    reciprocal_demand_ratio = 1 / (1 + total_demand_ratio)
                    risk_factor = np.power(np.maximum(0, 1 - (capacity - (demands[i] + demands[j])) / capacity), 2)
                    heuristics_matrix[i, j] = capacity_probability * exp_distance_score * proximity_inverse_score * reciprocal_demand_ratio * (penalty + 0.5 * risk_factor)
                    heuristics_matrix[j, i] = heuristics_matrix[i, j]
    return heuristics_matrix
