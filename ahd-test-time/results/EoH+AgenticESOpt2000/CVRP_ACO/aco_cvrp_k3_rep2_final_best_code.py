# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_cvrp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 8.97431
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.00019561928549563967

import numpy as np

def heuristics_v3(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    
    def calculate_heuristics(i, j):
        dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
        
        if demands[i] <= capacity:
            capacity_factor = 1
        else:
            capacity_factor = (demands[i] / capacity)**3
        
        if dist == 0:
            dist_factor = 1
        else:
            dist_factor = 1 / (dist**5)
        
        if i == j or i == 0 or j == 0:
            risk_factor = 0
        else:
            sorted_demands = np.sort(demands[demands > 0])
            subset_demands = demands[i] + np.sum(sorted_demands[sorted_demands > 0])
            subset_capacity = demands[i] + np.sum(demands <= capacity)
            variance = np.var(sorted_demands)
            median_demand = np.median(sorted_demands)
            deviation = np.mean(np.abs(sorted_demands - median_demand))
            largest_demand = np.max(sorted_demands)
            max_variation = variance / median_demand
            num_large_demands = np.sum(sorted_demands > 0.7 * subset_capacity)
            risk_factor = (np.mean(subset_demands / subset_capacity) * (1 + deviation / median_demand) + 0.4 * num_large_demands + 0.3 * max_variation)
        
        return (dist_factor / demands[i])**1.5 * (1 / (demands[i] * capacity_factor * (1 + dist / (distance_matrix[0, 0] * 2))) + 0.35) + risk_factor * 0.15
    
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0:
                heuristics_matrix[i, j] = 0
            elif j == 0:
                heuristics_matrix[i, j] = float('inf')
            else:
                heuristics_matrix[i, j] = calculate_heuristics(i, j)
    
    return heuristics_matrix
