# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: 5.82996
# m1m2_multiplier: 3.0

import numpy as np

def heuristics_v2(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    sorted_matrix = np.sort(distance_matrix, axis=1)
    for i in range(num_nodes):
        sorted_matrix[i] = sorted_matrix[i][sorted_matrix[i]!= 0]
    
    heuristics = np.zeros((num_nodes, num_nodes))
    alpha = 0.2  # scaling factor
    beta = 0.8  # scaling factor
    gamma = 2.0  # scaling factor
    delta = 0.3  # new scaling factor for uncertainty term
    for i in range(num_nodes):
        min_dist = np.min(distance_matrix[i])
        max_dist = np.max(distance_matrix[i])
        for j in range(num_nodes):
            if i!= j:
                dist = distance_matrix[i, j]
                if dist == 0:
                    rank = 1
                else:
                    rank = np.where(sorted_matrix[i] == dist)[0][0] + 1
                mirrored_rank = np.where(sorted_matrix[j] == dist)[0][0] + 1
                dist_norm = dist / (min_dist + max_dist)
                mean_dist = np.mean(sorted_matrix[i])
                std_dev = np.std(sorted_matrix[i])
                agentic_esopt_term = 1 / (1 + np.exp((dist - mean_dist) / std_dev))
                uncertainty_term = 0.5 * (std_dev * (rank / (rank + mirrored_rank)))
                heuristics[i, j] = (1 / np.sqrt(dist_norm)) * (1 / (rank ** beta)) * (1 / (mirrored_rank ** beta)) * alpha * agentic_esopt_term * (1 - uncertainty_term) ** gamma * (1 + delta * uncertainty_term)
    return heuristics
