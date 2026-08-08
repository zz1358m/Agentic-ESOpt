# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 5.80381
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    out_degrees = np.sum(distance_matrix, axis=0)
    avg_out_degree = np.mean(out_degrees)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics_matrix[i, j] = 0
            elif distance_matrix[i, j] == 0:
                heuristics_matrix[i, j] = 0
            else:
                relative_out_degree_i = out_degrees[i] / avg_out_degree if out_degrees[i]!= 0 else 0
                relative_out_degree_j = out_degrees[j] / avg_out_degree if out_degrees[j]!= 0 else 0
                heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] ** (2 + 2.2 * relative_out_degree_i * relative_out_degree_j))

    return heuristics_matrix
